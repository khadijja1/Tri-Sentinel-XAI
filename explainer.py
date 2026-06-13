from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from torch import nn
from lime.lime_text import LimeTextExplainer
from PIL import Image
from transformers import AutoImageProcessor

from modules import deepfake_image_analyzer as image_module
from modules import nlp_analyzer as text_module
from modules import url_analyzer as url_module


BASE_DIR = Path(__file__).resolve().parent
IMAGE_MODEL_NAME = "dima806/deepfake_vs_real_image_detection"


# ── Utilities ─────────────────────────────────────────────────────

def _to_dense_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_image(image_input: str | Image.Image) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    return Image.open(image_input).convert("RGB")


def _clean_feature_name(name: str) -> str:
    return name.split("__", 1)[-1].replace("_", " ")


def _directional_contributions(
    feature_names: Sequence[str],
    values: np.ndarray,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = [
        {
            "feature": _clean_feature_name(feature_names[index]),
            "value": float(values[index]),
        }
        for index in np.argsort(np.abs(values))[::-1]
    ]
    positive = [r for r in records if r["value"] > 0][:top_k]
    negative = [r for r in records if r["value"] < 0][:top_k]
    return records[:top_k], positive, negative


def _summarize_contributions(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    positive_label: str,
    negative_label: str,
) -> str:
    parts: list[str] = []
    if positive:
        parts.append(
            f"{positive_label}: "
            + ", ".join(f"{i['feature']} ({i['value']:+.3f})" for i in positive[:3])
        )
    if negative:
        parts.append(
            f"{negative_label}: "
            + ", ".join(f"{i['feature']} ({i['value']:+.3f})" for i in negative[:3])
        )
    return " | ".join(parts) if parts else "No strong feature-level signal was identified."


# ── NLP helpers ───────────────────────────────────────────────────

def _resolve_text_class_names() -> list[str]:
    id2label = getattr(text_module.classifier.model.config, "id2label", {}) or {}
    if id2label:
        return [str(id2label[i]).replace("_", " ").title() for i in sorted(id2label)]
    return ["Legitimate", "Spam"]


def _text_probability_matrix(texts: Sequence[str]) -> np.ndarray:
    classifier = text_module.classifier
    tokenizer  = classifier.tokenizer
    model      = classifier.model
    device     = next(model.parameters()).device
    encoded = tokenizer(
        list(texts), truncation=True, max_length=512,
        padding=True, return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probs  = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy().astype(np.float32)


def _url_free_nlp_predict_proba(texts: Sequence[str]) -> np.ndarray:
    return _text_probability_matrix(texts)


# ── SHAP for URL ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _url_shap_explainer() -> shap.TreeExplainer:
    # FIX: model_output="probability" conflicts with the default
    # feature_perturbation="tree_path_dependent" in shap 0.44+.
    # Solution: switch to feature_perturbation="interventional" which
    # IS compatible with model_output="probability".
    # We pass a small background sample so SHAP has a reference
    # distribution — using transformed zeros as a neutral baseline.
    n_features = len(url_module.preprocessor.get_feature_names_out(url_module.feature_cols))
    background = np.zeros((1, n_features), dtype=np.float32)
    return shap.TreeExplainer(
        url_module.model,
        data=background,
        model_output="probability",
        feature_perturbation="interventional",  # compatible with probability output
    )


# ── LIME for NLP ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _lime_text_explainer() -> LimeTextExplainer:
    return LimeTextExplainer(class_names=_resolve_text_class_names(), random_state=42)


# ── Image processor ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _image_processor() -> Any:
    processor = getattr(image_module.get_classifier(), "image_processor", None)
    if processor is not None:
        return processor
    return AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)


# ── ViT wrapper for Grad-CAM ──────────────────────────────────────

class _LogitsOnlyImageModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config     = base_model.config
        self.vit        = base_model.vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.base_model(pixel_values=pixel_values).logits


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    model      = image_module.get_classifier().model
    image_size = int(getattr(model.config, "image_size", 224))
    patch_size = int(getattr(model.config, "patch_size", 16))
    h = max(1, image_size // patch_size)
    w = max(1, image_size // patch_size)
    result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def _resolve_image_target_layer(model: nn.Module) -> nn.Module:
    try:
        return model.vit.encoder.layer[-1].layernorm_before
    except AttributeError as exc:
        raise RuntimeError("Unsupported image model architecture for Grad-CAM.") from exc


# ── Pure-PyTorch Grad-CAM (no opencv / libGL dependency) ─────────
# Replaces pytorch_grad_cam entirely.
# Uses gradient-weighted class activation mapping computed
# manually — identical math, zero system library dependencies.

def _pure_pytorch_gradcam(
    model: nn.Module,
    pixel_values: torch.Tensor,
    target_index: int,
    target_layer: nn.Module,
    reshape_transform,
) -> np.ndarray:
    """
    Pure PyTorch Grad-CAM implementation.
    Eliminates the pytorch_grad_cam → cv2 → libGL.so.1 dependency chain
    that caused Streamlit Cloud cold-start crashes.
    """
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        output = model(pixel_values=pixel_values)
        score  = output[:, target_index].sum()
        model.zero_grad()
        score.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if not activations or not gradients:
        # Fallback: return blank heatmap if hooks didn't fire
        return np.zeros((224, 224), dtype=np.float32)

    act  = reshape_transform(activations[0])   # (1, C, H, W)
    grad = reshape_transform(gradients[0])      # (1, C, H, W)

    # Global average pool the gradients over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam     = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam     = torch.relu(cam)

    return cam.squeeze().cpu().numpy().astype(np.float32)


def _overlay_heatmap(image: Image.Image, cam: np.ndarray) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap on the original image using pure PIL + numpy.
    No opencv required.
    """
    # Resize cam to match image
    h, w = image.size[1], image.size[0]
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_tensor, size=(h, w), mode="bilinear", align_corners=False)
    cam_arr = cam_resized.squeeze().numpy()

    # Normalize 0-1
    cam_max = cam_arr.max()
    if cam_max > 0:
        cam_arr = cam_arr / cam_max

    r = np.clip(1.5 - np.abs(4.0 * cam_arr - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * cam_arr - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * cam_arr - 1.0), 0, 1)
    heatmap_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB").resize(image.size, Image.BILINEAR)

    # Blend original image with heatmap (alpha=0.5)
    img_arr     = np.asarray(image).astype(np.float32)
    heat_arr    = np.asarray(heatmap_img).astype(np.float32)
    blended     = np.clip(0.5 * img_arr + 0.5 * heat_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


# ── Public explain functions ──────────────────────────────────────

def explain_url(url: str, top_k: int = 10) -> dict[str, Any]:
    raw_features       = url_module.extract_features_from_url(url)
    transformed        = _to_dense_array(url_module.preprocessor.transform(raw_features))
    feature_names      = list(url_module.preprocessor.get_feature_names_out(url_module.feature_cols))

    explainer          = _url_shap_explainer()
    explanation        = explainer(transformed, check_additivity=False)
    shap_values        = np.asarray(explanation.values, dtype=np.float32).reshape(-1)

    # model_output="probability" returns values for class 1 (legitimate).
    # Negate: positive = pushes toward phishing, negative = toward legitimate.
    shap_array = -shap_values

    contributions, positive_contributions, negative_contributions = (
        _directional_contributions(feature_names, shap_array, top_k)
    )
    prediction = url_module.predict_url(url)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])

    return {
        "url":                    url,
        "prediction":             prediction,
        "probabilities": {
            "legit_prob":         float(prediction["legit_prob"]),
            "phishing_prob":      float(prediction["phishing_prob"]),
        },
        "base_value":             base_value,
        "feature_contributions":  contributions,
        "positive_contributions": positive_contributions,
        "negative_contributions": negative_contributions,
        "summary": _summarize_contributions(
            positive_contributions,
            negative_contributions,
            "Signals pushing toward phishing",
            "Signals pushing toward legitimacy",
        ),
        "transformed_feature_names": feature_names,
    }


def explain_text(text: str, num_features: int = 10, num_samples: int = 500) -> dict[str, Any]:
    lime_explainer = _lime_text_explainer()
    explanation    = lime_explainer.explain_instance(
        text_instance=text,
        classifier_fn=_url_free_nlp_predict_proba,
        num_features=num_features,
        top_labels=1,
        num_samples=num_samples,
    )
    probabilities  = _text_probability_matrix([text])[0]
    label_index    = explanation.top_labels[0] if explanation.top_labels else 0
    class_names    = _resolve_text_class_names()
    label_name     = class_names[label_index] if label_index < len(class_names) else str(label_index)

    contributions  = [
        {"feature": feature, "value": float(weight)}
        for feature, weight in explanation.as_list(label=label_index)
    ]
    positive_contributions = [i for i in contributions if i["value"] > 0]
    negative_contributions = [i for i in contributions if i["value"] < 0]

    return {
        "text":                   text,
        "prediction":             text_module.predict_text(text),
        "label_index":            int(label_index),
        "label_name":             label_name,
        "feature_contributions":  contributions,
        "positive_contributions": positive_contributions,
        "negative_contributions": negative_contributions,
        "summary": _summarize_contributions(
            positive_contributions,
            negative_contributions,
            f"Words supporting {label_name.lower()}",
            "Words arguing against it",
        ),
        "probabilities": {
            "legit_prob": round(float(probabilities[0]) * 100, 2),
            "spam_prob":  round(float(probabilities[1]) * 100, 2),
        },
    }


def explain_image(image_input: str | Image.Image) -> dict[str, Any]:
    image      = _load_image(image_input)
    classifier = image_module.get_classifier()
    processor  = _image_processor()
    model      = _LogitsOnlyImageModel(classifier.model)
    device     = next(classifier.model.parameters()).device
    model.to(device)

    inputs       = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    model.eval()
    with torch.no_grad():
        logits        = model(pixel_values=pixel_values)
        probabilities = torch.softmax(logits, dim=-1)[0]

    target_index = int(torch.argmax(probabilities).item())
    target_layer = _resolve_image_target_layer(model)

    
    cam = _pure_pytorch_gradcam(
        model=model,
        pixel_values=pixel_values,
        target_index=target_index,
        target_layer=target_layer,
        reshape_transform=_vit_reshape_transform,
    )

    overlay_image  = _overlay_heatmap(image, cam)
    predicted_label = model.config.id2label.get(target_index, str(target_index))

    return {
        "image": image,
        "prediction": {
            "label":      predicted_label,
            "confidence": round(float(probabilities[target_index]) * 100, 2),
            "real_prob":  round(float(probabilities[0]) * 100, 2),
            "fake_prob":  round(float(probabilities[1]) * 100, 2),
        },
        "heatmap":      overlay_image,
        "target_layer": "vit.encoder.layer[-1].layernorm_before",
        "summary":      f"Grad-CAM highlights the regions most responsible for predicting {predicted_label}.",
    }


__all__ = ["explain_url", "explain_text", "explain_image"]
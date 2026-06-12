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
URL_DATASET_PATH = BASE_DIR / "datasets" / "PhiUSIIL_Phishing_URL_Dataset.csv"
IMAGE_MODEL_NAME = "dima806/deepfake_vs_real_image_detection"


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
    positive = [record for record in records if record["value"] > 0][:top_k]
    negative = [record for record in records if record["value"] < 0][:top_k]
    return records[:top_k], positive, negative


def _summarize_contributions(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    positive_label: str,
    negative_label: str,
) -> str:
    summary_parts: list[str] = []
    if positive:
        summary_parts.append(
            f"{positive_label}: "
            + ", ".join(f"{item['feature']} ({item['value']:+.3f})" for item in positive[:3])
        )
    if negative:
        summary_parts.append(
            f"{negative_label}: "
            + ", ".join(f"{item['feature']} ({item['value']:+.3f})" for item in negative[:3])
        )
    return " | ".join(summary_parts) if summary_parts else "No strong feature-level signal was identified."


def _resolve_text_class_names() -> list[str]:
    id2label = getattr(text_module.classifier.model.config, "id2label", {}) or {}
    if id2label:
        return [str(id2label[index]).replace("_", " ").title() for index in sorted(id2label)]
    return ["Legitimate", "Spam"]


def _text_spam_label_index() -> int:
    id2label = getattr(text_module.classifier.model.config, "id2label", {}) or {}
    for index, label in sorted(id2label.items()):
        normalized = str(label).strip().lower()
        if normalized in {"spam", "spam_label", "label_1", "positive"}:
            return int(index)
    if id2label:
        return int(sorted(id2label)[-1])
    return 1


def _text_probability_matrix(texts: Sequence[str]) -> np.ndarray:
    classifier = text_module.classifier
    tokenizer = classifier.tokenizer
    model = classifier.model
    device = next(model.parameters()).device

    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.detach().cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _url_shap_explainer() -> shap.TreeExplainer:
    # model_output="probability" makes SHAP work in probability space
    # and eliminates the need for a background dataset CSV entirely.
    # Without a background dataset the previous fallback (all-zeros array)
    # was inverting the SHAP reference point on Streamlit Cloud — this fixes that.
    return shap.TreeExplainer(
        url_module.model,
        model_output="probability",
    )


@lru_cache(maxsize=1)
def _lime_text_explainer() -> LimeTextExplainer:
    return LimeTextExplainer(class_names=_resolve_text_class_names(), random_state=42)


@lru_cache(maxsize=1)
def _image_processor() -> Any:
    processor = getattr(image_module.get_classifier(), "image_processor", None)
    if processor is not None:
        return processor
    return AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)


class _LogitsOnlyImageModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.vit = base_model.vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.base_model(pixel_values=pixel_values).logits


def _url_predict_proba(urls: Sequence[str]) -> np.ndarray:
    probabilities: list[list[float]] = []
    for url in urls:
        transformed = _to_dense_array(url_module.preprocessor.transform(url_module.extract_features_from_url(url)))
        raw_probabilities = url_module.model.predict_proba(transformed)[0]
        probabilities.append([float(raw_probabilities[0]), float(raw_probabilities[1])])
    return np.asarray(probabilities, dtype=np.float32)


def explain_url(url: str, top_k: int = 10) -> dict[str, Any]:
    raw_features = url_module.extract_features_from_url(url)
    transformed_features = _to_dense_array(
        url_module.preprocessor.transform(raw_features)
    )
    feature_names = list(
        url_module.preprocessor.get_feature_names_out(url_module.feature_cols)
    )
 
    explainer = _url_shap_explainer()
    explanation = explainer(transformed_features, check_additivity=False)
    shap_values = np.asarray(explanation.values, dtype=np.float32).reshape(-1)
 
    # XGBoost binary model with model_output="probability":
    # SHAP returns values for class 1 (legitimate).
    # Negate so that positive values = push toward phishing,
    # negative values = push toward legitimate.
    # This aligns with how predict_url defines label 0 = phishing.
    shap_array = -shap_values
 
    contributions, positive_contributions, negative_contributions = (
        _directional_contributions(feature_names, shap_array, top_k)
    )
 
    prediction = url_module.predict_url(url)
 
    # expected_value is a scalar for model_output="probability"
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
 
    return {
        "url": url,
        "prediction": prediction,
        "probabilities": {
            "legit_prob": float(prediction["legit_prob"]),
            "phishing_prob": float(prediction["phishing_prob"]),
        },
        "base_value": base_value,
        "feature_contributions": contributions,
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


def explain_text(text: str, num_features: int = 10, num_samples: int = 250) -> dict[str, Any]:
    lime_explainer = _lime_text_explainer()
    num_samples = max(num_samples, 500)
    explanation = lime_explainer.explain_instance(
        text_instance=text,
        classifier_fn=_url_free_nlp_predict_proba,
        num_features=num_features,
        top_labels=1,
        num_samples=num_samples,
    )

    probabilities = _text_probability_matrix([text])[0]
    label_index = explanation.top_labels[0] if explanation.top_labels else explanation.available_labels()[0]
    label_name = _resolve_text_class_names()[label_index] if label_index < len(_resolve_text_class_names()) else str(label_index)
    contributions = [
        {
            'feature': feature,
            'value': float(weight),
        }
        for feature, weight in explanation.as_list(label=label_index)
    ]
    positive_contributions = [item for item in contributions if item['value'] > 0]
    negative_contributions = [item for item in contributions if item['value'] < 0]
    return {
        'text': text,
        'prediction': text_module.predict_text(text),
        'label_index': int(label_index),
        'label_name': label_name,
        'feature_contributions': contributions,
        'positive_contributions': positive_contributions,
        'negative_contributions': negative_contributions,
        'summary': _summarize_contributions(
            positive_contributions,
            negative_contributions,
            f'Words supporting {label_name.lower()}',
            'Words arguing against it',
        ),
        'probabilities': {
            'legit_prob': round(float(probabilities[0]) * 100, 2),
            'spam_prob': round(float(probabilities[1]) * 100, 2),
        },
    }


def _url_free_nlp_predict_proba(texts: Sequence[str]) -> np.ndarray:
    return _text_probability_matrix(texts)


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    model = image_module.get_classifier().model
    image_size = int(getattr(model.config, 'image_size', 224))
    patch_size = int(getattr(model.config, 'patch_size', 16))
    height = max(1, image_size // patch_size)
    width = max(1, image_size // patch_size)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def _resolve_image_target_layer(model: nn.Module) -> nn.Module:
    try:
        return model.vit.encoder.layer[-1].layernorm_before
    except AttributeError as exc:
        raise RuntimeError('Unsupported image model architecture for Grad-CAM.') from exc


def explain_image(image_input: str | Image.Image) -> dict[str, Any]:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as e:
        return {
            'image': _load_image(image_input),
            'prediction': {},
            'heatmap': _load_image(image_input),
            'target_layer': 'unavailable',
            'summary': f'Grad-CAM unavailable on this load — please refresh. ({e})',
        }
    image = _load_image(image_input)
    classifier = image_module.get_classifier()
    processor = _image_processor()
    model = _LogitsOnlyImageModel(classifier.model)
    device = next(classifier.model.parameters()).device
    model.to(device)

    inputs = processor(images=image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=pixel_values)
        probabilities = torch.softmax(logits, dim=-1)[0]

    target_index = int(torch.argmax(probabilities).item())
    target_layer = _resolve_image_target_layer(model)
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=_vit_reshape_transform,
    )

    grayscale_cam = cam(
        input_tensor=pixel_values,
        targets=[ClassifierOutputTarget(target_index)],
        aug_smooth=True,
        eigen_smooth=True,
    )[0]

    # Resize CAM to match the original image size before overlaying.
    cam_arr = np.asarray(grayscale_cam)
    if cam_arr.ndim > 2:
        cam_arr = np.mean(cam_arr, axis=-1)

    target_w, target_h = image.size
    if (cam_arr.shape[0], cam_arr.shape[1]) != (target_h, target_w):
        cam_tensor = torch.from_numpy(cam_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        cam_tensor = F.interpolate(cam_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
        cam_arr = cam_tensor.squeeze(0).squeeze(0).cpu().numpy()

    cam_arr = cam_arr.astype(np.float32)
    cam_max = cam_arr.max() if cam_arr.size else 0.0
    if cam_max > 0:
        cam_arr = cam_arr / cam_max

    image_array = np.asarray(image).astype(np.float32) / 255.0
    overlay = show_cam_on_image(image_array, cam_arr, use_rgb=True)
    overlay_image = Image.fromarray(overlay)

    predicted_label = model.config.id2label.get(target_index, str(target_index))

    return {
        'image': image,
        'prediction': {
            'label': predicted_label,
            'confidence': round(float(probabilities[target_index]) * 100, 2),
            'real_prob': round(float(probabilities[0]) * 100, 2),
            'fake_prob': round(float(probabilities[1]) * 100, 2),
        },
        'heatmap': overlay_image,
        'target_layer': 'vit.encoder.layer[-1].layernorm_before',
        'summary': f"Grad-CAM highlights the regions most responsible for the model predicting {predicted_label}.",
    }


__all__ = [
    'explain_url',
    'explain_text',
    'explain_image',
]
from functools import lru_cache
from typing import Any

from PIL import Image
from transformers import pipeline


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline(
        "image-classification",
        model="dima806/deepfake_vs_real_image_detection"
    )

def predict_image(image_input) -> dict:
    classifier = get_classifier()
    
    # Accepts both a file path (local testing) and PIL image (Streamlit)
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")
    
    # Resize to model's expected image size to ensure consistent behavior
    try:
        model = classifier.model
        image_size = int(getattr(model.config, 'image_size', 224))
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    except Exception:
        # If resizing fails for any reason, continue with original image
        pass
    
    raw_results = classifier(image) or []
    results: list[Any] = list(raw_results)
    scores = {str(result.get('label', '')).strip().lower(): float(result.get('score', 0.0)) for result in results}
    fake_prob = scores.get('fake', scores.get('deepfake', 0.0))
    real_prob = scores.get('real', 0.0)

    if not real_prob and results:
        top_label = str(results[0].get('label', '')).strip().lower()
        if top_label not in {'fake', 'deepfake', 'real'}:
            real_prob = max(0.0, 1.0 - fake_prob)

    return {
        'label'      : 'Deepfake' if fake_prob > 0.5 else 'Real',
        'confidence' : round(float(max(fake_prob, real_prob)) * 100, 2),
        'fake_prob'  : round(float(fake_prob) * 100, 2),
        'real_prob'  : round(float(real_prob) * 100, 2),
    }

# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    test_images = [
        r'C:\Users\User\Documents\XAI_Social_Engineering_Detection__System\test_images\real_person (2).png',    # ← replace henerycavil.png
        r'C:\Users\User\Documents\XAI_Social_Engineering_Detection__System\test_images\stablediffusion.png'    # ← replace image_.png
    ]
    for path in test_images:
        result = predict_image(path)
        flag   = '🔴' if result['label'] == 'Deepfake' else '🟢'
        print(f"\n{flag} Image      : {path}")
        print(f"   Prediction : {result['label']}")
        print(f"   Confidence : {result['confidence']}%")
        print(f"   Fake: {result['fake_prob']}%  |  Real: {result['real_prob']}%")
        print("-" * 60)
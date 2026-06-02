from transformers import pipeline

# HuggingFace downloads and caches the model automatically
# on first run (~440MB, cached locally after that)
classifier = pipeline(
    "text-classification",
    model="cybert79/spamai",
    tokenizer="cybert79/spamai"
)

def predict_text(text: str) -> dict:
    prediction = classifier(text, truncation=True, max_length=512)[0]
    label = str(prediction['label']).strip().lower()
    score = float(prediction['score'])

    # Normalize so we always have both probabilities
    spam_prob = score if label in {'label_1', 'spam', 'spam_label'} else 1 - score
    legit_prob = 1 - spam_prob

    return {
        'text'       : text[:80] + '...' if len(text) > 80 else text,
        'label'      : 'Spam' if spam_prob > 0.5 else 'Legitimate',
        'confidence' : round(float(max(spam_prob, legit_prob)) * 100, 2),
        'spam_prob'  : round(float(spam_prob) * 100, 2),
        'legit_prob' : round(float(legit_prob) * 100, 2),
    }

# ── Quick test ────────────────────────────────────────────────────
if __name__ == '__main__':
    test_messages = [
        "Hey, are we still on for the meeting tomorrow at 3pm?",
        "URGENT: Your bank account has been suspended. Click here to verify now.",
        "Congratulations! You have won a $1000 gift card. Reply WIN to claim.",
        "Can you send me the report by end of day? Thanks!",]

    for msg in test_messages:
        result = predict_text(msg)
        print(f"\nText       : {result['text']}")
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']}%")
        print(f"Spam: {result['spam_prob']}%  |  Legit: {result['legit_prob']}%")
        print("-" * 60)
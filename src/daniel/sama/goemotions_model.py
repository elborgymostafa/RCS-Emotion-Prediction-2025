import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm

model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Load your file
with open("train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Prediction and rewrite
predicted_data = []
emotion_counter = Counter()

for entry in tqdm(data, desc="Predicting emotions"):
    review = entry["input"]
    outputs = []
    for aspect_data in entry["output"]:
        aspect = aspect_data["aspect"]
        joint_text = f"Aspect: {aspect}. Review: {review}"
        inputs = tokenizer(joint_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            top_idx = torch.argmax(probs, dim=1).item()
            emotion = labels[top_idx]

        aspect_data["emotion"] = emotion
        emotion_counter[emotion] += 1
        outputs.append(aspect_data)

    predicted_data.append({
        "input": review,
        "output": outputs
    })

# Save to new JSONL
with open("annotation_300_with_emotions.jsonl", "w", encoding="utf-8") as f:
    for entry in predicted_data:
        json.dump(entry, f)
        f.write("\n")

# Print summary
print("\nEmotion Distribution:")
for emotion, count in emotion_counter.most_common():
    print(f"{emotion}: {count}")
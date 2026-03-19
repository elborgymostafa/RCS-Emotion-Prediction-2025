import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score

# -----------------------------
# CONSTANTS
# -----------------------------

ASPECT_LIST = [
    "food", "service", "ambience", "price",
    "miscellaneous", "menu", "staff", "place"
]

EMOTION_MAP = {
    "satisfaction": 0,
    "admiration": 1,
    "gratitude": 2,
    "disappointment": 3,
    "annoyance": 4,
    "disgust": 5,
    "neutral": 6,
    "mentioned_only": 7,
    "mixed_emotions": 8
}

POLARITY_MAP = {
    "positive": 0,
    "negative": 1,
    "neutral": 2
}

IDX2POL = {v: k for k, v in POLARITY_MAP.items()}
NUM_POLARITIES = len(POLARITY_MAP)

IDX2EMO = {v: k for k, v in EMOTION_MAP.items()}

NUM_ASPECTS = len(ASPECT_LIST)
NUM_EMOTIONS = len(EMOTION_MAP)

# -----------------------------
# JSONL LOADER
# -----------------------------

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["input"]

            emo_labels = [-100] * NUM_ASPECTS
            pol_labels = [-100] * NUM_ASPECTS
            mask = [0] * NUM_ASPECTS
            present_aspects = []

            for out in item.get("output", []):
                asp = out["aspect"].lower()
                emo = out["emotion"].lower()
                pol = out["polarity"].lower()

                if (
                    asp in ASPECT_LIST and
                    emo in EMOTION_MAP and
                    pol in POLARITY_MAP
                ):
                    idx = ASPECT_LIST.index(asp)
                    emo_labels[idx] = EMOTION_MAP[emo]
                    pol_labels[idx] = POLARITY_MAP[pol]
                    mask[idx] = 1
                    present_aspects.append(asp)

            data.append({
                "text": text,
                "emo_labels": emo_labels,
                "pol_labels": pol_labels,
                "mask": mask,
                "present_aspects": present_aspects
            })
    return data


# -----------------------------
# DATASET
# -----------------------------

class RestaurantDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        enc = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emo_labels": torch.tensor(item["emo_labels"], dtype=torch.long),
            "pol_labels": torch.tensor(item["pol_labels"], dtype=torch.long),
            "mask": torch.tensor(item["mask"], dtype=torch.float),
            "present_aspects": item["present_aspects"]
        }

# -----------------------------
# COLLATE FUNCTION
# -----------------------------

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "emo_labels": torch.stack([x["emo_labels"] for x in batch]),
        "pol_labels": torch.stack([x["pol_labels"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
        "present_aspects": [x["present_aspects"] for x in batch]
    }

# -----------------------------
# MODEL (DistilBERT)
# -----------------------------

class MultiHeadDistilBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.emo_heads = nn.ModuleList([
            nn.Linear(768, NUM_EMOTIONS) for _ in range(NUM_ASPECTS)
        ])

        self.pol_heads = nn.ModuleList([
            nn.Linear(768, NUM_POLARITIES) for _ in range(NUM_ASPECTS)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:, 0, :]  # CLS

        emo_logits = torch.stack(
            [h(pooled) for h in self.emo_heads],
            dim=1
        )  # [B, A, E]

        pol_logits = torch.stack(
            [h(pooled) for h in self.pol_heads],
            dim=1
        )  # [B, A, P]

        return emo_logits, pol_logits


# -----------------------------
# TRAINING + EMOTION F1
# -----------------------------

def train_model(train_data, val_data, epochs=10, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    train_loader = DataLoader(
        RestaurantDataset(train_data, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        RestaurantDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    model = MultiHeadDistilBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            emo_labels = batch["emo_labels"].to(device)
            pol_labels = batch["pol_labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            emo_logits, pol_logits = model(ids, att)

            loss = 0.0
            for i in range(NUM_ASPECTS):
                emo_loss = criterion(emo_logits[:, i, :], emo_labels[:, i])
                pol_loss = criterion(pol_logits[:, i, :], pol_labels[:, i])

                loss += (
                    (emo_loss + pol_loss) * mask[:, i]
                ).sum() / (mask[:, i].sum() + 1e-9)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch+1} | Train Loss: {total_loss:.4f}")

        # -------- Validation: Emotion F1 (Macro + Weighted) --------
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                att = batch["attention_mask"].to(device)
                emo_labels = batch["emo_labels"].to(device)
                mask = batch["mask"].to(device)

                emo_logits, _ = model(ids, att)
                preds = torch.argmax(emo_logits, dim=2)  # [B, A]

                for i in range(NUM_ASPECTS):
                    valid = mask[:, i] == 1
                    all_preds.extend(preds[valid, i].cpu().tolist())
                    all_labels.extend(emo_labels[valid, i].cpu().tolist())

        # Macro F1 (treats all classes equally)
        emotion_macro_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )

        # Weighted F1 (accounts for class imbalance)
        emotion_weighted_f1 = f1_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0
        )

        print(
            f"Epoch {epoch+1} | "
            f"Emotion Macro-F1: {emotion_macro_f1:.4f} | "
            f"Emotion Weighted-F1: {emotion_weighted_f1:.4f}"
        )


# -----------------------------
# PREDICTION ON NEW JSONL
# -----------------------------

def predict_jsonl(model, tokenizer, input_jsonl, output_jsonl):
    device = next(model.parameters()).device
    model.eval()

    data = load_jsonl(input_jsonl)
    outputs = []

    with torch.no_grad():
        for item in data:
            text = item["text"]
            present_aspects = item["present_aspects"]

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            emo_logits, pol_logits = model(enc["input_ids"], enc["attention_mask"])

            emo_preds = torch.argmax(emo_logits, dim=2)[0]
            pol_preds = torch.argmax(pol_logits, dim=2)[0]

            aspect_outputs = []
            for asp in present_aspects:
                idx = ASPECT_LIST.index(asp)
                aspect_outputs.append({
                    "aspect": asp,
                    "emotion": IDX2EMO[emo_preds[idx].item()],
                    "polarity": IDX2POL[pol_preds[idx].item()]
                })

            outputs.append({
                "input": text,
                "output": aspect_outputs
            })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    train_data = load_jsonl("train_final.jsonl")
    val_data = load_jsonl("validationdata.jsonl")

    model, tokenizer = train_model(
        train_data,
        val_data,
        epochs=10
    )

    predict_jsonl(
        model,
        tokenizer,
        input_jsonl="edited_300_sample_cleaned_14jan.jsonl",
        output_jsonl="predictions_all.jsonl"
    )

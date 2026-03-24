import json
import os
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file, load_file

# =============================
# CONFIG
# =============================

REMOVE_ASPECT = "miscellaneous"

ASPECT_LIST = [
    "food", "service", "ambience", "price",
    "menu", "staff", "place"
]  # 7 aspects (miscellaneous removed)

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

IDX2EMO = {v: k for k, v in EMOTION_MAP.items()}
IDX2POL = {v: k for k, v in POLARITY_MAP.items()}

NUM_ASPECTS = len(ASPECT_LIST)
NUM_EMOTIONS = len(EMOTION_MAP)
NUM_POLARITIES = len(POLARITY_MAP)

BASE_MODEL_NAME = "bert-base-uncased"

CKPT_DIR = "checkpoints_bert_no_misc"
MODEL_FILE = f"{CKPT_DIR}/best_model.safetensors"

# =============================
# JSONL FILTER + LOADER
# =============================

def filter_jsonl_remove_aspect(
    in_path: str,
    out_path: str,
    remove_aspect: str = REMOVE_ASPECT
) -> None:
    """
    Writes a new JSONL without any output entries where aspect == remove_aspect.
    If a line becomes empty (no outputs left), it is dropped.
    """
    kept = 0
    dropped = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            outputs = item.get("output", [])

            new_outputs = [
                o for o in outputs
                if o.get("aspect", "").lower() != remove_aspect.lower()
            ]

            # drop reviews that have no labeled aspects left
            if len(new_outputs) == 0:
                dropped += 1
                continue

            item["output"] = new_outputs
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[filter_jsonl] Wrote: {out_path} | kept={kept} | dropped={dropped}")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
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

                if asp in ASPECT_LIST and emo in EMOTION_MAP and pol in POLARITY_MAP:
                    idx = ASPECT_LIST.index(asp)
                    emo_labels[idx] = EMOTION_MAP[emo]
                    pol_labels[idx] = POLARITY_MAP[pol]
                    mask[idx] = 1
                    present_aspects.append(asp)

            # Drop examples that somehow end up with no present aspects
            if sum(mask) == 0:
                continue

            data.append({
                "text": text,
                "emo_labels": emo_labels,
                "pol_labels": pol_labels,
                "mask": mask,
                "present_aspects": present_aspects
            })
    return data

# =============================
# DATASET + COLLATE
# =============================

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
            "present_aspects": item["present_aspects"]  # variable length
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "emo_labels": torch.stack([x["emo_labels"] for x in batch]),
        "pol_labels": torch.stack([x["pol_labels"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
        "present_aspects": [x["present_aspects"] for x in batch]
    }

# =============================
# MODEL (BERT-base)
# =============================

class MultiHeadBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL_NAME)
        hidden = self.encoder.config.hidden_size  # 768 for bert-base

        self.emo_heads = nn.ModuleList([nn.Linear(hidden, NUM_EMOTIONS) for _ in range(NUM_ASPECTS)])
        self.pol_heads = nn.ModuleList([nn.Linear(hidden, NUM_POLARITIES) for _ in range(NUM_ASPECTS)])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # CLS

        emo_logits = torch.stack([h(pooled) for h in self.emo_heads], dim=1)  # [B, A, E]
        pol_logits = torch.stack([h(pooled) for h in self.pol_heads], dim=1)  # [B, A, P]
        return emo_logits, pol_logits

# =============================
# TRAIN + EVAL (Emotion F1 only)
# =============================

def evaluate_emotion_f1(model, val_loader, device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            emo_labels = batch["emo_labels"].to(device)
            mask = batch["mask"].to(device)

            emo_logits, _ = model(ids, att)
            preds = torch.argmax(emo_logits, dim=2)

            for i in range(NUM_ASPECTS):
                valid = mask[:, i] == 1
                all_preds.extend(preds[valid, i].cpu().tolist())
                all_labels.extend(emo_labels[valid, i].cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return macro_f1, weighted_f1


def train_model(
    train_data,
    val_data,
    max_epochs=10,
    batch_size=8,
    lr=2e-5,
    patience=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

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

    model = MultiHeadBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_weighted_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            emo_labels = batch["emo_labels"].to(device)
            pol_labels = batch["pol_labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            emo_logits, pol_logits = model(ids, att)

            loss = 0.0
            for i in range(NUM_ASPECTS):
                e_loss = criterion(emo_logits[:, i], emo_labels[:, i])
                p_loss = criterion(pol_logits[:, i], pol_labels[:, i])

                # only for aspects present
                loss += ((e_loss + p_loss) * mask[:, i]).sum() / (mask[:, i].sum() + 1e-9)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        macro_f1, weighted_f1 = evaluate_emotion_f1(model, val_loader, device)

        print(
            f"\nEpoch {epoch} | Train Loss: {avg_loss:.4f} | "
            f"Val Emotion Macro-F1: {macro_f1:.4f} | "
            f"Val Emotion Weighted-F1: {weighted_f1:.4f}"
        )

        # Early stopping based on WEIGHTED F1 (your target)
        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            patience_counter = 0

            save_file(model.state_dict(), MODEL_FILE)
            tokenizer.save_pretrained(CKPT_DIR)
            print("Best model saved (SafeTensors)")

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print(f"\n🏆 Best Val Emotion Weighted-F1: {best_weighted_f1:.4f}")
    return best_weighted_f1

# =============================
# LOAD BEST MODEL (SAFE)
# =============================

def load_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)

    model = MultiHeadBERT().to(device)
    state_dict = load_file(MODEL_FILE, device=str(device))
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, device

# =============================
# PREDICT JSONL (only present aspects)
# =============================

# =============================
# PREDICT JSONL
# - Emotion: predicted
# - Polarity: copied from input JSONL
# =============================

# =============================
# PREDICT JSONL
# - Emotion: predicted
# - Polarity: copied from raw test JSONL
# =============================

def load_raw_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def predict_jsonl(model, tokenizer, device, input_jsonl, output_jsonl):
    processed = load_jsonl(input_jsonl)      # has present_aspects
    raw = load_raw_jsonl(input_jsonl)         # has original polarity

    outputs = []

    with torch.no_grad():
        for proc_item, raw_item in zip(processed, raw):
            enc = tokenizer(
                proc_item["text"],
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            emo_logits, _ = model(enc["input_ids"], enc["attention_mask"])
            emo_preds = torch.argmax(emo_logits, dim=2)[0]

            # Build aspect → polarity map from RAW json
            polarity_map = {
                o["aspect"].lower(): o["polarity"]
                for o in raw_item["output"]
            }

            aspect_outputs = []
            for asp in proc_item["present_aspects"]:
                idx = ASPECT_LIST.index(asp)

                aspect_outputs.append({
                    "aspect": asp,
                    "polarity": polarity_map[asp],
                    "emotion": IDX2EMO[emo_preds[idx].item()] 
                })

            outputs.append({
                "input": proc_item["text"],
                "output": aspect_outputs
            })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"[predict_jsonl] Wrote emotion-only predictions (polarity preserved) to: {output_jsonl}")


# =============================
# MAIN
# =============================

if __name__ == "__main__":
    '''
    filter_jsonl_remove_aspect("train_final.jsonl", "train_no_misc.jsonl", REMOVE_ASPECT)
    filter_jsonl_remove_aspect("validationdata.jsonl", "val_no_misc.jsonl", REMOVE_ASPECT)
    filter_jsonl_remove_aspect("edited_300_sample_cleaned_14jan.jsonl", "test_no_misc.jsonl", REMOVE_ASPECT)

    # ---- Step 2: load filtered data ----
    train_data = load_jsonl("train_no_misc.jsonl")
    val_data = load_jsonl("val_no_misc.jsonl")

    # ---- Step 3: train ----
    train_model(
        train_data,
        val_data,
        max_epochs=10,
        batch_size=8,
        lr=2e-5,
        patience=3
    )
    '''
    # ---- Step 4: load best and predict on test ----
    model, tokenizer, device = load_best_model()
    predict_jsonl(
        model,
        tokenizer,
        device,
        input_jsonl="testdata.jsonl",
        output_jsonl="predictions_BERT.jsonl"
    )

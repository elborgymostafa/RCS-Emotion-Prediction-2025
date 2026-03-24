#!/usr/bin/env python
# coding: utf-8
"""
classifier_v3_test_eval.py

Identical to classifier_v3.py with one addition:
after training completes, load test.jsonl, apply the same label merging
and LabelEncoder, then run a full test-set evaluation, saving
confusion matrix PNG + classification report CSV + test_metrics.txt
into the run directory.

V3 label merging:
  {gratitude, disgust, mixed_emotions, mentioned_only} -> neutral
Resulting 5-class label space:
  {admiration, annoyance, disappointment, neutral, satisfaction}
"""

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_MARKERS = ("src", "data")

def find_project_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    while True:
        if all(os.path.isdir(os.path.join(current, m)) for m in PROJECT_MARKERS):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Project root not found. Expected dirs: {PROJECT_MARKERS}")
        current = parent

cwd = os.getcwd()
try:
    start_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    start_path = cwd

project_root = find_project_root(start_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT      = Path(project_root)
data_root = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "annotated")

print(f"Project root : {project_root}")
print(f"Data root    : {data_root}")

# ─── Run directory ───────────────────────────────────────────────────────────
EXPERIMENT_NAME = Path(__file__).stem  # classifier_v3_test_eval

BASE_RESULTS_DIR = ROOT / "results_v2" / EXPERIMENT_NAME
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

existing_runs = [p for p in BASE_RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]
run_ids = []
for p in existing_runs:
    try:
        run_ids.append(int(p.name.split("_")[1]))
    except Exception:
        pass

RUN_ID = (max(run_ids) + 1) if run_ids else 1
RUN_DIR = BASE_RESULTS_DIR / f"run_{RUN_ID:03d}"
RUN_DIR.mkdir(parents=True, exist_ok=False)
RESULTS_ROOT = RUN_DIR

print(f"Starting {EXPERIMENT_NAME}, run {RUN_ID:03d}")
print(f"Run directory: {RUN_DIR}")

(RUN_DIR / "run_info.txt").write_text(
    f"experiment={EXPERIMENT_NAME}\n"
    f"run_id={RUN_ID:03d}\n"
    f"timestamp={datetime.now().isoformat(timespec='seconds')}\n",
    encoding="utf-8"
)

EVAL_CALL = 0

# ─── V3 label merging map ────────────────────────────────────────────────────
MERGE_TO_NEUTRAL = {
    "gratitude":      "neutral",
    "disgust":        "neutral",
    "mixed_emotions": "neutral",
    "mentioned_only": "neutral",
}
ALLOWED_LABELS = {"admiration", "annoyance", "disappointment", "neutral", "satisfaction"}

# ─── Imports ─────────────────────────────────────────────────────────────────
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

# ─── Helper ──────────────────────────────────────────────────────────────────
def load_and_merge(path: Path) -> pd.DataFrame:
    """Load JSONL, explode to rows, apply V3 label merging."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    records = []
    for r in rows:
        text = r["input"]
        for o in r["output"]:
            records.append({
                "text":     text,
                "aspect":   o["aspect"],
                "polarity": o["polarity"],
                "emotion":  o["emotion"],
            })
    df = pd.DataFrame(records)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["emotion"] = df["emotion"].replace(MERGE_TO_NEUTRAL)
    extra = set(df["emotion"].unique()) - ALLOWED_LABELS
    if extra:
        raise ValueError(f"Unexpected labels after merge: {sorted(extra)}")
    return df

# ─── 1) Load train.jsonl and apply merging ───────────────────────────────────
df = load_and_merge(Path(data_root) / "train.jsonl")

print("\nEmotion distribution after V3 merge (train.jsonl):")
print(df["emotion"].value_counts(dropna=False))

# ─── 2) Label encoding ───────────────────────────────────────────────────────
emotion_encoder  = LabelEncoder()
polarity_encoder = LabelEncoder()

df["emotion_id"]  = emotion_encoder.fit_transform(df["emotion"])
df["polarity_id"] = polarity_encoder.fit_transform(df["polarity"])

num_emotions = len(emotion_encoder.classes_)
num_polarity = len(polarity_encoder.classes_)
EMOTION_NAMES = list(emotion_encoder.classes_)

print("\nEmotion labels (V3):", EMOTION_NAMES)

# ─── 3) Train / val split (same 80/20, same seed as V3) ─────────────────────
unique_texts = df["text"].unique()
train_texts, val_texts = train_test_split(unique_texts, test_size=0.2, random_state=42)

train_df = df[df["text"].isin(train_texts)]
val_df   = df[df["text"].isin(val_texts)]

# ─── 4) Load test.jsonl with V3 merging ──────────────────────────────────────
test_df = load_and_merge(Path(data_root) / "test.jsonl")
test_df["emotion_id"]  = emotion_encoder.transform(test_df["emotion"])
test_df["polarity_id"] = polarity_encoder.transform(test_df["polarity"])

print("\nTest emotion distribution (after V3 merge):")
print(test_df["emotion"].value_counts(dropna=False))

# ─── 5) Tokenizer + Dataset ──────────────────────────────────────────────────
MODEL_NAME = "distilroberta-base"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.loc[idx]
        text = f"ASPECT: {row['aspect']} | TEXT: {row['text']}"
        enc  = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids":       enc["input_ids"].squeeze(0),
            "attention_mask":  enc["attention_mask"].squeeze(0),
            "emotion_labels":  torch.tensor(row["emotion_id"]),
            "polarity_labels": torch.tensor(row["polarity_id"]),
        }

train_ds = EmotionDataset(train_df, tokenizer)
val_ds   = EmotionDataset(val_df,   tokenizer)
test_ds  = EmotionDataset(test_df,  tokenizer)

# ─── 6) Model ────────────────────────────────────────────────────────────────
class EmotionPolarityModel(torch.nn.Module):
    def __init__(self, model_name, num_emotions, num_polarity):
        super().__init__()
        self.encoder       = AutoModel.from_pretrained(model_name)
        hidden             = self.encoder.config.hidden_size
        self.emotion_head  = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(self, input_ids, attention_mask, emotion_labels=None, polarity_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        emotion_logits  = self.emotion_head(cls)
        polarity_logits = self.polarity_head(cls)
        loss = None
        if emotion_labels is not None:
            loss_e = torch.nn.functional.cross_entropy(emotion_logits,  emotion_labels)
            loss_p = torch.nn.functional.cross_entropy(polarity_logits, polarity_labels)
            loss   = loss_e + 0.3 * loss_p
        return {"loss": loss, "emotion_logits": emotion_logits, "polarity_logits": polarity_logits}

model = EmotionPolarityModel(MODEL_NAME, num_emotions=num_emotions, num_polarity=num_polarity)

# ─── 7) Metrics (per-epoch) ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    global EVAL_CALL
    logits, y_true = eval_pred
    y_pred     = np.argmax(logits, axis=1)
    num_classes = len(EMOTION_NAMES)
    all_labels  = list(range(num_classes))

    cm    = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"G_{n}" for n in EMOTION_NAMES],
        columns=[f"P_{n}" for n in EMOTION_NAMES]
    )

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Oranges",
                     linewidths=0.5, linecolor="gray", square=True,
                     cbar_kws={"label": "Count"})
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    plt.setp(ax.get_yticklabels(), rotation=0)
    for i in range(num_classes):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))
    plt.title("Emotion Confusion Matrix (Counts) — V3", pad=30)
    plt.xlabel("Predicted Emotion"); plt.ylabel("Gold Emotion")
    plt.tight_layout()

    EVAL_CALL += 1
    epoch_id = EVAL_CALL

    plt.savefig(RESULTS_ROOT / f"confusion_matrix_emotion_epoch_{epoch_id}.png", dpi=300)
    plt.close()

    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, target_names=EMOTION_NAMES,
        digits=4, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T
    report_df = report_df[["precision", "recall", "f1-score", "support"]]
    report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(4)
    report_df["support"] = report_df["support"].astype(int)
    report_df.index.name = "label"
    report_df.to_csv(RESULTS_ROOT / f"classification_report_epoch_{epoch_id}.csv")

    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    print(f"\n[Epoch {epoch_id}] macro_f1 = {macro_f1:.4f}")
    return {"macro_f1": macro_f1}

# ─── 8) Custom Trainer ───────────────────────────────────────────────────────
class EmotionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            emotion_labels=inputs["emotion_labels"],
            polarity_labels=inputs["polarity_labels"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        logits = outputs["emotion_logits"]
        labels = inputs.get("emotion_labels")
        if prediction_loss_only:
            return (None, None, None)
        return (None, logits, labels)

# ─── 9) Train ────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=str(RESULTS_ROOT),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_steps=50,
    report_to="none",
)

trainer = EmotionTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ─── 10) TEST SET EVALUATION ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST SET EVALUATION (test.jsonl — V3 label space)")
print("=" * 50)

test_output = trainer.predict(test_ds)
test_logits = test_output.predictions
y_test_pred = np.argmax(test_logits, axis=1)
y_test_true = test_df["emotion_id"].values

all_labels = list(range(len(EMOTION_NAMES)))

macro_f1    = f1_score(y_test_true, y_test_pred, labels=all_labels, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_test_true, y_test_pred, labels=all_labels, average="weighted", zero_division=0)
micro_f1    = f1_score(y_test_true, y_test_pred, labels=all_labels, average="micro",    zero_division=0)

print(f"macro_f1    = {macro_f1:.4f}")
print(f"weighted_f1 = {weighted_f1:.4f}")
print(f"micro_f1    = {micro_f1:.4f}")

(RESULTS_ROOT / "test_metrics.txt").write_text(
    f"TEST metrics (V3 — 5-class, label-merged)\n"
    f"macro_f1    = {macro_f1:.4f}\n"
    f"weighted_f1 = {weighted_f1:.4f}\n"
    f"micro_f1    = {micro_f1:.4f}\n",
    encoding="utf-8"
)

report_dict = classification_report(
    y_test_true, y_test_pred, labels=all_labels, target_names=EMOTION_NAMES,
    digits=4, zero_division=0, output_dict=True
)
report_df = pd.DataFrame(report_dict).T
report_df = report_df[["precision", "recall", "f1-score", "support"]]
report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(4)
report_df["support"] = report_df["support"].astype(int)
report_df.index.name = "label"
report_df.to_csv(RESULTS_ROOT / "test_classification_report.csv")

print("\nClassification Report (test — V3 label space):")
print(report_df)

# Confusion matrix
cm    = confusion_matrix(y_test_true, y_test_pred, labels=all_labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"G_{n}" for n in EMOTION_NAMES],
    columns=[f"P_{n}" for n in EMOTION_NAMES]
)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Oranges",
                 linewidths=0.5, linecolor="gray", square=True,
                 cbar_kws={"label": "Count"})
ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
plt.setp(ax.get_yticklabels(), rotation=0)
for i in range(len(EMOTION_NAMES)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))
plt.title("Emotion Confusion Matrix — TEST SET (V3, 5-class)", pad=30)
plt.xlabel("Predicted Emotion"); plt.ylabel("Gold Emotion")
plt.tight_layout()
plt.savefig(RESULTS_ROOT / "test_confusion_matrix_emotion.png", dpi=300)
plt.close()

print(f"\nSaved to: {RESULTS_ROOT}")
print("Done.")

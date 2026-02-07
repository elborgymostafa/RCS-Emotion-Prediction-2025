#!/usr/bin/env python
# coding: utf-8

"""
classifier_v3.py (DATAFRAME-ONLY label fix)

Goal:
- Improve macro-F1 by fixing the label space *only via dataframe preprocessing*,
  without changing the Trainer/model training logic.

Key issue in your report:
- Some classes have 0–5 samples in validation (e.g., gratitude=0, disgust=5, mixed_emotions=5).
  Those classes will have near-zero F1 and drag down macro-F1.
- "mentioned_only" is effectively "no emotion expressed for the aspect" → behaves like neutral.

v3 approach (df-only):
- Normalize emotion strings
- Merge unlearnable / ambiguous labels into "neutral" BEFORE LabelEncoder
- Make EMOTION_NAMES consistent with LabelEncoder classes (important!)
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import os
import sys
from pathlib import Path
from datetime import datetime

# -----------------------------
# 1) Find project root (robust)
# -----------------------------
# Only require the stable anchors.
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

# Add project root to sys.path (optional, only if you import your own modules)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = Path(project_root)
print(f"Project root found at: {project_root}")

# -----------------------------
# 2) Define your important dirs
# -----------------------------
src_root     = os.path.join(project_root, "src", "daniel", "model")
data_root    = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "annotated")
schemas_root = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "schema")
prompts_root = os.path.join(project_root, "prompts", "daniel", "llama")  # ok if missing, only used if you need it
utils_root   = os.path.join(project_root, "utils")

print(
    f"cwd          : {cwd}\n"
    f"Project root : {project_root}\n"
    f"Source root  : {src_root}\n"
    f"Data root    : {data_root}\n"
    f"Prompts root : {prompts_root}\n"
    f"Utils root   : {utils_root}\n"
)

# -----------------------------
# 3) Run output directory (no overwrites)
# -----------------------------
EXPERIMENT_NAME = Path(__file__).stem  # e.g., classifier_v1 / classifier_v3

BASE_RESULTS_DIR = ROOT / "results" / EXPERIMENT_NAME
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

RESULTS_ROOT = RUN_DIR  # <-- use this everywhere (Trainer output_dir + PNG + CSV)

print(f"Starting {EXPERIMENT_NAME}, run {RUN_ID:03d}")
print(f"Run directory: {RUN_DIR}")


(RUN_DIR / "run_info.txt").write_text(
    f"experiment={EXPERIMENT_NAME}\n"
    f"run_id={RUN_ID:03d}\n"
    f"timestamp={datetime.now().isoformat(timespec='seconds')}\n",
    encoding="utf-8"
)

# -----------------------------
# 4) Metrics epoch counter
# -----------------------------
EVAL_CALL = 0

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)

# ===============================
# 1) LOAD JSONL
# ===============================
data_file = Path(data_root) / "train.jsonl"  # adjust if needed

rows = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

print("Loaded rows:", len(rows))
from transformers import TrainerCallback


def light_check(rows, name):
    for i, r in enumerate(rows):
        assert isinstance(r, dict), f"{name}[{i}] is not a dict"

        assert "input" in r, f"{name}[{i}] missing 'input'"
        assert isinstance(r["input"], str), f"{name}[{i}]['input'] not a string"
        assert r["input"].strip(), f"{name}[{i}] empty 'input'"

        assert "output" in r, f"{name}[{i}] missing 'output'"
        assert isinstance(r["output"], list), f"{name}[{i}]['output'] not a list"
        assert len(r["output"]) > 0, f"{name}[{i}] empty 'output' list"

        for j, o in enumerate(r["output"]):
            assert isinstance(o, dict), f"{name}[{i}]['output'][{j}] not a dict"
            for k in ("aspect", "polarity", "emotion"):
                assert k in o, f"{name}[{i}]['output'][{j}] missing '{k}'"
                assert isinstance(o[k], str), f"{name}[{i}]['output'][{j}]['{k}'] not a string"
                assert o[k].strip(), f"{name}[{i}]['output'][{j}] empty '{k}'"

    print(f"{name}: {len(rows)} rows passed")

light_check(rows=rows, name="train")

# ===============================
# 2) EXPAND ROWS TO TABULAR DF
# ===============================
records = []
for r in rows:
    text = r["input"]
    for o in r["output"]:
        records.append({
            "text": text,
            "aspect": o["aspect"],
            "polarity": o["polarity"],
            "emotion": o["emotion"],
        })

df = pd.DataFrame(records)
print("\nRaw emotion distribution (BEFORE v3):")
print(df["emotion"].value_counts(dropna=False))

# ===============================
# 3) v3 DATAFRAME-ONLY LABEL FIX
#    (THIS IS THE ONLY "F1 IMPROVEMENT" CHANGE)
# ===============================

# 3.1 Normalize emotion strings (defensive)
df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

# 3.2 Merge unlearnable / ambiguous classes to neutral
# - gratitude: often absent in val (support=0 in your report)
# - disgust, mixed_emotions: extremely rare (support=5 each in your report)
# - mentioned_only: essentially "aspect mentioned without emotion" → behaves like neutral
MERGE_TO_NEUTRAL = {
    "gratitude": "neutral",
    "disgust": "neutral",
    "mixed_emotions": "neutral",
    "mentioned_only": "neutral",
}
df["emotion"] = df["emotion"].replace(MERGE_TO_NEUTRAL)

print("\n[v3] Emotion distribution (AFTER merge):")
print(df["emotion"].value_counts(dropna=False))

# Optional strict check: ensure we only have the intended label set
allowed = {"admiration", "annoyance", "disappointment", "neutral", "satisfaction"}
extra = set(df["emotion"].unique()) - allowed
if extra:
    raise ValueError(f"[v3] Unexpected emotion labels found: {sorted(extra)}")

# ===============================
# 4) ENCODE LABELS
# ===============================
emotion_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

df["emotion_id"]  = emotion_encoder.fit_transform(df["emotion"])
df["polarity_id"] = polarity_encoder.fit_transform(df["polarity"])

num_emotions = len(emotion_encoder.classes_)
num_polarity = len(polarity_encoder.classes_)

print("\nEmotion labels (after v3):")
for i, label in enumerate(emotion_encoder.classes_):
    print(f"{i} → {label}")

print("\nPolarity labels:")
for i, label in enumerate(polarity_encoder.classes_):
    print(f"{i} → {label}")

# IMPORTANT:
# Use the encoder classes as the canonical label names.
# This keeps EMOTION_NAMES aligned with emotion_id values after the dataframe merge.
EMOTION_NAMES = list(emotion_encoder.classes_)

# ===============================
# 5) TRAIN/VAL SPLIT BY UNIQUE TEXT
# ===============================
unique_texts = df["text"].unique()
train_texts, val_texts = train_test_split(
    unique_texts,
    test_size=0.2,
    random_state=42
)

train_df = df[df["text"].isin(train_texts)]
val_df   = df[df["text"].isin(val_texts)]

print("\nTrain emotion distribution:")
print(train_df["emotion"].value_counts())

print("\nVal emotion distribution:")
print(val_df["emotion"].value_counts())

# ===============================
# 6) TOKENIZER
# ===============================
MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===============================
# 7) DATASET
# ===============================
class EmotionDataset(Dataset):
    """
    Dataset for aspect-conditioned emotion classification (primary)
    + polarity classification (auxiliary).

    Input is constructed as:
      ASPECT: <aspect> | TEXT: <text>
    """
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = f"ASPECT: {row['aspect']} | TEXT: {row['text']}"

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emotion_labels": torch.tensor(row["emotion_id"]),
            "polarity_labels": torch.tensor(row["polarity_id"])
        }

# ===============================
# 8) MODEL
# ===============================
class EmotionPolarityModel(torch.nn.Module):
    """
    Multi-task model:
    - emotion head (primary)
    - polarity head (aux)
    """
    def __init__(self, model_name, num_emotions, num_polarity):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(self, input_ids, attention_mask, emotion_labels=None, polarity_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]

        emotion_logits = self.emotion_head(cls)
        polarity_logits = self.polarity_head(cls)

        loss = None
        if emotion_labels is not None:
            loss_e = torch.nn.functional.cross_entropy(emotion_logits, emotion_labels)
            loss_p = torch.nn.functional.cross_entropy(polarity_logits, polarity_labels)
            loss = loss_e + 0.3 * loss_p

        return {"loss": loss, "emotion_logits": emotion_logits, "polarity_logits": polarity_logits}

# ===============================
# 9) METRICS
# ===============================
def compute_metrics(eval_pred):
    """
    Trainer metrics:
    - Confusion matrix (counts) saved as PNG (per epoch)
    - Classification report saved as CSV (per epoch)
    - macro_f1 returned for Trainer
    """
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)

    num_classes = len(EMOTION_NAMES)
    all_labels = list(range(num_classes))

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"G_{name}" for name in EMOTION_NAMES],
        columns=[f"P_{name}" for name in EMOTION_NAMES]
    )

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Oranges",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        cbar_kws={"label": "Count"}
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    plt.setp(ax.get_yticklabels(), rotation=0)

    for i in range(num_classes):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2)
        )

    plt.title("Emotion Confusion Matrix (Counts)", pad=30)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Gold Emotion")
    plt.tight_layout()

    # -----------------------------
    # EPOCH COUNTER
    # -----------------------------
    global EVAL_CALL
    EVAL_CALL += 1
    epoch_id = EVAL_CALL

    png_path = RESULTS_ROOT / f"confusion_matrix_emotion_epoch_{epoch_id}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    # -----------------------------
    # CLASSIFICATION REPORT (CSV)
    # -----------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    # Convert to DataFrame
    report_df = pd.DataFrame(report_dict).T

    # Reorder & clean columns
    desired_cols = ["precision", "recall", "f1-score", "support"]
    report_df = report_df[desired_cols]

    # Round metrics (keep support as int-looking)
    report_df[["precision", "recall", "f1-score"]] = (
        report_df[["precision", "recall", "f1-score"]].round(4)
    )
    report_df["support"] = report_df["support"].astype(int)

    # Rename index for clarity
    report_df.index.name = "label"

    csv_path = RESULTS_ROOT / f"classification_report_epoch_{epoch_id}.csv"
    report_df.to_csv(csv_path)

    # -----------------------------
    # PRINT (still useful for logs)
    # -----------------------------
    print("\nClassification Report (epoch {})\n".format(epoch_id))
    print(report_df)
    print(f"Saved confusion matrix → {png_path}")
    print(f"Saved classification report → {csv_path}")

    # -----------------------------
    # SCALAR METRIC (Trainer uses)
    # -----------------------------
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0
    )

    return {"macro_f1": macro_f1}


# ===============================
# 10) CUSTOM TRAINER (unchanged)
# ===============================
class EmotionTrainer(Trainer):
    """
    Custom Trainer that supports joint emotion–polarity training
    while evaluating only emotion predictions.
    """
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

# ===============================
# 11) TRAIN
# ===============================
train_ds = EmotionDataset(train_df, tokenizer)
val_ds   = EmotionDataset(val_df, tokenizer)

model = EmotionPolarityModel(
    MODEL_NAME,
    num_emotions=num_emotions,
    num_polarity=num_polarity
)

args = TrainingArguments(
    output_dir=str(RESULTS_ROOT),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_steps=50,
    report_to="none"
)

trainer = EmotionTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()

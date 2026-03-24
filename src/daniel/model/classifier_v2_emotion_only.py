#!/usr/bin/env python
# coding: utf-8

"""
===========================================================
classifier_v2_emotion_only.py
===========================================================

Goal
----
Train a model that predicts ONLY EMOTION (no polarity head).

Why
---
Your current multi-task setup predicts:
  - emotion (main task)
  - polarity (auxiliary task)

This emotion-only variant lets you run a clean A/B comparison:
  (A) emotion+polarity   vs   (B) emotion-only
and see whether macro-F1 improves.

What stays identical (for fair comparison)
------------------------------------------
- Same dataset (train/val/test jsonl)
- Same input formatting: "ASPECT: ... | TEXT: ..."
- Same tokenizer/model backbone: distilroberta-base
- Same run folder logic: results/<experiment_name>/run_XXX
- Same metrics: confusion matrix PNG + classification report CSV per epoch
- Same TrainingArguments (epochs, batch size, LR, etc.)

What changes
------------
- Dataset returns only "emotion_labels"
- Model has only "emotion_head"
- Loss = cross_entropy(emotion_logits, emotion_labels)
- Trainer passes only emotion labels to the model
"""

# ============================================================
# 0) Imports + project-root discovery (kept same structure)
# ============================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer


# ============================================================
# 1) Find project root (robust)
# ============================================================

# We detect the repo root by searching upward for these anchor folders.
PROJECT_MARKERS = ("src", "data")

def find_project_root(start_path: str) -> str:
    """
    Walk upward from start_path until we find a folder containing PROJECT_MARKERS.
    This makes the script runnable from anywhere inside the repo.
    """
    current = os.path.abspath(start_path)
    while True:
        if all(os.path.isdir(os.path.join(current, m)) for m in PROJECT_MARKERS):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Project root not found. Expected dirs: {PROJECT_MARKERS}")
        current = parent

cwd = os.getcwd()

# __file__ exists when running as a script; in notebooks it may not.
try:
    start_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    start_path = cwd

project_root = find_project_root(start_path)

# Optional: allow importing your own modules relative to repo root.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = Path(project_root)
print(f"Project root found at: {project_root}")


# ============================================================
# 2) Define important directories (same as before)
# ============================================================

src_root     = os.path.join(project_root, "src", "daniel", "model")
data_root    = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "annotated")
schemas_root = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "schema")
prompts_root = os.path.join(project_root, "prompts", "daniel", "llama")  # can be missing, fine
utils_root   = os.path.join(project_root, "utils")

print(
    f"cwd          : {cwd}\n"
    f"Project root : {project_root}\n"
    f"Source root  : {src_root}\n"
    f"Data root    : {data_root}\n"
    f"Prompts root : {prompts_root}\n"
    f"Utils root   : {utils_root}\n"
)


# ============================================================
# 3) Run output directory (no overwrites)
# ============================================================

# IMPORTANT:
# - This script should be saved under a new filename so EXPERIMENT_NAME differs.
# - Example filename: classifier_v2_emotion_only.py
# Then results will go to: results/classifier_v2_emotion_only/run_XXX
EXPERIMENT_NAME = Path(__file__).stem

BASE_RESULTS_DIR = ROOT / "results" / EXPERIMENT_NAME
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Detect existing run_* folders to auto-increment run id
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

RESULTS_ROOT = RUN_DIR  # use everywhere (Trainer output_dir + PNG + CSV)

print(f"Starting {EXPERIMENT_NAME}, run {RUN_ID:03d}")
print(f"Run directory: {RUN_DIR}")

# Save run metadata (nice for reproducibility)
(RUN_DIR / "run_info.txt").write_text(
    f"experiment={EXPERIMENT_NAME}\n"
    f"run_id={RUN_ID:03d}\n"
    f"timestamp={datetime.now().isoformat(timespec='seconds')}\n",
    encoding="utf-8"
)


# ============================================================
# 4) Epoch counter for metric files
# ============================================================

# We increment this every time compute_metrics runs.
# Because eval_strategy="epoch", it should roughly match epoch number.
EVAL_CALL = 0


# ============================================================
# 5) Load JSONL (train/val/test)
# ============================================================

def load_jsonl(path: Path):
    """
    Read a JSONL file (one JSON dict per line).
    Returns: list[dict]
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def light_check_min(rows, name):
    """
    Minimal format check to fail early if data is corrupted.
    """
    for i, r in enumerate(rows):
        assert isinstance(r, dict), f"{name}[{i}] is not a dict"
        assert "input" in r and isinstance(r["input"], str) and r["input"].strip(), f"{name}[{i}] bad/missing input"
        assert "output" in r and isinstance(r["output"], list) and len(r["output"]) > 0, f"{name}[{i}] bad/missing output"
    print(f"{name}: {len(rows)} rows passed minimal check")

train_rows = load_jsonl(Path(data_root) / "train.jsonl")
val_rows   = load_jsonl(Path(data_root) / "val.jsonl")
test_rows  = load_jsonl(Path(data_root) / "test.jsonl")

light_check_min(train_rows, "train")
light_check_min(val_rows, "val")
light_check_min(test_rows, "test")


# ============================================================
# 6) Explode JSONL into a flat DataFrame
# ============================================================

def explode_rows(rows):
    """
    Turns each (text, outputs[]) into multiple rows:
      text, aspect, polarity, emotion

    Even though we won't train polarity, we keep it in the DF
    for analysis/printing distributions and potential debugging.
    """
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
    return pd.DataFrame(records)

train_df = explode_rows(train_rows)
val_df   = explode_rows(val_rows)
test_df  = explode_rows(test_rows)

print("\nTrain emotion distribution:")
print(train_df["emotion"].value_counts(dropna=False))
print("\nVal emotion distribution:")
print(val_df["emotion"].value_counts(dropna=False))
print("\nTest emotion distribution:")
print(test_df["emotion"].value_counts(dropna=False))


# ============================================================
# 7) Label encoding (emotion only for training)
# ============================================================

emotion_encoder = LabelEncoder()

# Fit on TRAIN emotions only (correct methodology)
train_df["emotion_id"] = emotion_encoder.fit_transform(train_df["emotion"])

# Transform val/test using same encoder
val_df["emotion_id"]   = emotion_encoder.transform(val_df["emotion"])
test_df["emotion_id"]  = emotion_encoder.transform(test_df["emotion"])

num_emotions = len(emotion_encoder.classes_)

# IMPORTANT:
# LabelEncoder assigns IDs alphabetically.
# We store names in the SAME order to keep mapping consistent.
EMOTION_NAMES = list(emotion_encoder.classes_)

print("\nEmotion labels (LabelEncoder order):")
for i, name in enumerate(EMOTION_NAMES):
    print(f"{i} -> {name}")


# ============================================================
# 8) Tokenizer
# ============================================================

MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 9) Dataset (emotion-only labels)
# ============================================================

class EmotionOnlyDataset(Dataset):
    """
    Torch dataset returning tensors for:
      - input_ids
      - attention_mask
      - emotion_labels   (single-label classification)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]

        # IMPORTANT: keep exact input formatting you used before
        # so comparisons are fair across experiments.
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
            "emotion_labels": torch.tensor(int(row["emotion_id"]), dtype=torch.long),
        }

train_ds = EmotionOnlyDataset(train_df, tokenizer)
val_ds   = EmotionOnlyDataset(val_df, tokenizer)
test_ds  = EmotionOnlyDataset(test_df, tokenizer)


# ============================================================
# 10) Model (emotion-only head)
# ============================================================

class EmotionOnlyModel(torch.nn.Module):
    """
    Single-task classifier:
      encoder -> emotion_head -> emotion_logits
    """

    def __init__(self, model_name: str, num_emotions: int):
        super().__init__()

        # Pretrained transformer encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        # Hidden size (e.g., 768) depends on the encoder config
        hidden = self.encoder.config.hidden_size

        # One classification head for emotion
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)

    def forward(self, input_ids, attention_mask, emotion_labels=None):
        """
        Forward pass:
          - compute encoder outputs
          - take CLS token embedding
          - map to emotion logits
          - compute loss if labels provided
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use first token representation as pooled sentence embedding
        cls = out.last_hidden_state[:, 0]

        emotion_logits = self.emotion_head(cls)

        loss = None
        if emotion_labels is not None:
            loss = torch.nn.functional.cross_entropy(emotion_logits, emotion_labels)

        return {"loss": loss, "emotion_logits": emotion_logits}

model = EmotionOnlyModel(MODEL_NAME, num_emotions=num_emotions)


# ============================================================
# 11) Metrics (emotion-only)
# ============================================================

def compute_metrics(eval_pred):
    """
    Hugging Face Trainer calls this on evaluation.

    Inputs:
      eval_pred = (logits, labels) because we return logits in prediction_step.

    Outputs:
      dict with scalar metrics for Trainer logs (macro_f1)
    Also saves:
      - confusion_matrix_emotion_epoch_{k}.png
      - classification_report_epoch_{k}.csv
    """
    logits, y_true = eval_pred

    # Pred class = argmax over logits
    y_pred = np.argmax(logits, axis=1)

    num_classes = len(EMOTION_NAMES)
    all_labels = list(range(num_classes))

    # ---- confusion matrix (counts) ----
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

    # Draw bold diagonal boxes to highlight correct predictions
    for i in range(num_classes):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

    plt.title("Emotion Confusion Matrix (Counts)", pad=30)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Gold Emotion")
    plt.tight_layout()

    # ---- epoch counter (increment each eval call) ----
    global EVAL_CALL
    EVAL_CALL += 1
    epoch_id = EVAL_CALL

    png_path = RESULTS_ROOT / f"confusion_matrix_emotion_epoch_{epoch_id}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    # ---- classification report (CSV) ----
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).T
    desired_cols = ["precision", "recall", "f1-score", "support"]
    report_df = report_df[desired_cols]

    report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(4)
    report_df["support"] = report_df["support"].astype(int)
    report_df.index.name = "label"

    csv_path = RESULTS_ROOT / f"classification_report_epoch_{epoch_id}.csv"
    report_df.to_csv(csv_path)

    # Useful console output for logs
    print(f"\nClassification Report (epoch {epoch_id})")
    print(report_df)
    print(f"Saved confusion matrix → {png_path}")
    print(f"Saved classification report → {csv_path}")

    # ---- macro F1 (main training signal) ----
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0
    )

    return {"macro_f1": macro_f1}


# ============================================================
# 12) Custom Trainer (emotion-only)
# ============================================================

class EmotionOnlyTrainer(Trainer):
    """
    We override:
    - compute_loss: because our labels key is "emotion_labels"
    - prediction_step: so evaluation returns emotion logits + gold labels
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            emotion_labels=inputs["emotion_labels"],
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

        # Trainer expects: (loss, logits, labels)
        # We don't return loss here (None) because we focus on logits+labels.
        return (None, logits, labels)


# ============================================================
# 13) TrainingArguments + training
# ============================================================

args = TrainingArguments(
    output_dir=str(RESULTS_ROOT),

    # IMPORTANT:
    # HF uses "evaluation_strategy" (not eval_strategy) on many versions.
    # Your previous script used eval_strategy and worked in your env.
    # We'll set BOTH safely: evaluation_strategy is the official key.
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,

    logging_steps=50,
    report_to="none",
)

trainer = EmotionOnlyTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()


# ============================================================
# 14) Final test evaluation
# ============================================================

print("\n=== FINAL TEST EVALUATION (emotion-only) ===")
trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

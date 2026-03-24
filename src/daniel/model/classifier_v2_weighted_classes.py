#!/usr/bin/env python
# coding: utf-8

"""
DistilBERT + class-balanced emotion loss (ABSA-style emotion + polarity multi-task)

What this script does
- Loads your JSONL (train/val/test) with inputs and per-aspect outputs.
- Explodes each review into (text, aspect, polarity, emotion) rows.
- Encodes labels using LabelEncoder (fit on TRAIN only).
- Fine-tunes a DistilBERT encoder with two heads:
    1) Emotion head (9 classes in your case)
    2) Polarity head (3 classes)
- Adds CLASS WEIGHTS to the emotion CrossEntropyLoss to mitigate severe imbalance.
- Evaluates emotion only (macro-F1) and saves:
    - confusion matrix PNG per eval call
    - classification report CSV per eval call

Why class weights
- Your train counts have extreme tail classes (e.g., gratitude=2, mixed_emotions=12, disgust=30).
- Macro-F1 penalizes rare-class failures strongly.
- Weighted CE increases the gradient impact of rare classes to improve recall.

Important note
- Gratitude (2 samples) is almost impossible to learn reliably; weights help but cannot create signal.
"""

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
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer


# =============================================================================
# 1) Find project root (robust)
# =============================================================================
PROJECT_MARKERS = ("src", "data")

def find_project_root(start_path: str) -> str:
    """
    Walk upward from start_path until we find a directory containing PROJECT_MARKERS.
    This makes the script runnable from anywhere (IDE, terminal, cron, etc.).
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
try:
    start_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    start_path = cwd

project_root = find_project_root(start_path)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = Path(project_root)
print(f"Project root found at: {project_root}")


# =============================================================================
# 2) Paths (adapt if your directory layout changes)
# =============================================================================
src_root     = os.path.join(project_root, "src", "daniel", "model")
data_root    = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "annotated")
schemas_root = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "schema")
prompts_root = os.path.join(project_root, "prompts", "daniel", "llama")
utils_root   = os.path.join(project_root, "utils")

print(
    f"cwd          : {cwd}\n"
    f"Project root : {project_root}\n"
    f"Source root  : {src_root}\n"
    f"Data root    : {data_root}\n"
    f"Prompts root : {prompts_root}\n"
    f"Utils root   : {utils_root}\n"
)


# =============================================================================
# 3) Run output directory (no overwrites)
# =============================================================================
EXPERIMENT_NAME = Path(__file__).stem
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

RESULTS_ROOT = RUN_DIR

print(f"Starting {EXPERIMENT_NAME}, run {RUN_ID:03d}")
print(f"Run directory: {RUN_DIR}")

(RUN_DIR / "run_info.txt").write_text(
    f"experiment={EXPERIMENT_NAME}\n"
    f"run_id={RUN_ID:03d}\n"
    f"timestamp={datetime.now().isoformat(timespec='seconds')}\n",
    encoding="utf-8"
)


# =============================================================================
# 4) Small global counter for naming eval artifacts
# =============================================================================
EVAL_CALL = 0


# =============================================================================
# 5) Load JSONL (train/val/test)
# =============================================================================
def load_jsonl(path: Path):
    """Load a JSONL file into a list of Python dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def light_check_min(rows, name):
    """
    Minimal integrity check:
    - each row is dict
    - 'input' is a non-empty string
    - 'output' is a non-empty list
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


# =============================================================================
# 6) Explode JSON into a flat classification dataframe
# =============================================================================
def explode_rows(rows):
    """
    Each original row contains:
      - input: full review text
      - output: list of {aspect, polarity, emotion}
    We explode to one row per (review, aspect) so the model learns aspect-conditioned emotion.
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


# =============================================================================
# 7) Label encoding (fit on TRAIN, transform val/test)
# =============================================================================
emotion_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

train_df["emotion_id"]  = emotion_encoder.fit_transform(train_df["emotion"])
train_df["polarity_id"] = polarity_encoder.fit_transform(train_df["polarity"])

val_df["emotion_id"]    = emotion_encoder.transform(val_df["emotion"])
val_df["polarity_id"]   = polarity_encoder.transform(val_df["polarity"])
test_df["emotion_id"]   = emotion_encoder.transform(test_df["emotion"])
test_df["polarity_id"]  = polarity_encoder.transform(test_df["polarity"])

num_emotions = len(emotion_encoder.classes_)
num_polarity = len(polarity_encoder.classes_)

# IMPORTANT: this order matches LabelEncoder IDs exactly
EMOTION_NAMES = list(emotion_encoder.classes_)
POLARITY_NAMES = list(polarity_encoder.classes_)

print("\nEmotion labels (LabelEncoder order):")
for i, name in enumerate(EMOTION_NAMES):
    print(f"{i} -> {name}")

print("\nPolarity labels (LabelEncoder order):")
for i, name in enumerate(POLARITY_NAMES):
    print(f"{i} -> {name}")


# =============================================================================
# 8) Compute class weights for emotion (TRAIN only)
# =============================================================================
# Why: Macro-F1 punishes rare-class errors. Balanced CE helps recover tail recall.
y_train = train_df["emotion_id"].to_numpy()

# "balanced" = n_samples / (n_classes * count_c)
emotion_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_emotions),
    y=y_train
).astype(np.float32)

print("\nEmotion class weights (aligned with LabelEncoder IDs):")
for i, w in enumerate(emotion_weights_np):
    print(f"{i:2d} ({EMOTION_NAMES[i]:>15s}) -> {w:.4f}")

# Keep on CPU for now; we will move to the correct device later
emotion_class_weights = torch.tensor(emotion_weights_np, dtype=torch.float32)


# =============================================================================
# 9) Tokenizer (DistilBERT)
# =============================================================================
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =============================================================================
# 10) Dataset
# =============================================================================
class EmotionDataset(Dataset):
    """
    Produces tokenized inputs + two labels:
    - emotion_labels: int in [0, num_emotions)
    - polarity_labels: int in [0, num_polarity)
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]

        # Aspect-conditioned prompt style input
        # (You can later upgrade to sentence-pair encoding: tokenizer(aspect, text))
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
            "polarity_labels": torch.tensor(int(row["polarity_id"]), dtype=torch.long),
        }

train_ds = EmotionDataset(train_df, tokenizer)
val_ds   = EmotionDataset(val_df, tokenizer)
test_ds  = EmotionDataset(test_df, tokenizer)


# =============================================================================
# 11) Model (DistilBERT encoder + 2 classification heads)
# =============================================================================
class EmotionPolarityModel(torch.nn.Module):
    """
    Multi-task classifier:
    - shared DistilBERT encoder
    - emotion head: num_emotions classes
    - polarity head: num_polarity classes

    Class balancing:
    - emotion CE uses per-class weights computed from train distribution.
    """
    def __init__(
        self,
        model_name: str,
        num_emotions: int,
        num_polarity: int,
        emotion_class_weights: torch.Tensor,
        polarity_loss_weight: float = 0.3
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.emotion_head = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

        # Store weights and move them to the correct device during forward
        self.register_buffer("emotion_class_weights", emotion_class_weights)
        self.polarity_loss_weight = float(polarity_loss_weight)

    def forward(self, input_ids, attention_mask, emotion_labels=None, polarity_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT returns last_hidden_state; CLS token is position 0
        cls = out.last_hidden_state[:, 0]

        emotion_logits = self.emotion_head(cls)
        polarity_logits = self.polarity_head(cls)

        loss = None
        if emotion_labels is not None:
            # Emotion loss with class weights (the core imbalance fix)
            loss_e = torch.nn.functional.cross_entropy(
                emotion_logits,
                emotion_labels,
                weight=self.emotion_class_weights
            )

            # Optional auxiliary polarity loss (can help, but can also interfere)
            loss_p = torch.nn.functional.cross_entropy(polarity_logits, polarity_labels)

            loss = loss_e + self.polarity_loss_weight * loss_p

        return {"loss": loss, "emotion_logits": emotion_logits, "polarity_logits": polarity_logits}

model = EmotionPolarityModel(
    MODEL_NAME,
    num_emotions=num_emotions,
    num_polarity=num_polarity,
    emotion_class_weights=emotion_class_weights,
    polarity_loss_weight=0.3,
)


# =============================================================================
# 12) Metrics (emotion only)
# =============================================================================
def compute_metrics(eval_pred):
    """
    Trainer metrics:
    - Saves confusion matrix PNG
    - Saves classification report CSV
    - Returns macro_f1 for Trainer tracking
    """
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)

    num_classes = len(EMOTION_NAMES)
    all_labels = list(range(num_classes))

    # Confusion matrix (fixed labels ensures stable shape even if a class is absent in eval)
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

    # Highlight diagonal (correct predictions)
    for i in range(num_classes):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

    plt.title("Emotion Confusion Matrix (Counts)", pad=30)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Gold Emotion")
    plt.tight_layout()

    global EVAL_CALL
    EVAL_CALL += 1
    epoch_id = EVAL_CALL

    png_path = RESULTS_ROOT / f"confusion_matrix_emotion_epoch_{epoch_id}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

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

    print(f"\nClassification Report (eval call {epoch_id})\n")
    print(report_df)
    print(f"Saved confusion matrix → {png_path}")
    print(f"Saved classification report → {csv_path}")

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_labels,
        average="macro",
        zero_division=0
    )
    return {"macro_f1": macro_f1}


# =============================================================================
# 13) Custom Trainer (emotion eval only)
# =============================================================================
class EmotionTrainer(Trainer):
    """
    - compute_loss: uses both labels (emotion + polarity) for training
    - prediction_step: returns emotion logits + emotion labels for evaluation
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


# =============================================================================
# 14) TrainingArguments + train
# =============================================================================
args = TrainingArguments(
    output_dir=str(RESULTS_ROOT),

    # IMPORTANT: HF key is eval_strategy (not eval_strategy)
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

print("\n=== FINAL TEST EVALUATION ===")
trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

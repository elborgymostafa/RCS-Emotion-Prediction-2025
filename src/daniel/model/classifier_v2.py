#!/usr/bin/env python
# coding: utf-8

"""
Modeling Rationale

Although classical machine learning models such as Logistic Regression or
Support Vector Machines can be trained from scratch for text classification,
they rely on shallow lexical representations (e.g., bag-of-words or TF–IDF)
and therefore lack an explicit notion of context, word order, and semantic
composition.

Emotion classification, and especially aspect-based emotion classification,
is inherently context-dependent. The same sentence may express different
emotions depending on the referenced aspect, and emotional meaning is often
conveyed implicitly through contrast, expectation, or discourse structure
rather than through isolated keywords.

Training a model from scratch under these conditions would require learning
both linguistic structure and emotional semantics simultaneously, which is
impractical with limited labeled data. Instead, we fine-tune a pretrained
transformer encoder that already captures rich syntactic and semantic
regularities of natural language. Fine-tuning allows the model to adapt this
general linguistic knowledge to the specific task of emotion classification
with substantially fewer examples and improved generalization.

For these reasons, a fine-tuned transformer-based classifier is more suitable
than classical models trained from scratch for the task considered here.
"""

"""
Rationale for model choice (ABSA emotion prediction: 7 aspects, 3 polarity, emotion)

Task shape
- Input is free text, but targets are structured: emotion is predicted per aspect (7) and is often multi-label.
- Polarity (3 classes) is tightly related to emotion and should be learned jointly or at least not treated as independent noise.

Why SVM is disadvantaged
1) Not native for multi-label:
   - SVM is fundamentally binary; multi-class uses one-vs-rest/one-vs-one, and multi-label needs wrappers.
   - With aspect-conditioned emotion, this quickly becomes many separate classifiers (labels × aspects), increasing complexity and training time.

2) Weak at aspect–emotion binding:
   - SVM consumes a single fixed feature vector; it does not inherently “attach” an emotion to a specific aspect.
   - You must engineer the input (e.g., duplicate each text 7 times with an aspect indicator) and still rely on sparse features to separate meaning.

3) No contextual semantics:
   - With TF-IDF / n-grams, SVM cannot robustly represent negation, intensity, sarcasm, or long-range dependencies.
   - Emotions are context-dependent (the same word can signal different emotions depending on surrounding text/aspect).

4) Poor probability estimates and label coupling:
   - Multi-label decisions benefit from calibrated probabilities; SVM margins are not probabilities without extra calibration.
   - SVM does not naturally model correlations/constraints (e.g., negative polarity aligning with anger/sadness) unless explicitly engineered.

Logistic Regression vs SVM (classical baselines)
- Logistic Regression with TF-IDF is often the stronger classical baseline because it:
  - supports multi-label naturally via one-vs-rest with sigmoid outputs,
  - provides probabilities directly (useful for thresholds),
  - is simpler and more scalable.
- However, both LR and SVM still lack contextual understanding and require feature engineering.

Why fine-tune DistilBERT (recommended)
- DistilBERT provides contextual embeddings that capture meaning in context, improving emotion detection in nuanced text.
- Aspect conditioning can be done cleanly by formatting input as: "[CLS] aspect [SEP] text", enabling true aspect-aware predictions.
- Multi-label emotion prediction is straightforward with a sigmoid classification head.
- DistilBERT is smaller/faster than BERT-base and usually a good trade-off for training cost vs performance.

Practical recommendation
- Use Logistic Regression (TF-IDF) as a quick baseline/sanity check.
- Use fine-tuned DistilBERT as the main model for multi-aspect, multi-label emotion prediction due to superior context modeling and easier aspect conditioning.
"""

"""
Model Choice Justification

We use DistilRoBERTa as the underlying language encoder for emotion classification.
DistilRoBERTa is a distilled version of RoBERTa that retains most of its
representational power while being significantly smaller and faster. This makes
it well-suited for experimental settings with limited computational resources
and moderate-sized datasets, while still providing strong contextual language
understanding.

RoBERTa-based models are particularly effective for sentiment and emotion-related
tasks because they are pretrained on large amounts of diverse text using a
masked language modeling objective that captures nuanced lexical, syntactic,
and semantic relationships. These properties are critical for emotion detection,
where affect is often expressed implicitly rather than through explicit emotion
words.

DistilRoBERTa offers a practical trade-off between performance and efficiency:
it enables stable fine-tuning, faster iteration, and reduced overfitting risk
compared to larger models, without sacrificing the core benefits of contextual
representations.

Alternative pretrained models could also be used within the same framework.
For example:
- BERT-base can serve as a widely adopted baseline, though it is generally
  weaker than RoBERTa-style models on sentiment-oriented tasks.
- RoBERTa-base may yield slightly higher performance at the cost of increased
  memory usage and slower training.
- ALBERT provides parameter sharing for efficiency but can be less stable during
  fine-tuning.
- Domain-specific models (e.g., sentiment-adapted or review-trained transformers)
  may further improve performance if available.

The choice of DistilRoBERTa therefore reflects a deliberate balance between model
capacity, training stability, computational efficiency, and suitability for
aspect-based emotion classification.
"""

"""
classifier_v2_FIXED.py

Fixes (no label merging, no dataset changes):
1) Defines num_emotions / num_polarity correctly (from train encoders).
2) Makes EMOTION_NAMES consistent with LabelEncoder IDs (no hardcoded ID->name map).
3) Fixes compute_metrics indentation + the "8 vs 9" sklearn mismatch by forcing labels=range(num_classes).
4) Uses HF-safe TrainingArguments key: evaluation_strategy (and keeps save_strategy).
5) Saves confusion-matrix PNG once per epoch (eval_strategy="epoch") using a TrainerCallback.
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)
# ===============================
# 1) LOAD JSONL (train/val/test)
# ===============================
def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def light_check_min(rows, name):
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

# ===============================
# 2) EXPLODE TO DATAFRAME
# ===============================
def explode_rows(rows):
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

# ===============================
# 3) LABEL ENCODING (fit on train, transform val/test)
# ===============================
emotion_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

train_df["emotion_id"]  = emotion_encoder.fit_transform(train_df["emotion"])
train_df["polarity_id"] = polarity_encoder.fit_transform(train_df["polarity"])

# NOTE: these can raise ValueError if val/test contain unseen labels (that's correct behavior)
val_df["emotion_id"]    = emotion_encoder.transform(val_df["emotion"])
val_df["polarity_id"]   = polarity_encoder.transform(val_df["polarity"])
test_df["emotion_id"]   = emotion_encoder.transform(test_df["emotion"])
test_df["polarity_id"]  = polarity_encoder.transform(test_df["polarity"])

num_emotions = len(emotion_encoder.classes_)
num_polarity = len(polarity_encoder.classes_)

# IMPORTANT: EMOTION_NAMES must match LabelEncoder ID order.
# LabelEncoder assigns IDs in alphabetical order of seen labels.
EMOTION_NAMES = list(emotion_encoder.classes_)

print("\nEmotion labels (LabelEncoder order):")
for i, name in enumerate(EMOTION_NAMES):
    print(f"{i} -> {name}")

print("\nPolarity labels (LabelEncoder order):")
for i, name in enumerate(list(polarity_encoder.classes_)):
    print(f"{i} -> {name}")

# ===============================
# 4) TOKENIZER
# ===============================
MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===============================
# 5) DATASET
# ===============================
class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
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
            "emotion_labels": torch.tensor(int(row["emotion_id"]), dtype=torch.long),
            "polarity_labels": torch.tensor(int(row["polarity_id"]), dtype=torch.long),
        }

train_ds = EmotionDataset(train_df, tokenizer)
val_ds   = EmotionDataset(val_df, tokenizer)
test_ds  = EmotionDataset(test_df, tokenizer)

# ===============================
# 6) MODEL (multi-task)
# ===============================
class EmotionPolarityModel(torch.nn.Module):
    def __init__(self, model_name: str, num_emotions: int, num_polarity: int):
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

model = EmotionPolarityModel(MODEL_NAME, num_emotions=num_emotions, num_polarity=num_polarity)


# ===============================
# 8) METRICS
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
# 9) CUSTOM TRAINER (emotion eval only)
# ===============================
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

# ===============================
# 10) TRAINING ARGS + TRAIN
# ===============================
args = TrainingArguments(
    output_dir=str(RESULTS_ROOT),
    eval_strategy="epoch",   # HF-stable key
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
    compute_metrics=compute_metrics
 
)

trainer.train()

# ===============================
# 11) OPTIONAL: FINAL TEST EVAL
# ===============================
print("\n=== FINAL TEST EVALUATION ===")
trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

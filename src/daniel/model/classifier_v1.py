#!/usr/bin/env python
# coding: utf-8
# We fine-tune a pretrained transformer encoder to perform aspect-based emotion classification.”
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
Aspect-Conditioned Emotion Classification
Supervised Fine-Tuning of a Pretrained Transformer (Multi-Task Learning)

This notebook implements supervised fine-tuning of a pretrained
transformer encoder for aspect-conditioned emotion classification
on restaurant reviews.

The task is formulated as a classification problem, where each input
consists of a review text and a target aspect, and the model predicts
the associated emotion label.

Polarity classification is included as an auxiliary task during training
to regularize the shared representation via multi-task learning.
The auxiliary polarity head is not used during inference or evaluation.

This approach corresponds to task-specific supervised fine-tuning,
not prompting and not instruction fine-tuning. Model parameters are
updated using labeled examples and cross-entropy loss, and performance
is evaluated using macro-averaged F1 on emotion labels.
"""

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

# -----------------------------
# 2) Define your important dirs
# -----------------------------
src_root     = os.path.join(project_root, "src", "daniel", "gemini")
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

import json
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix


from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)


data_file = Path(data_root) / "train.jsonl"   # adjust filename 

rows = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

len(rows)

def light_check(rows, name):
    for i, r in enumerate(rows):
        assert isinstance(r, dict), f"{name}[{i}] is not a dict"

        # input
        assert "input" in r, f"{name}[{i}] missing 'input'"
        assert isinstance(r["input"], str), f"{name}[{i}]['input'] not a string"
        assert r["input"].strip(), f"{name}[{i}] empty 'input'"

        # output
        assert "output" in r, f"{name}[{i}] missing 'output'"
        assert isinstance(r["output"], list), f"{name}[{i}]['output'] not a list"
        assert len(r["output"]) > 0, f"{name}[{i}] empty 'output' list"

        # each label item
        for j, o in enumerate(r["output"]):
            assert isinstance(o, dict), f"{name}[{i}]['output'][{j}] not a dict"
            for k in ("aspect", "polarity", "emotion"):
                assert k in o, f"{name}[{i}]['output'][{j}] missing '{k}'"
                assert isinstance(o[k], str), f"{name}[{i}]['output'][{j}]['{k}'] not a string"
                assert o[k].strip(), f"{name}[{i}]['output'][{j}] empty '{k}'"

    print(f"{name}: {len(rows)} rows passed")


light_check(rows=rows, name="train")

# Expand rows because each input can have multiple aspect-emotion-polarity labels
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
df.head()

# df["emotion"] = df["emotion"].replace({"mentioned_only": "neutral"})
df["emotion"].value_counts() # This shows the distribution of emotion labels

emotion_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

df["emotion_id"]  = emotion_encoder.fit_transform(df["emotion"])
df["polarity_id"] = polarity_encoder.fit_transform(df["polarity"])

num_emotions  = len(emotion_encoder.classes_)
num_polarity  = len(polarity_encoder.classes_)

emotion_encoder.classes_, polarity_encoder.classes_

print("Emotion labels:")
for i, label in enumerate(emotion_encoder.classes_):
    print(f"{i} → {label}")

print("\nPolarity labels:")
for i, label in enumerate(polarity_encoder.classes_):
    print(f"{i} → {label}")

EMOTION_NAMES = list(emotion_encoder.classes_)
POLARITY_NAMES = list(polarity_encoder.classes_)

unique_texts = df["text"].unique()

train_texts, val_texts = train_test_split(
    unique_texts,
    test_size=0.2,
    random_state=42
)

train_df = df[df["text"].isin(train_texts)]
val_df   = df[df["text"].isin(val_texts)]

MODEL_NAME = "distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EmotionDataset(Dataset):
    """
    PyTorch Dataset for aspect-conditioned emotion and polarity classification.

    This dataset converts a tabular representation of reviews into model-ready
    tensors. Each item corresponds to a single (text, aspect) pair and produces:

    - tokenized input text, where the aspect is explicitly prepended to guide
      the model's attention,
    - attention masks for the transformer encoder,
    - numeric labels for emotion (primary task),
    - numeric labels for polarity (auxiliary task).

    The dataset is used by the HuggingFace Trainer to dynamically construct
    training and evaluation batches during fine-tuning.
    """
    def __init__(self, df, tokenizer, max_len=256):
        
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # We explicitly prepend the aspect to the input so the model knows
        # *which part* of the sentence it should evaluate. This allows the
        # same sentence to yield different emotions depending on the aspect
        # (e.g., "food" vs. "service"), which is essential for aspect-based
        # emotion classification.
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


class EmotionPolarityModel(torch.nn.Module):
    """
    Multi-task transformer model for aspect-conditioned emotion classification.

    The model uses a shared pretrained language encoder to read the input text
    and learns two task-specific heads:
    - Emotion classification (primary task)
    - Polarity classification (auxiliary task)

    Polarity supervision is used only to regularize training. During evaluation,
    only emotion predictions are considered.
    """

    def __init__(self, model_name, num_emotions, num_polarity):
        super().__init__()

        # Load a pretrained transformer encoder (e.g., DistilRoBERTa).
        # This encoder already understands language structure and semantics.
        self.encoder = AutoModel.from_pretrained(model_name)

        # Dimensionality of the encoder's hidden representations
        hidden = self.encoder.config.hidden_size

        # Linear classification head for emotion prediction (main objective)
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)

        # Linear classification head for polarity prediction (auxiliary objective)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(
        self,
        input_ids,
        attention_mask,
        emotion_labels=None,
        polarity_labels=None
    ):
        # Pass tokenized input through the transformer encoder
        # Output contains contextual representations for each token
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use the representation of the first token as a summary of the sequence.
        # This vector encodes the overall meaning of the (aspect-conditioned) input.
        cls = out.last_hidden_state[:, 0]

        # Compute raw (unnormalized) prediction scores for each task
        emotion_logits = self.emotion_head(cls)
        polarity_logits = self.polarity_head(cls)

        loss = None
        if emotion_labels is not None:
            # Cross-entropy loss for emotion classification (primary task)
            loss_e = torch.nn.functional.cross_entropy(
                emotion_logits, emotion_labels
            )

            # Cross-entropy loss for polarity classification (auxiliary task)
            loss_p = torch.nn.functional.cross_entropy(
                polarity_logits, polarity_labels
            )

            # Joint loss: emotion dominates, polarity acts as a regularizer
            loss = loss_e + 0.3 * loss_p

        # Return a dictionary so HuggingFace Trainer can:
        # - use 'loss' for backpropagation
        # - access logits for metric computation
        return {
            "loss": loss,
            "emotion_logits": emotion_logits,
            "polarity_logits": polarity_logits
        }


# Create PyTorch Dataset objects from the processed dataframes.
# These datasets handle aspect-conditioning, tokenization, and
# conversion of text and labels into tensors that the Trainer
# can batch and feed into the model during training and validation.
train_ds = EmotionDataset(train_df, tokenizer)
val_ds   = EmotionDataset(val_df, tokenizer)

# Initialize the multi-task transformer model.
# The pretrained encoder is loaded using MODEL_NAME, and the sizes
# of the emotion and polarity classification heads are set according
# to the number of unique labels observed in the training data.
model = EmotionPolarityModel(
    MODEL_NAME,
    num_emotions=num_emotions,
    num_polarity=num_polarity
)
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



class EmotionTrainer(Trainer):
    """
    Custom Trainer that supports joint emotion–polarity training
    while evaluating only emotion predictions.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Accept **kwargs to stay compatible with newer Transformers Trainer,
        which may pass arguments like num_items_in_batch.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            emotion_labels=inputs["emotion_labels"],
            polarity_labels=inputs["polarity_labels"],
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only=False,
        ignore_keys=None,
        **kwargs
    ):
        """
        During eval/predict: run forward without labels and return only emotion logits.
        Return torch tensors (Trainer will handle gathering); avoid converting to numpy here.
        """
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # emotion-only
        logits = outputs["emotion_logits"]
        labels = inputs.get("emotion_labels")

        # Return (loss, logits, labels) as tensors
        # loss None is fine if you don't want eval loss
        if prediction_loss_only:
            return (None, None, None)

        return (None, logits, labels)


args = TrainingArguments(
    output_dir=str(RESULTS_ROOT),
    # Where checkpoints and logs would be written (even if saving is disabled)

    eval_strategy="epoch",
    # Run evaluation on the validation set at the end of every epoch.
    # This gives a stable signal of learning progress without over-evaluating.

    save_strategy="epoch",
    # Disable checkpoint saving.
    # This is intentional for experimentation and avoids disk clutter.

    learning_rate=2e-5,
    # Standard fine-tuning learning rate for transformer models.
    # Low enough to preserve pretrained knowledge, high enough to adapt.

    per_device_train_batch_size=16,
    # Number of training samples per GPU/CPU step.
    # 16 is a safe default for DistilRoBERTa on most hardware.

    per_device_eval_batch_size=16,
    # Same batch size for evaluation to keep memory usage predictable.

    num_train_epochs=5,
    # Small number of epochs is sufficient when fine-tuning pretrained models.
    # Prevents overfitting on a relatively small dataset (~3k examples).

    logging_steps=50,
    # Log training metrics every 50 steps for monitoring convergence.

    report_to="none"
    # Disable external logging frameworks (e.g., WandB, TensorBoard).
)

trainer = EmotionTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    # Dataset providing (text + aspect) → emotion + polarity supervision

    eval_dataset=val_ds,
    # Validation set used only for emotion evaluation metrics

    compute_metrics=compute_metrics
    # Computes macro-F1 on emotion labels after each evaluation
)

# Start fine-tuning the pretrained transformer model
trainer.train()


#!/usr/bin/env python
# coding: utf-8
"""
classifier_v4_logreg.py

Baseline replacement for transformer fine-tuning:
- Features: TF-IDF over "ASPECT: <aspect> | TEXT: <text>"
- Classifier: Logistic Regression (multiclass)
- Splits: uses separate train/val/test JSONL files (no merging of classes)
- Artifacts: confusion-matrix PNG + classification report for VAL and TEST
- Saves: model + vectorizer + label encoders + JSON-safe metadata into results/<experiment>/run_###/artifacts/

IMPORTANT:
- We do NOT merge or rename any emotion classes here.
- We fit LabelEncoders on TRAIN only (correct, no leakage).
- If VAL/TEST contain labels not seen in TRAIN, LabelEncoder will raise a ValueError (this is correct).
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score


# -----------------------------
# JSON-safe helper (fixes: "Object of type type is not JSON serializable")
# -----------------------------
def make_json_safe(obj):
    """
    Recursively convert non-JSON-serializable objects into JSON-safe ones.
    - dict/list/tuple -> recurse
    - numpy scalars -> python scalars
    - Path -> str
    - type / callables / anything else -> str fallback
    """
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, type):
        return obj.__name__
    # Try plain JSON
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


# -----------------------------
# 1) Find project root (robust)
# -----------------------------
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

ROOT = Path(project_root)
print(f"Project root found at: {project_root}")


# -----------------------------
# 2) Define your important dirs
# -----------------------------
src_root     = os.path.join(project_root, "src", "daniel", "model")
data_root    = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "annotated")
schemas_root = os.path.join(project_root, "data", "MAMS-ACSA", "raw", "data_jsonl", "schema")
prompts_root = os.path.join(project_root, "prompts", "daniel", "llama")  # ok if missing
utils_root   = os.path.join(project_root, "utils")

print(
    f"cwd          : {cwd}\n"
    f"Project root : {project_root}\n"
    f"Source root  : {src_root}\n"
    f"Data root    : {data_root}\n"
    f"Prompts root : {prompts_root}\n"
    f"Utils root   : {utils_root}\n"
)

DATA_ROOT = Path(data_root)


# -----------------------------
# 3) Run output directory (no overwrites)
# -----------------------------
try:
    EXPERIMENT_NAME = Path(__file__).stem  # classifier_v4_logreg
except NameError:
    EXPERIMENT_NAME = "classifier_v4_logreg"

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

ARTIFACTS_DIR = RESULTS_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Starting {EXPERIMENT_NAME}, run {RUN_ID:03d}")
print(f"Run directory: {RUN_DIR}")

(RUN_DIR / "run_info.txt").write_text(
    f"experiment={EXPERIMENT_NAME}\n"
    f"run_id={RUN_ID:03d}\n"
    f"timestamp={datetime.now().isoformat(timespec='seconds')}\n",
    encoding="utf-8"
)


# ===============================
# 1) LOAD + EXPLODE JSONL
# ===============================
def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def light_check_min(rows, name: str):
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"{name}[{i}] not a dict")
        if "input" not in r or not isinstance(r["input"], str) or not r["input"].strip():
            raise ValueError(f"{name}[{i}] missing/empty 'input'")
        if "output" not in r or not isinstance(r["output"], list) or len(r["output"]) == 0:
            raise ValueError(f"{name}[{i}] missing/empty 'output'")
    print(f"{name}: {len(rows)} rows passed minimal check")

def explode_rows(rows):
    records = []
    for r in rows:
        text = r["input"]
        for o in r["output"]:
            records.append({
                "text": text,
                "aspect": o.get("aspect"),
                "polarity": o.get("polarity"),
                "emotion": o.get("emotion"),
            })
    return pd.DataFrame(records)

train_path = DATA_ROOT / "train.jsonl"
val_path   = DATA_ROOT / "val.jsonl"
test_path  = DATA_ROOT / "test.jsonl"

if not train_path.exists():
    raise FileNotFoundError(f"Missing: {train_path}")
if not val_path.exists():
    raise FileNotFoundError(f"Missing: {val_path}")
if not test_path.exists():
    raise FileNotFoundError(f"Missing: {test_path}")

train_rows = load_jsonl(train_path)
val_rows   = load_jsonl(val_path)
test_rows  = load_jsonl(test_path)

light_check_min(train_rows, "train")
light_check_min(val_rows, "val")
light_check_min(test_rows, "test")

train_df = explode_rows(train_rows)
val_df   = explode_rows(val_rows)
test_df  = explode_rows(test_rows)

print("\nTrain emotion distribution:\n", train_df["emotion"].value_counts(dropna=False))
print("\nVal emotion distribution:\n", val_df["emotion"].value_counts(dropna=False))
print("\nTest emotion distribution:\n", test_df["emotion"].value_counts(dropna=False))


# ===============================
# 2) LABEL ENCODING (FIT ON TRAIN ONLY)
# ===============================
emotion_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

train_df["emotion_id"]  = emotion_encoder.fit_transform(train_df["emotion"])
train_df["polarity_id"] = polarity_encoder.fit_transform(train_df["polarity"])

# These will error if unseen labels exist in val/test (correct behavior).
val_df["emotion_id"]    = emotion_encoder.transform(val_df["emotion"])
val_df["polarity_id"]   = polarity_encoder.transform(val_df["polarity"])
test_df["emotion_id"]   = emotion_encoder.transform(test_df["emotion"])
test_df["polarity_id"]  = polarity_encoder.transform(test_df["polarity"])

EMOTION_NAMES = list(emotion_encoder.classes_)
num_classes = len(EMOTION_NAMES)

print("\nEmotion label mapping (LabelEncoder order):")
for i, name in enumerate(EMOTION_NAMES):
    print(f"{i} -> {name}")


# ===============================
# 3) FEATURE BUILDING (TF-IDF)
# ===============================
def make_inputs(df: pd.DataFrame) -> pd.Series:
    return ("ASPECT: " + df["aspect"].astype(str) + " | TEXT: " + df["text"].astype(str))

X_train_text = make_inputs(train_df)
X_val_text   = make_inputs(val_df)
X_test_text  = make_inputs(test_df)

y_train = train_df["emotion_id"].values
y_val   = val_df["emotion_id"].values
y_test  = test_df["emotion_id"].values

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train = vectorizer.fit_transform(X_train_text)
X_val   = vectorizer.transform(X_val_text)
X_test  = vectorizer.transform(X_test_text)

print("\nTF-IDF shapes:", X_train.shape, X_val.shape, X_test.shape)


# ===============================
# 4) MODEL: LOGISTIC REGRESSION
# ===============================
clf = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=None,  # lbfgs ignores n_jobs; keep None
)

clf.fit(X_train, y_train)


# ===============================
# 5) EVALUATION + PLOTS
# ===============================
def save_confusion_matrix_png(y_true, y_pred, label_names, out_path: Path, title: str):
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"G_{n}" for n in label_names],
        columns=[f"P_{n}" for n in label_names],
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
        cbar_kws={"label": "Count"},
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    plt.setp(ax.get_yticklabels(), rotation=0)

    for i in range(len(label_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

    plt.title(title, pad=30)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Gold Emotion")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def evaluate_split(name: str, X, y_true):
    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=list(range(num_classes)), average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
    )

    print(f"\n=== {name} RESULTS (LogReg) ===")
    print(f"accuracy   : {acc:.4f}")
    print(f"macro_f1   : {macro:.4f}")
    print(f"weighted_f1: {weighted:.4f}")
    print("\nClassification Report:\n", report)

    (RESULTS_ROOT / f"{name.lower()}_classification_report.txt").write_text(report, encoding="utf-8")

    png_path = RESULTS_ROOT / f"confusion_matrix_{name.lower()}.png"
    save_confusion_matrix_png(
        y_true=y_true,
        y_pred=y_pred,
        label_names=EMOTION_NAMES,
        out_path=png_path,
        title=f"Emotion Confusion Matrix (Counts) — {name} (LogReg)",
    )
    print("Saved:", png_path)

    return {"split": name, "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}

metrics = []
metrics.append(evaluate_split("VAL", X_val, y_val))
metrics.append(evaluate_split("TEST", X_test, y_test))

summary_df = pd.DataFrame(metrics)
summary_path = RESULTS_ROOT / "metrics_summary.csv"
summary_df.to_csv(summary_path, index=False)
print("\nSaved:", summary_path)


# ===============================
# 6) SAVE TRAINED ARTIFACTS
# ===============================
model_path      = ARTIFACTS_DIR / "logreg_model.joblib"
vectorizer_path = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
emo_enc_path    = ARTIFACTS_DIR / "emotion_encoder.joblib"
pol_enc_path    = ARTIFACTS_DIR / "polarity_encoder.joblib"
meta_path       = ARTIFACTS_DIR / "metadata.json"

joblib.dump(clf, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(emotion_encoder, emo_enc_path)
joblib.dump(polarity_encoder, pol_enc_path)

metadata = {
    "experiment_name": EXPERIMENT_NAME,
    "run_id": f"{RUN_ID:03d}",
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "results_root": str(RESULTS_ROOT),
    "artifacts_dir": str(ARTIFACTS_DIR),
    "paths": {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    },
    "data_sizes": {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    },
    "labels": {
        "emotion_names": EMOTION_NAMES,
        "num_classes": int(num_classes),
    },
    "vectorizer_params": make_json_safe(vectorizer.get_params()),
    "model_params": make_json_safe(clf.get_params()),
}

meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("\nSaved artifacts:")
print(" -", model_path)
print(" -", vectorizer_path)
print(" -", emo_enc_path)
print(" -", pol_enc_path)
print(" -", meta_path)

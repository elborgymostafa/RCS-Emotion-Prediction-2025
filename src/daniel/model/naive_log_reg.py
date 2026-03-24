import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path
import os
import sys
from datetime import datetime

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

BASE_RESULTS_DIR = ROOT / "results_v2" / EXPERIMENT_NAME
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


train_path = Path(data_root) / "train.jsonl"
val_path   = Path(data_root) / "val.jsonl"
test_path  = Path(data_root) / "test.jsonl"
# --- Loading and preparing data ---

# Function to load and flatten jsonl with nested aspects and emotions
def load_and_flatten_data(filepath):
    texts = []
    polarities = []
    aspects = []
    emotions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["input"]
            outputs = data["output"]
            for out in outputs:
                texts.append(text)
                polarities.append(out["polarity"])
                aspects.append(out["aspect"])
                emotions.append(out["emotion"])
    return texts, polarities, aspects, emotions


# Load data
train_texts, train_polarities, train_aspects, train_emotions = load_and_flatten_data(train_path)
val_texts, val_polarities, val_aspects, val_emotions = load_and_flatten_data(val_path)
test_texts, test_polarities, test_aspects, test_emotions = load_and_flatten_data(test_path)

# --- Create new features: combine text with aspect and polarity to simplify the model ---
def combine_features(texts, polarities, aspects):
    combined = []
    for t, p, a in zip(texts, polarities, aspects):
        combined.append(f"{t} [POLARITY_{p.upper()}] [ASPECT_{a.upper()}]")
    return combined

train_combined = combine_features(train_texts, train_polarities, train_aspects)
val_combined = combine_features(val_texts, val_polarities, val_aspects)
test_combined = combine_features(test_texts, test_polarities, test_aspects)

# --- Encode emotions as numeric labels ---
label_encoder = LabelEncoder()
all_emotions = train_emotions + val_emotions + test_emotions
label_encoder.fit(all_emotions)

y_train = label_encoder.transform(train_emotions)
y_val = label_encoder.transform(val_emotions)
y_test = label_encoder.transform(test_emotions)

# --- Vectorize text including polarity and aspect ---
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_combined)
X_val = vectorizer.transform(val_combined)
X_test = vectorizer.transform(test_combined)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Predictions ---
val_preds = model.predict(X_val)
test_preds = model.predict(X_test)

# --- Reports considering only actually occurring labels ---

unique_labels_val = np.unique(y_val)
target_names_val = label_encoder.inverse_transform(unique_labels_val)

print("Validation classification report:")
print(classification_report(y_val, val_preds,
                            labels=unique_labels_val,
                            target_names=target_names_val))

unique_labels_test = np.unique(y_test)
target_names_test = label_encoder.inverse_transform(unique_labels_test)

print("Test classification report:")
print(classification_report(y_test, test_preds,
                            labels=unique_labels_test,
                            target_names=target_names_test))

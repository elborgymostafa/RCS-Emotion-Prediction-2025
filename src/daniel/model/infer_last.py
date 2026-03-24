#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
INFERENCE SCRIPT — classifier_v2 (latest checkpoint loader)
===========================================================

What this script does:

1. Finds the latest training run inside:
      results/classifier_v2/run_XXX/

2. Loads the most recent checkpoint:
      checkpoint-YYYY/

3. Rebuilds label encoders from train.jsonl
   (ensures prediction label IDs match training).

4. Prints FIRST_N samples from test.jsonl.

5. Runs inference on those samples:
      - shows raw text
      - aspect
      - gold labels
      - predicted labels
      - top-k probabilities

This allows quick manual inspection of model behaviour.
"""


# ==============================
# Imports
# ==============================
import re
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


# ==============================
# GLOBAL SETTINGS (edit freely)
# ==============================

FIRST_N = 20        # How many test rows to inspect
TOPK = 3            # How many top predictions to show
MODEL_NAME = "distilroberta-base"
MAX_LEN = 256       # Tokenizer max length

# Folder structure assumptions
EXPERIMENT_DIR = Path("results") / "classifier_v2"
DATA_ROOT = Path("data") / "MAMS-ACSA" / "raw" / "data_jsonl" / "annotated"
TRAIN_JSONL = DATA_ROOT / "train.jsonl"
TEST_JSONL  = DATA_ROOT / "test.jsonl"


# ============================================================
# JSONL LOADING HELPERS
# ============================================================

def load_jsonl(path: Path):
    """
    Reads JSONL file line by line.

    JSONL = one JSON object per line.
    Useful for large NLP datasets.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def explode_rows(rows):
    """
    Converts dataset structure:

    {
      "input": "...text...",
      "output": [
          {"aspect": "...", "polarity": "...", "emotion": "..."},
          ...
      ]
    }

    → into flat rows:
       one row per aspect/emotion pair.

    This matches your training preprocessing.
    """
    recs = []
    for r in rows:
        text = r["input"]
        for o in r["output"]:
            recs.append({
                "text": text,
                "aspect": o["aspect"],
                "polarity": o["polarity"],
                "emotion": o["emotion"],
            })
    return recs


# ============================================================
# FIND LATEST RUN + CHECKPOINT
# ============================================================

def latest_run_dir(exp_dir: Path):
    """
    Finds highest-numbered run folder:
        run_001, run_002, ...

    Ensures we load the newest experiment.
    """
    if not exp_dir.exists():
        raise RuntimeError(f"Experiment dir missing: {exp_dir.resolve()}")

    runs = [p for p in exp_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")]

    if not runs:
        raise RuntimeError("No training runs found.")

    runs.sort(key=lambda p: int(p.name.split("_")[1]))
    return runs[-1]


def latest_checkpoint_dir(run_dir: Path):
    """
    Inside a run folder, pick the checkpoint
    with the largest training step number.

    Example:
        checkpoint-1000
        checkpoint-2000  ← newest
    """
    ckpts = [p for p in run_dir.iterdir()
             if p.is_dir() and p.name.startswith("checkpoint-")]

    if not ckpts:
        raise RuntimeError("No checkpoints found.")

    def step(p):
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=step)
    return ckpts[-1]


# ============================================================
# MODEL DEFINITION (must match training architecture)
# ============================================================

class EmotionPolarityModel(torch.nn.Module):
    """
    Multi-task transformer classifier:

    - Shared encoder (DistilRoBERTa)
    - Two classification heads:
        1. Emotion prediction
        2. Polarity prediction
    """

    def __init__(self, model_name, num_emotions, num_polarity):
        super().__init__()

        # Pretrained contextual encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        # Separate linear heads
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(self, input_ids, attention_mask):

        # Transformer forward pass
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)

        # CLS token embedding
        cls = out.last_hidden_state[:, 0]

        return {
            "emotion_logits": self.emotion_head(cls),
            "polarity_logits": self.polarity_head(cls),
        }
def load_weights(model: torch.nn.Module, ckpt_dir: Path):
    """
    Load model weights from a Trainer checkpoint folder.

    Supports:
      - pytorch_model.bin
      - model.safetensors
    """
    bin_path = ckpt_dir / "pytorch_model.bin"
    safe_path = ckpt_dir / "model.safetensors"

    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from: {bin_path.name}")
        return

    if safe_path.exists():
        # safetensors is commonly installed with transformers, but let's import safely
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError(
                "Found model.safetensors but 'safetensors' is not installed.\n"
                "Fix: pip install safetensors\n"
                f"Original error: {e}"
            )

        state = load_file(str(safe_path))  # returns a state_dict-like mapping
        model.load_state_dict(state)
        print(f"Loaded weights from: {safe_path.name}")
        return

    raise RuntimeError(
        f"Missing weights in {ckpt_dir.resolve()}\n"
        f"Expected: pytorch_model.bin OR model.safetensors\n"
        f"Found: {[p.name for p in ckpt_dir.iterdir()]}"
    )



# ============================================================
# INFERENCE FUNCTION
# ============================================================

@torch.no_grad()
def predict_topk(model, tokenizer, text, aspect,
                 emotion_names, polarity_names,
                 device, topk=3):
    """
    Runs one inference example.

    Steps:
      1. Format input same as training
      2. Tokenize
      3. Forward pass
      4. Softmax probabilities
      5. Return top-k predictions
    """

    formatted = f"ASPECT: {aspect} | TEXT: {text}"

    enc = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids,
                attention_mask=attention_mask)

    # Convert logits → probabilities
    e_probs = torch.softmax(out["emotion_logits"][0], dim=0).cpu().numpy()
    p_probs = torch.softmax(out["polarity_logits"][0], dim=0).cpu().numpy()

    # Highest probabilities first
    e_top = np.argsort(-e_probs)[:topk]
    p_top = np.argsort(-p_probs)[:topk]

    e_topk = [(emotion_names[i], float(e_probs[i])) for i in e_top]
    p_topk = [(polarity_names[i], float(p_probs[i])) for i in p_top]

    return e_topk, p_topk


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    # --- locate checkpoint ---
    run_dir = latest_run_dir(EXPERIMENT_DIR)
    ckpt_dir = latest_checkpoint_dir(run_dir)

    print("Experiment:", EXPERIMENT_DIR.resolve())
    print("Run       :", run_dir.name)
    print("Checkpoint:", ckpt_dir.name)

    # --- rebuild label encoders ---
    train_rows = load_jsonl(TRAIN_JSONL)
    train_recs = explode_rows(train_rows)

    emotion_enc = LabelEncoder().fit([r["emotion"] for r in train_recs])
    polarity_enc = LabelEncoder().fit([r["polarity"] for r in train_recs])

    EMOTION_NAMES = list(emotion_enc.classes_)
    POLARITY_NAMES = list(polarity_enc.classes_)

    # --- load tokenizer + model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = EmotionPolarityModel(
        MODEL_NAME,
        len(EMOTION_NAMES),
        len(POLARITY_NAMES),
    )

    load_weights(model, ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print("Device:", device)

    # --- load test data ---
    test_rows = load_jsonl(TEST_JSONL)
    test_recs = explode_rows(test_rows)

    n = min(FIRST_N, len(test_recs))

    print(f"\n===== FIRST {n} TEST ROWS + PREDICTIONS =====")

    # --- inference loop ---
    for i, rec in enumerate(test_recs[:n], start=1):

        e_topk, p_topk = predict_topk(
            model, tokenizer,
            rec["text"], rec["aspect"],
            EMOTION_NAMES, POLARITY_NAMES,
            device,
            TOPK,
        )

        print(f"\n--- SAMPLE {i:02d} ---")
        print("TEXT   :", rec["text"])
        print("ASPECT :", rec["aspect"])
        print("GOLD   :", rec["emotion"], "|", rec["polarity"])
        print("PRED   :", e_topk[0], "|", p_topk[0])
        print("TOP EMO:", e_topk)
        print("TOP POL:", p_topk)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
Flask Demo App — ABSA Emotion + Polarity (classifier_v2)
===========================================================

Features (presentation-ready):
1) "Row lookup" mode:
   - enter test.jsonl row id (0-based)
   - shows text + ALL gold aspects (emotion + polarity)
   - runs inference for each aspect

2) "Crazy tester" mode:
   - paste your own review text
   - choose aspect from dropdown (auto-filled from dataset)
   - run inference (no gold labels for custom text)

3) Output style:
   - clean blocks per aspect
   - top-1 + top-k
   - full probability distribution (bars) for ALL labels
   - shows checkpoint info: run, checkpoint, weight file, device, label sets

How to run:
-----------
source /path/to/.venv/bin/activate
python app.py
open http://127.0.0.1:5000
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from flask import Flask, request, render_template

from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG — change only if your paths differ
# ============================================================

MODEL_NAME = "distilroberta-base"
MAX_LEN = 256

# IMPORTANT: run this app from your project root so these relative paths resolve
EXPERIMENT_DIR = Path("results") / "classifier_v2"
DATA_ROOT = Path("data") / "MAMS-ACSA" / "raw" / "data_jsonl" / "annotated"
TRAIN_JSONL = DATA_ROOT / "train.jsonl"
TEST_JSONL  = DATA_ROOT / "test.jsonl"

DEFAULT_TOPK = 3


# ============================================================
# Small helpers
# ============================================================

def load_jsonl(path: Path) -> List[dict]:
    """Read JSONL (one JSON object per line) into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def latest_run_dir(exp_dir: Path) -> Path:
    """
    Find newest run folder: run_001, run_002, ...
    We pick the highest run number.
    """
    if not exp_dir.exists():
        raise RuntimeError(f"Experiment dir missing: {exp_dir.resolve()}")

    runs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise RuntimeError(f"No run_* folders found in: {exp_dir.resolve()}")

    runs.sort(key=lambda p: int(p.name.split("_")[1]))
    return runs[-1]


def latest_checkpoint_dir(run_dir: Path) -> Path:
    """
    Find newest checkpoint folder: checkpoint-1000, checkpoint-2000, ...
    We pick the highest step number.
    """
    ckpts = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        raise RuntimeError(f"No checkpoint-* folders found in: {run_dir.resolve()}")

    def step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=step)
    return ckpts[-1]


# ============================================================
# Model definition — MUST match training architecture exactly
# ============================================================

class EmotionPolarityModel(torch.nn.Module):
    """
    Multi-task model:
      encoder(text+aspect) -> shared representation -> two heads:
        - emotion_head
        - polarity_head
    """

    def __init__(self, model_name: str, num_emotions: int, num_polarity: int):
        super().__init__()

        # Pretrained encoder (DistilRoBERTa)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        # Task heads
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # CLS-like first token embedding
        cls = out.last_hidden_state[:, 0]

        return {
            "emotion_logits": self.emotion_head(cls),
            "polarity_logits": self.polarity_head(cls),
        }


def load_weights(model: torch.nn.Module, ckpt_dir: Path) -> str:
    """
    Load weights from HF Trainer checkpoint.

    HF can store:
      - pytorch_model.bin
      - model.safetensors

    Return which file was used (nice to show in UI).
    """
    bin_path = ckpt_dir / "pytorch_model.bin"
    safe_path = ckpt_dir / "model.safetensors"

    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state)
        return bin_path.name

    if safe_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError(
                "Found model.safetensors but 'safetensors' is not installed.\n"
                "Fix: pip install safetensors\n"
                f"Original error: {e}"
            )
        state = load_file(str(safe_path))
        model.load_state_dict(state)
        return safe_path.name

    raise RuntimeError(
        f"Missing weights in {ckpt_dir.resolve()}.\n"
        f"Found files: {[p.name for p in ckpt_dir.iterdir()]}"
    )


# ============================================================
# Inference logic — returns FULL distributions + top-k
# ============================================================

@torch.no_grad()
def infer_one(
    model: EmotionPolarityModel,
    tokenizer,
    text: str,
    aspect: str,
    emotion_names: List[str],
    polarity_names: List[str],
    device: str,
    topk: int,
) -> Dict[str, Any]:
    """
    Run inference for one (text, aspect).

    Returns a dict with:
      - formatted_input (exact string fed into tokenizer)
      - latency_ms
      - emotion: top1, topk, all(probabilities)
      - polarity: top1, topk, all(probabilities)
    """

    # IMPORTANT: must match training formatting
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

    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    latency_ms = (time.time() - t0) * 1000.0

    # logits -> probabilities
    e_probs = torch.softmax(out["emotion_logits"][0], dim=0).cpu().numpy()
    p_probs = torch.softmax(out["polarity_logits"][0], dim=0).cpu().numpy()

    # full distributions sorted by prob desc
    e_all = sorted([(emotion_names[i], float(e_probs[i])) for i in range(len(emotion_names))],
                   key=lambda x: -x[1])
    p_all = sorted([(polarity_names[i], float(p_probs[i])) for i in range(len(polarity_names))],
                   key=lambda x: -x[1])

    # top-k is simply first k after sorting
    e_topk = e_all[:topk]
    p_topk = p_all[:topk]

    return {
        "formatted_input": formatted,
        "latency_ms": float(latency_ms),
        "emotion": {
            "top1": e_topk[0][0],
            "top1_conf": e_topk[0][1],
            "topk": e_topk,
            "all": e_all,
        },
        "polarity": {
            "top1": p_topk[0][0],
            "top1_conf": p_topk[0][1],
            "topk": p_topk,
            "all": p_all,
        }
    }


# ============================================================
# Runtime bootstrap — load everything ONCE at startup
# ============================================================

def build_runtime():
    # ---- validate paths early (fail fast) ----
    if not TRAIN_JSONL.exists():
        raise RuntimeError(f"Missing: {TRAIN_JSONL.resolve()}")
    if not TEST_JSONL.exists():
        raise RuntimeError(f"Missing: {TEST_JSONL.resolve()}")

    # ---- find latest checkpoint ----
    run_dir = latest_run_dir(EXPERIMENT_DIR)
    ckpt_dir = latest_checkpoint_dir(run_dir)

    # ---- load datasets ----
    train_rows = load_jsonl(TRAIN_JSONL)
    test_rows = load_jsonl(TEST_JSONL)

    # ---- rebuild encoders from TRAIN (must match training label mapping) ----
    emotions = []
    polarities = []
    aspects = set()

    for r in train_rows:
        for o in r.get("output", []):
            emotions.append(o["emotion"])
            polarities.append(o["polarity"])
            aspects.add(o["aspect"])

    # Also add aspects from test to make dropdown robust (optional but helpful)
    for r in test_rows:
        for o in r.get("output", []):
            aspects.add(o["aspect"])

    emotion_enc = LabelEncoder().fit(emotions)
    polarity_enc = LabelEncoder().fit(polarities)

    EMOTION_NAMES = list(emotion_enc.classes_)
    POLARITY_NAMES = list(polarity_enc.classes_)
    ASPECTS = sorted(list(aspects))

    # ---- create tokenizer + model + load weights ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmotionPolarityModel(MODEL_NAME, len(EMOTION_NAMES), len(POLARITY_NAMES))
    weights_file = load_weights(model, ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # ---- meta shown in UI ----
    meta = {
        "model_name": MODEL_NAME,
        "device": device,
        "experiment_dir": str(EXPERIMENT_DIR.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "weights_file": weights_file,
        "num_test_rows": len(test_rows),
        "emotion_labels": EMOTION_NAMES,
        "polarity_labels": POLARITY_NAMES,
        "aspects": ASPECTS,
    }

    return model, tokenizer, train_rows, test_rows, EMOTION_NAMES, POLARITY_NAMES, ASPECTS, meta


MODEL, TOKENIZER, TRAIN_ROWS, TEST_ROWS, EMOTION_NAMES, POLARITY_NAMES, ASPECTS, META = build_runtime()


# ============================================================
# Flask app
# ============================================================

app = Flask(__name__)


def clamp_topk(topk: int) -> int:
    """Top-k should not exceed number of labels."""
    max_k = max(len(EMOTION_NAMES), len(POLARITY_NAMES))
    if topk < 1:
        return 1
    if topk > max_k:
        return max_k
    return topk


@app.route("/", methods=["GET"])
def index():
    """
    Single page with TWO forms:
      - Row lookup (test.jsonl row id)
      - Custom review + aspect dropdown
    """
    return render_template(
        "index.html",
        meta=META,
        aspects=ASPECTS,
        row_mode=None,
        custom_mode=None,
        error=None,
        default_topk=DEFAULT_TOPK,
    )


@app.route("/predict_row", methods=["POST"])
def predict_row():
    """
    Row lookup:
      - user gives row_id from test.jsonl (0-based)
      - we show gold outputs and predictions for ALL aspects in that row
    """
    error = None
    row_mode = None
    custom_mode = None

    row_id_raw = request.form.get("row_id", "").strip()
    topk_raw = request.form.get("topk_row", str(DEFAULT_TOPK)).strip()

    # parse topk
    try:
        topk = clamp_topk(int(topk_raw))
    except Exception:
        topk = DEFAULT_TOPK

    # parse row_id
    try:
        row_id = int(row_id_raw)
    except Exception:
        error = "Row id must be an integer (0-based)."
        return render_template("index.html", meta=META, aspects=ASPECTS,
                               row_mode=None, custom_mode=None, error=error, default_topk=DEFAULT_TOPK)

    if row_id < 0 or row_id >= len(TEST_ROWS):
        error = f"Row id out of range. Valid: 0..{len(TEST_ROWS)-1}"
        return render_template("index.html", meta=META, aspects=ASPECTS,
                               row_mode=None, custom_mode=None, error=error, default_topk=DEFAULT_TOPK)

    item = TEST_ROWS[row_id]
    text = item.get("input", "")
    outs = item.get("output", [])

    if not isinstance(outs, list) or len(outs) == 0:
        error = "This row has no output aspects."
        return render_template("index.html", meta=META, aspects=ASPECTS,
                               row_mode=None, custom_mode=None, error=error, default_topk=DEFAULT_TOPK)

    # run inference for each aspect in gold outputs
    predictions = []
    for o in outs:
        aspect = o.get("aspect", "")
        gold_e = o.get("emotion", "")
        gold_p = o.get("polarity", "")

        pred = infer_one(
            model=MODEL,
            tokenizer=TOKENIZER,
            text=text,
            aspect=aspect,
            emotion_names=EMOTION_NAMES,
            polarity_names=POLARITY_NAMES,
            device=META["device"],
            topk=topk
        )

        predictions.append({
            "aspect": aspect,
            "gold": {"emotion": gold_e, "polarity": gold_p},
            "pred": pred
        })

    row_mode = {
        "row_id": row_id,
        "topk": topk,
        "text": text,
        "outputs": outs,
        "predictions": predictions,
    }

    return render_template(
        "index.html",
        meta=META,
        aspects=ASPECTS,
        row_mode=row_mode,
        custom_mode=None,
        error=None,
        default_topk=DEFAULT_TOPK,
    )


@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    """
    Crazy tester:
      - user provides custom text
      - chooses aspect from dropdown OR types their own
      - we run inference once (no gold labels)
    """
    error = None
    row_mode = None
    custom_mode = None

    text = (request.form.get("custom_text", "") or "").strip()
    chosen_aspect = (request.form.get("custom_aspect", "") or "").strip()
    free_aspect = (request.form.get("custom_aspect_free", "") or "").strip()
    topk_raw = request.form.get("topk_custom", str(DEFAULT_TOPK)).strip()

    # parse topk
    try:
        topk = clamp_topk(int(topk_raw))
    except Exception:
        topk = DEFAULT_TOPK

    # pick aspect: free text overrides dropdown if provided
    aspect = free_aspect if free_aspect else chosen_aspect

    if not text:
        error = "Please paste a review text."
    elif not aspect:
        error = "Please choose an aspect (or type one)."

    if error:
        return render_template(
            "index.html",
            meta=META,
            aspects=ASPECTS,
            row_mode=None,
            custom_mode=None,
            error=error,
            default_topk=DEFAULT_TOPK,
        )

    pred = infer_one(
        model=MODEL,
        tokenizer=TOKENIZER,
        text=text,
        aspect=aspect,
        emotion_names=EMOTION_NAMES,
        polarity_names=POLARITY_NAMES,
        device=META["device"],
        topk=topk
    )

    custom_mode = {
        "topk": topk,
        "text": text,
        "aspect": aspect,
        "pred": pred
    }

    return render_template(
        "index.html",
        meta=META,
        aspects=ASPECTS,
        row_mode=None,
        custom_mode=custom_mode,
        error=None,
        default_topk=DEFAULT_TOPK,
    )


if __name__ == "__main__":
    print("\n[ABSA Demo] Runtime loaded:")
    print("  model_name     :", META["model_name"])
    print("  device         :", META["device"])
    print("  run_dir        :", META["run_dir"])
    print("  checkpoint_dir :", META["checkpoint_dir"])
    print("  weights_file   :", META["weights_file"])
    print("  #test_rows     :", META["num_test_rows"])
    print("\nOpen: http://127.0.0.1:5000\n")

    # For demo day: debug=False is cleaner (no auto reload)
    app.run(host="127.0.0.1", port=5000, debug=False)

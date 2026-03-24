#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
INFERENCE + QUICK F1 CHECK — classifier_v2 (latest checkpoint)
===========================================================

IMPORTANT FIX (your issue):
---------------------------
Your test.jsonl has this structure per LINE (per JSONL item):

{
  "input": "<same text>",
  "output": [
     {"aspect": "...", "polarity": "...", "emotion": "..."},
     {"aspect": "...", "polarity": "...", "emotion": "..."},
     ...
  ]
}

So *one* JSONL row can contain MULTIPLE aspects.

YourIght now, your script:
- explodes the file first (turns each aspect into a separate row),
- then slices FIRST_N exploded rows,
- so if FIRST_N=1, you see only the FIRST aspect of the FIRST JSONL item.

WHAT WE DO NOW:
---------------
We support BOTH modes:

Mode A: JSONL ITEMS mode (recommended for "row 1 has multiple aspects")
- FIRST_JSONL_ITEMS = 1   => print + infer the first JSONL item
- and for that item, run inference on ALL aspects inside output[]

Mode B: EXPLODED ROWS mode (your old behavior)
- FIRST_N_EXPLODED = 20   => print + infer first 20 exploded rows

You can enable one or the other by setting:
- FIRST_JSONL_ITEMS to an int (e.g., 1) and FIRST_N_EXPLODED to None
OR
- FIRST_N_EXPLODED to an int and FIRST_JSONL_ITEMS to None

Additionally:
- We compute quick emotion F1 on whatever subset we evaluated.
- We keep polarity prediction printing, because your model still has polarity head.
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModel


# ==============================
# GLOBAL SETTINGS (edit freely)
# ==============================

# ----- choose ONE mode -----
FIRST_JSONL_ITEMS = 400      # <-- how many ORIGINAL JSONL items to run (prints ALL aspects per item)
FIRST_N_EXPLODED = None    # <-- how many exploded rows to run (set this instead of FIRST_JSONL_ITEMS)

TOPK = 3                   # how many top predictions to show per sample
MODEL_NAME = "distilroberta-base"
MAX_LEN = 256              # tokenizer max length

# Folder structure assumptions (run script from project root)
EXPERIMENT_DIR = Path("results") / "classifier_v2"
DATA_ROOT = Path("data") / "MAMS-ACSA" / "raw" / "data_jsonl" / "annotated"
TRAIN_JSONL = DATA_ROOT / "train.jsonl"
TEST_JSONL  = DATA_ROOT / "test.jsonl"


# ============================================================
# JSONL LOADING HELPERS
# ============================================================

def load_jsonl(path: Path):
    """
    Read a JSONL file line-by-line.
    JSONL = one JSON object per line.
    Returns: list[dict]
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def explode_rows(rows):
    """
    Convert JSONL items into "exploded rows" (one row per aspect annotation).

    Input JSONL item:
      {"input": "...",
       "output": [{"aspect":..,"polarity":..,"emotion":..}, ...]}

    Output exploded list:
      [{"text":..., "aspect":..., "polarity":..., "emotion":...},
       {"text":..., "aspect":..., "polarity":..., "emotion":...}, ...]
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
    Find newest run directory: run_001, run_002, ...
    """
    if not exp_dir.exists():
        raise RuntimeError(f"Experiment dir missing: {exp_dir.resolve()}")

    runs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise RuntimeError(f"No run_* folders found in: {exp_dir.resolve()}")

    runs.sort(key=lambda p: int(p.name.split("_")[1]))
    return runs[-1]


def latest_checkpoint_dir(run_dir: Path):
    """
    Pick checkpoint with highest step number:
      checkpoint-1000 < checkpoint-2000  => checkpoint-2000 is newer
    """
    ckpts = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        raise RuntimeError(f"No checkpoint-* folders found in: {run_dir.resolve()}")

    def step(p: Path):
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=step)
    return ckpts[-1]


# ============================================================
# MODEL (must match your training architecture)
# ============================================================

class EmotionPolarityModel(torch.nn.Module):
    """
    Multi-task model:
    - Shared transformer encoder (DistilRoBERTa)
    - Emotion head: predicts one of N emotions
    - Polarity head: predicts one of 3 polarities
    """

    def __init__(self, model_name: str, num_emotions: int, num_polarity: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass:
        - Encode tokens
        - Take first-token embedding (CLS-like)
        - Produce logits for both heads
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # CLS-like first token embedding

        return {
            "emotion_logits": self.emotion_head(cls),
            "polarity_logits": self.polarity_head(cls),
        }


def load_weights(model: torch.nn.Module, ckpt_dir: Path):
    """
    Load weights from a HuggingFace Trainer checkpoint.

    HF can save as either:
      - pytorch_model.bin
      - model.safetensors

    Your run_005/checkpoint-2220 contains model.safetensors,
    so we support both formats.
    """
    bin_path = ckpt_dir / "pytorch_model.bin"
    safe_path = ckpt_dir / "model.safetensors"

    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from: {bin_path.name}")
        return

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
        print(f"Loaded weights from: {safe_path.name}")
        return

    raise RuntimeError(
        f"Missing weights in {ckpt_dir.resolve()}\n"
        f"Expected: pytorch_model.bin OR model.safetensors\n"
        f"Found: {[p.name for p in ckpt_dir.iterdir()]}"
    )


# ============================================================
# INFERENCE HELPERS
# ============================================================

@torch.no_grad()
def predict_topk(model, tokenizer, text: str, aspect: str,
                 emotion_names, polarity_names,
                 device: str, topk: int = 3):
    """
    Predict emotion + polarity for one (text, aspect) instance.

    Key: the input formatting MUST match training exactly:
      "ASPECT: {aspect} | TEXT: {text}"

    Returns:
      e_topk = [(emotion_name, prob), ...] length=topk
      p_topk = [(polarity_name, prob), ...] length=topk
    """
    formatted = f"ASPECT: {aspect} | TEXT: {text}"

    # Tokenize into model inputs
    enc = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Forward pass
    out = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert logits -> probabilities
    e_probs = torch.softmax(out["emotion_logits"][0], dim=0).cpu().numpy()
    p_probs = torch.softmax(out["polarity_logits"][0], dim=0).cpu().numpy()

    # Pick top-k indices by probability (descending)
    e_top = np.argsort(-e_probs)[:topk]
    p_top = np.argsort(-p_probs)[:topk]

    # Map indices to names
    e_topk = [(emotion_names[i], float(e_probs[i])) for i in e_top]
    p_topk = [(polarity_names[i], float(p_probs[i])) for i in p_top]

    return e_topk, p_topk


def print_and_collect_one_prediction(idx: int, text: str, aspect: str,
                                     gold_e: str, gold_p: str,
                                     e_topk, p_topk,
                                     gold_emotions: list, pred_emotions: list):
    """
    Centralized printing so both modes (JSONL-items and exploded-rows)
    use the same formatting and F1 collection.

    We collect only emotion labels for the F1 you asked for.
    """
    pred_e = e_topk[0][0]
    pred_p = p_topk[0][0]

    gold_emotions.append(gold_e)
    pred_emotions.append(pred_e)

    print(f"\n--- SAMPLE {idx:02d} ---")
    print("TEXT   :", text)
    print("ASPECT :", aspect)
    print("GOLD   :", f"emotion={gold_e} | polarity={gold_p}")
    print("PRED   :", f"emotion={pred_e} ({e_topk[0][1]:.3f}) | polarity={pred_p} ({p_topk[0][1]:.3f})")
    print("TOP EMO:", ", ".join([f"{n}:{p:.3f}" for n, p in e_topk]))
    print("TOP POL:", ", ".join([f"{n}:{p:.3f}" for n, p in p_topk]))


def quick_emotion_f1(gold_emotions: list, pred_emotions: list, emotion_names: list):
    """
    Compute quick macro/micro F1 + print report + confusion matrix.
    This is a "smoke test": useful to confirm pipeline correctness.
    """
    if not gold_emotions:
        print("\n(No samples evaluated => skipping F1.)")
        return

    macro = f1_score(gold_emotions, pred_emotions, average="macro", zero_division=0)
    micro = f1_score(gold_emotions, pred_emotions, average="micro", zero_division=0)

    print("\n===== QUICK EMOTION F1 ON EVALUATED SAMPLES =====")
    print(f"Rows evaluated : {len(gold_emotions)}")
    print(f"Macro F1       : {macro:.4f}")
    print(f"Micro F1       : {micro:.4f}")

    print("\nPer-class report (emotion):")
    print(classification_report(
        gold_emotions,
        pred_emotions,
        labels=emotion_names,
        zero_division=0,
        digits=4
    ))

    cm = confusion_matrix(gold_emotions, pred_emotions, labels=emotion_names)
    print("Confusion matrix (emotion) labels order:")
    print(emotion_names)
    print(cm)


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------
    # 0) Sanity check data paths
    # --------------------------
    if not TRAIN_JSONL.exists():
        raise RuntimeError(f"Missing train jsonl: {TRAIN_JSONL.resolve()}")
    if not TEST_JSONL.exists():
        raise RuntimeError(f"Missing test jsonl: {TEST_JSONL.resolve()}")

    # --------------------------
    # 1) Locate latest checkpoint
    # --------------------------
    run_dir = latest_run_dir(EXPERIMENT_DIR)
    ckpt_dir = latest_checkpoint_dir(run_dir)

    print("Experiment :", EXPERIMENT_DIR.resolve())
    print("Latest run :", run_dir.name)
    print("Latest ckpt:", ckpt_dir.name)

    # --------------------------
    # 2) Rebuild label encoders from TRAIN
    #    Why:
    #    - LabelEncoder assigns IDs alphabetically.
    #    - The model learned those IDs during training.
    #    - We must reconstruct EXACT same mapping to interpret predictions.
    # --------------------------
    train_rows = load_jsonl(TRAIN_JSONL)
    train_recs = explode_rows(train_rows)

    emotion_enc = LabelEncoder().fit([r["emotion"] for r in train_recs])
    polarity_enc = LabelEncoder().fit([r["polarity"] for r in train_recs])

    EMOTION_NAMES = list(emotion_enc.classes_)
    POLARITY_NAMES = list(polarity_enc.classes_)

    print("\nEmotion labels (ID -> name):")
    for i, n in enumerate(EMOTION_NAMES):
        print(f"  {i} -> {n}")

    print("\nPolarity labels (ID -> name):")
    for i, n in enumerate(POLARITY_NAMES):
        print(f"  {i} -> {n}")

    # --------------------------
    # 3) Build model + load weights
    # --------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = EmotionPolarityModel(
        model_name=MODEL_NAME,
        num_emotions=len(EMOTION_NAMES),
        num_polarity=len(POLARITY_NAMES),
    )

    load_weights(model, ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("\nDevice:", device)

    # --------------------------
    # 4) Load test.jsonl
    # --------------------------
    test_rows = load_jsonl(TEST_JSONL)

    # Lists for quick emotion F1 (on whichever subset we evaluate)
    gold_emotions = []
    pred_emotions = []

    # ==========================================================
    # MODE A: JSONL ITEMS (prints ALL aspects per JSONL item)
    # ==========================================================
    if FIRST_JSONL_ITEMS is not None:
        k = min(FIRST_JSONL_ITEMS, len(test_rows))

        print(f"\n===== FIRST {k} JSONL ITEMS (ALL ASPECTS) + PREDICTIONS =====")

        sample_counter = 0

        # Iterate first k JSONL items (lines)
        for item_idx, item in enumerate(test_rows[:k], start=1):
            text = item["input"]
            outputs = item["output"]  # list of aspect annotations

            print(f"\n================ JSONL ITEM {item_idx:02d} ================")
            print("TEXT:", text)
            print(f"Number of aspects in this item: {len(outputs)}")

            # For each aspect in this JSONL item, run prediction
            for o in outputs:
                sample_counter += 1
                aspect = o["aspect"]
                gold_e = o["emotion"]
                gold_p = o["polarity"]

                e_topk, p_topk = predict_topk(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    aspect=aspect,
                    emotion_names=EMOTION_NAMES,
                    polarity_names=POLARITY_NAMES,
                    device=device,
                    topk=TOPK
                )

                print_and_collect_one_prediction(
                    idx=sample_counter,
                    text=text,
                    aspect=aspect,
                    gold_e=gold_e,
                    gold_p=gold_p,
                    e_topk=e_topk,
                    p_topk=p_topk,
                    gold_emotions=gold_emotions,
                    pred_emotions=pred_emotions
                )

        # Quick F1 over all aspect-predictions printed above
        quick_emotion_f1(gold_emotions, pred_emotions, EMOTION_NAMES)
        print("\nDone.")
        return

    # ==========================================================
    # MODE B: EXPLODED ROWS (old behavior)
    # ==========================================================
    if FIRST_N_EXPLODED is not None:
        test_recs = explode_rows(test_rows)
        n = min(FIRST_N_EXPLODED, len(test_recs))

        print(f"\n===== FIRST {n} EXPLODED TEST ROWS + PREDICTIONS =====")

        for idx, rec in enumerate(test_recs[:n], start=1):
            text = rec["text"]
            aspect = rec["aspect"]
            gold_e = rec["emotion"]
            gold_p = rec["polarity"]

            e_topk, p_topk = predict_topk(
                model=model,
                tokenizer=tokenizer,
                text=text,
                aspect=aspect,
                emotion_names=EMOTION_NAMES,
                polarity_names=POLARITY_NAMES,
                device=device,
                topk=TOPK
            )

            print_and_collect_one_prediction(
                idx=idx,
                text=text,
                aspect=aspect,
                gold_e=gold_e,
                gold_p=gold_p,
                e_topk=e_topk,
                p_topk=p_topk,
                gold_emotions=gold_emotions,
                pred_emotions=pred_emotions
            )

        quick_emotion_f1(gold_emotions, pred_emotions, EMOTION_NAMES)
        print("\nDone.")
        return

    # If neither mode is configured, fail loudly so you notice.
    raise RuntimeError(
        "No evaluation mode selected.\n"
        "Set either FIRST_JSONL_ITEMS (JSONL-level) or FIRST_N_EXPLODED (exploded-level)."
    )


if __name__ == "__main__":
    main()

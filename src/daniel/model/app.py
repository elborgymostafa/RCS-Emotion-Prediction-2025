#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
Flask Demo App — ABSA Emotion + Polarity (classifier_v2)
===========================================================

What this app does (for your presentation):
-------------------------------------------
1) Loads the LATEST checkpoint from:
      results/classifier_v2/run_XXX/checkpoint-YYYY/
   (supports model.safetensors or pytorch_model.bin)

2) Rebuilds LabelEncoders from train.jsonl (so label IDs match training).

3) Web UI:
   - Input: JSONL row id (line index) from test.jsonl
   - Shows: the full text, ALL gold aspect annotations (emotion + polarity)
   - Runs inference for each aspect and shows:
       * top-1 prediction + confidence
       * top-k list (emotion + polarity)
       * full probability distribution (all classes)
   - Shows checkpoint info: run, checkpoint, device, model name, timestamp.

Run:
----
(From project root)
  pip install flask safetensors transformers torch scikit-learn numpy
  python app.py
  open http://127.0.0.1:5000

Notes:
------
- Row id is 0-based by default (row 0 = first line in test.jsonl).
- For presentation: this app is intentionally "pretty" and clear.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from flask import Flask, request, render_template_string

from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel


# ==============================
# CONFIG (edit if needed)
# ==============================

MODEL_NAME = "distilroberta-base"
MAX_LEN = 256

# Your folders (assumes you run from project root)
EXPERIMENT_DIR = Path("results") / "classifier_v2"
DATA_ROOT = Path("data") / "MAMS-ACSA" / "raw" / "data_jsonl" / "annotated"
TRAIN_JSONL = DATA_ROOT / "train.jsonl"
TEST_JSONL  = DATA_ROOT / "test.jsonl"

TOPK = 3  # top-k shown in UI


# ==============================
# Utility: JSONL load
# ==============================

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ==============================
# Utility: latest run + ckpt
# ==============================

def latest_run_dir(exp_dir: Path) -> Path:
    runs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise RuntimeError(f"No run_* folders found in: {exp_dir.resolve()}")
    runs.sort(key=lambda p: int(p.name.split("_")[1]))
    return runs[-1]


def latest_checkpoint_dir(run_dir: Path) -> Path:
    ckpts = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not ckpts:
        raise RuntimeError(f"No checkpoint-* folders found in: {run_dir.resolve()}")
    def step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1
    ckpts.sort(key=step)
    return ckpts[-1]


# ==============================
# Model definition (must match training)
# ==============================

class EmotionPolarityModel(torch.nn.Module):
    """
    Shared encoder + 2 heads:
      - emotion_head: num_emotions
      - polarity_head: num_polarity
    """
    def __init__(self, model_name: str, num_emotions: int, num_polarity: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.emotion_head = torch.nn.Linear(hidden, num_emotions)
        self.polarity_head = torch.nn.Linear(hidden, num_polarity)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return {
            "emotion_logits": self.emotion_head(cls),
            "polarity_logits": self.polarity_head(cls),
        }


def load_weights(model: torch.nn.Module, ckpt_dir: Path) -> str:
    """
    Load weights from checkpoint dir.
    Supports:
      - pytorch_model.bin
      - model.safetensors
    Returns a string describing which file was used.
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
                "Found model.safetensors but safetensors is not installed.\n"
                "Fix: pip install safetensors\n"
                f"Original error: {e}"
            )
        state = load_file(str(safe_path))
        model.load_state_dict(state)
        return safe_path.name

    raise RuntimeError(
        f"Missing weights in {ckpt_dir.resolve()}. "
        f"Found: {[p.name for p in ckpt_dir.iterdir()]}"
    )


# ==============================
# Inference helpers
# ==============================

@torch.no_grad()
def predict_distributions(
    model: EmotionPolarityModel,
    tokenizer,
    text: str,
    aspect: str,
    emotion_names: List[str],
    polarity_names: List[str],
    device: str,
) -> Dict[str, Any]:
    """
    Runs inference for ONE (text, aspect) pair and returns:
      - topk lists
      - full probability distributions
      - top1 label + confidence
      - latency ms
    """
    formatted = f"ASPECT: {aspect} | TEXT: {text}"

    enc = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    t_ms = (time.time() - t0) * 1000.0

    e_probs = torch.softmax(out["emotion_logits"][0], dim=0).cpu().numpy()
    p_probs = torch.softmax(out["polarity_logits"][0], dim=0).cpu().numpy()

    # full distributions as (label, prob)
    e_all = list(zip(emotion_names, [float(x) for x in e_probs]))
    p_all = list(zip(polarity_names, [float(x) for x in p_probs]))

    # top-k
    e_top_idx = np.argsort(-e_probs)[:TOPK]
    p_top_idx = np.argsort(-p_probs)[:TOPK]

    e_topk = [(emotion_names[i], float(e_probs[i])) for i in e_top_idx]
    p_topk = [(polarity_names[i], float(p_probs[i])) for i in p_top_idx]

    return {
        "latency_ms": float(t_ms),
        "input_preview": formatted,
        "emotion": {
            "top1": e_topk[0][0],
            "top1_conf": e_topk[0][1],
            "topk": e_topk,
            "all": sorted(e_all, key=lambda x: -x[1]),
        },
        "polarity": {
            "top1": p_topk[0][0],
            "top1_conf": p_topk[0][1],
            "topk": p_topk,
            "all": sorted(p_all, key=lambda x: -x[1]),
        },
    }


# ==============================
# App boot: load everything ONCE
# ==============================

def build_runtime():
    # 1) find latest ckpt
    run_dir = latest_run_dir(EXPERIMENT_DIR)
    ckpt_dir = latest_checkpoint_dir(run_dir)

    # 2) rebuild label encoders from TRAIN (same mapping as training)
    train_rows = load_jsonl(TRAIN_JSONL)

    # explode for fitting label encoders
    em = []
    pol = []
    for r in train_rows:
        for o in r["output"]:
            em.append(o["emotion"])
            pol.append(o["polarity"])

    emotion_enc = LabelEncoder().fit(em)
    polarity_enc = LabelEncoder().fit(pol)

    emotion_names = list(emotion_enc.classes_)
    polarity_names = list(polarity_enc.classes_)

    # 3) build model + tokenizer + load weights
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmotionPolarityModel(MODEL_NAME, len(emotion_names), len(polarity_names))
    weights_file = load_weights(model, ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # 4) load test rows once (fast UI)
    test_rows = load_jsonl(TEST_JSONL)

    meta = {
        "model_name": MODEL_NAME,
        "device": device,
        "experiment_dir": str(EXPERIMENT_DIR.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "weights_file": weights_file,
        "num_test_rows": len(test_rows),
        "num_emotions": len(emotion_names),
        "num_polarities": len(polarity_names),
        "emotion_labels": emotion_names,
        "polarity_labels": polarity_names,
    }

    return model, tokenizer, test_rows, emotion_names, polarity_names, meta


MODEL, TOKENIZER, TEST_ROWS, EMOTION_NAMES, POLARITY_NAMES, META = build_runtime()


# ==============================
# Flask app + UI template
# ==============================

app = Flask(__name__)

TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ABSA Emotion Demo — classifier_v2</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#0b1220;
      --panel:#121b2f;
      --panel2:#0f172a;
      --text:#e7eefc;
      --muted:#9fb0d0;
      --accent:#7c3aed;
      --good:#22c55e;
      --warn:#f59e0b;
      --bad:#ef4444;
      --line:rgba(255,255,255,0.08);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:var(--sans);
      background: radial-gradient(1200px 800px at 20% 10%, rgba(124,58,237,0.25), transparent 50%),
                  radial-gradient(1000px 700px at 80% 40%, rgba(34,197,94,0.18), transparent 55%),
                  var(--bg);
      color:var(--text);
    }
    .wrap{max-width:1100px;margin:0 auto;padding:24px}
    h1{margin:0 0 6px 0;font-size:28px;letter-spacing:0.2px}
    .subtitle{color:var(--muted);margin:0 0 18px 0}
    .grid{display:grid;grid-template-columns: 1.2fr 0.8fr; gap:14px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      border:1px solid var(--line);
      border-radius:16px;
      padding:16px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .card h2{margin:0 0 10px 0;font-size:16px;color:#dbe7ff}
    .meta{display:grid;grid-template-columns: 1fr 1fr; gap:10px}
    .kv{background:rgba(255,255,255,0.03);border:1px solid var(--line);border-radius:14px;padding:10px}
    .k{font-size:12px;color:var(--muted)}
    .v{font-family:var(--mono);font-size:12px;word-break:break-all;margin-top:4px}
    form{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap}
    label{font-size:12px;color:var(--muted)}
    input[type="number"]{
      width:140px;
      padding:10px 12px;
      border-radius:12px;
      border:1px solid var(--line);
      background:rgba(255,255,255,0.03);
      color:var(--text);
      font-family:var(--mono);
      outline:none;
    }
    input[type="number"]:focus{border-color:rgba(124,58,237,0.7);box-shadow:0 0 0 3px rgba(124,58,237,0.18)}
    .btn{
      padding:10px 14px;
      border-radius:12px;
      border:1px solid rgba(124,58,237,0.55);
      background:linear-gradient(180deg, rgba(124,58,237,0.85), rgba(124,58,237,0.65));
      color:white;
      cursor:pointer;
      font-weight:600;
    }
    .btn:hover{filter:brightness(1.05)}
    .hint{font-size:12px;color:var(--muted);margin-top:6px}
    .textblock{
      font-family:var(--sans);
      line-height:1.45;
      color:#f0f6ff;
      background:rgba(0,0,0,0.20);
      border:1px solid var(--line);
      padding:12px;
      border-radius:14px;
      white-space:pre-wrap;
    }
    .pill{
      display:inline-flex;align-items:center;gap:8px;
      padding:6px 10px;border-radius:999px;
      border:1px solid var(--line);
      background:rgba(255,255,255,0.03);
      font-size:12px;color:#dbe7ff;
    }
    .row{display:flex;gap:10px;flex-wrap:wrap}
    .split{display:grid;grid-template-columns: 1fr 1fr; gap:12px}
    .block{
      background:rgba(255,255,255,0.03);
      border:1px solid var(--line);
      border-radius:16px;
      padding:12px;
    }
    .block h3{margin:0 0 8px 0;font-size:14px;color:#e7eefc}
    .gold{border-left:4px solid rgba(245,158,11,0.8)}
    .pred{border-left:4px solid rgba(34,197,94,0.8)}
    .bad{border-left:4px solid rgba(239,68,68,0.8)}
    .mono{font-family:var(--mono);font-size:12px;color:#dbe7ff}
    .muted{color:var(--muted)}
    .bars{display:flex;flex-direction:column;gap:6px;margin-top:8px}
    .barrow{display:grid;grid-template-columns: 160px 1fr 56px;gap:10px;align-items:center}
    .barlabel{font-family:var(--mono);font-size:12px;color:#dbe7ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .barwrap{height:10px;border-radius:999px;background:rgba(255,255,255,0.07);overflow:hidden;border:1px solid var(--line)}
    .bar{height:100%;border-radius:999px;background:linear-gradient(90deg, rgba(124,58,237,0.9), rgba(34,197,94,0.8))}
    .barpct{font-family:var(--mono);font-size:12px;text-align:right;color:#e7eefc}
    .divider{height:1px;background:var(--line);margin:14px 0}
    .err{color:#ffd6d6;background:rgba(239,68,68,0.14);border:1px solid rgba(239,68,68,0.25);padding:10px;border-radius:12px}
    .footer{margin-top:18px;color:var(--muted);font-size:12px}
    a{color:#bda7ff}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ABSA Emotion Demo <span class="pill">classifier_v2</span></h1>
    <p class="subtitle">Row-based demo: show gold labels (all aspects) + model predictions + full confidence distributions.</p>

    <div class="grid">
      <div class="card">
        <h2>Query</h2>
        <form method="get" action="/">
          <div>
            <label for="row_id">Row id (0-based)</label><br/>
            <input id="row_id" name="row_id" type="number" min="0" max="{{ meta.num_test_rows-1 }}" value="{{ row_id if row_id is not none else 0 }}" />
            <div class="hint">Valid range: 0 … {{ meta.num_test_rows-1 }}. Example: 0, 1, 2 ...</div>
          </div>
          <div>
            <label for="topk">Top-K</label><br/>
            <input id="topk" name="topk" type="number" min="1" max="9" value="{{ topk }}" />
            <div class="hint">How many top labels to list per head.</div>
          </div>
          <button class="btn" type="submit">Run</button>
        </form>

        {% if error %}
          <div class="divider"></div>
          <div class="err"><b>Error:</b> {{ error }}</div>
        {% endif %}

        {% if item %}
          <div class="divider"></div>
          <h2>Text</h2>
          <div class="textblock">{{ item.input }}</div>

          <div class="divider"></div>
          <h2>Aspects in this row</h2>
          <div class="row">
            {% for o in item.output %}
              <span class="pill">Aspect: <span class="mono">{{ o.aspect }}</span></span>
            {% endfor %}
          </div>
        {% endif %}
      </div>

      <div class="card">
        <h2>Runtime / Checkpoint</h2>
        <div class="meta">
          <div class="kv"><div class="k">Model</div><div class="v">{{ meta.model_name }}</div></div>
          <div class="kv"><div class="k">Device</div><div class="v">{{ meta.device }}</div></div>
          <div class="kv"><div class="k">Run dir</div><div class="v">{{ meta.run_dir }}</div></div>
          <div class="kv"><div class="k">Checkpoint</div><div class="v">{{ meta.checkpoint_dir }}</div></div>
          <div class="kv"><div class="k">Weights file</div><div class="v">{{ meta.weights_file }}</div></div>
          <div class="kv"><div class="k">#Test rows</div><div class="v">{{ meta.num_test_rows }}</div></div>
          <div class="kv"><div class="k">#Emotion labels</div><div class="v">{{ meta.num_emotions }}</div></div>
          <div class="kv"><div class="k">#Polarity labels</div><div class="v">{{ meta.num_polarities }}</div></div>
        </div>
        <div class="footer">
          Tip: if you retrain tonight, refresh this page — it auto-picks the newest checkpoint at startup.
        </div>
      </div>
    </div>

    {% if results %}
      <div class="divider"></div>

      <div class="card">
        <h2>Gold vs Prediction (per aspect)</h2>

        {% for r in results %}
          <div class="block {{ 'bad' if (r.gold.emotion != r.pred.emotion.top1) else 'pred' }}">
            <div class="row">
              <span class="pill">Aspect: <span class="mono">{{ r.aspect }}</span></span>
              <span class="pill">Latency: <span class="mono">{{ "%.1f"|format(r.pred.latency_ms) }} ms</span></span>
              <span class="pill">Top-K: <span class="mono">{{ r.topk }}</span></span>
            </div>

            <div class="split" style="margin-top:10px;">
              <div class="block gold">
                <h3>Gold</h3>
                <div class="mono">Emotion: <b>{{ r.gold.emotion }}</b></div>
                <div class="mono">Polarity: <b>{{ r.gold.polarity }}</b></div>
              </div>

              <div class="block pred">
                <h3>Prediction (Top-1)</h3>
                <div class="mono">Emotion: <b>{{ r.pred.emotion.top1 }}</b> <span class="muted">({{ "%.3f"|format(r.pred.emotion.top1_conf) }})</span></div>
                <div class="mono">Polarity: <b>{{ r.pred.polarity.top1 }}</b> <span class="muted">({{ "%.3f"|format(r.pred.polarity.top1_conf) }})</span></div>
              </div>
            </div>

            <div class="split" style="margin-top:10px;">
              <div class="block">
                <h3>Emotion — Top {{ r.topk }}</h3>
                <div class="mono">
                  {% for (lab, pr) in r.pred.emotion.topk %}
                    <div>{{ lab }} : {{ "%.3f"|format(pr) }}</div>
                  {% endfor %}
                </div>
              </div>

              <div class="block">
                <h3>Polarity — Top {{ r.topk }}</h3>
                <div class="mono">
                  {% for (lab, pr) in r.pred.polarity.topk %}
                    <div>{{ lab }} : {{ "%.3f"|format(pr) }}</div>
                  {% endfor %}
                </div>
              </div>
            </div>

            <div class="split" style="margin-top:10px;">
              <div class="block">
                <h3>Emotion — All labels (confidence bars)</h3>
                <div class="bars">
                  {% for (lab, pr) in r.pred.emotion.all %}
                    <div class="barrow">
                      <div class="barlabel" title="{{ lab }}">{{ lab }}</div>
                      <div class="barwrap"><div class="bar" style="width: {{ (pr*100) }}%;"></div></div>
                      <div class="barpct">{{ "%.1f"|format(pr*100) }}%</div>
                    </div>
                  {% endfor %}
                </div>
              </div>

              <div class="block">
                <h3>Polarity — All labels (confidence bars)</h3>
                <div class="bars">
                  {% for (lab, pr) in r.pred.polarity.all %}
                    <div class="barrow">
                      <div class="barlabel" title="{{ lab }}">{{ lab }}</div>
                      <div class="barwrap"><div class="bar" style="width: {{ (pr*100) }}%;"></div></div>
                      <div class="barpct">{{ "%.1f"|format(pr*100) }}%</div>
                    </div>
                  {% endfor %}
                </div>
              </div>
            </div>

            <div class="divider"></div>
            <div class="mono muted">Model input (exact): {{ r.pred.input_preview }}</div>
          </div>
          <div style="height:12px;"></div>
        {% endfor %}
      </div>
    {% endif %}

    <div class="footer">
      Presentation tip: use a few hand-picked row ids that show (1) correct predictions, (2) confusion between negative emotions, (3) “mentioned_only” cases.
    </div>

  </div>
</body>
</html>
"""


# ==============================
# Routes
# ==============================

@app.route("/", methods=["GET"])
def index():
    global TOPK

    row_id_raw = request.args.get("row_id", default=None, type=str)
    topk_raw = request.args.get("topk", default=str(TOPK), type=str)

    error = None
    item = None
    results = None

    # Parse topk safely
    try:
        topk = int(topk_raw)
        if topk < 1:
            topk = 1
        if topk > max(len(EMOTION_NAMES), len(POLARITY_NAMES)):
            topk = max(len(EMOTION_NAMES), len(POLARITY_NAMES))
    except Exception:
        topk = TOPK

    # If row_id not provided, show UI only
    if row_id_raw is None:
        return render_template_string(TEMPLATE, meta=META, item=None, results=None, error=None, row_id=None, topk=topk)

    # Parse row_id
    try:
        row_id = int(row_id_raw)
    except Exception:
        return render_template_string(TEMPLATE, meta=META, item=None, results=None,
                                      error="row_id must be an integer (0-based).", row_id=row_id_raw, topk=topk)

    if row_id < 0 or row_id >= len(TEST_ROWS):
        return render_template_string(
            TEMPLATE,
            meta=META,
            item=None,
            results=None,
            error=f"row_id out of range. Valid: 0..{len(TEST_ROWS)-1}",
            row_id=row_id,
            topk=topk
        )

    # Grab JSONL item
    raw = TEST_ROWS[row_id]
    item = {
        "input": raw.get("input", ""),
        "output": raw.get("output", [])
    }

    # Validate structure
    if not isinstance(item["output"], list) or len(item["output"]) == 0:
        return render_template_string(TEMPLATE, meta=META, item=item, results=None,
                                      error="This row has no outputs/aspects.", row_id=row_id, topk=topk)

    # Run inference for EACH aspect in this JSONL item
    results = []
    for o in item["output"]:
        aspect = o.get("aspect", "")
        gold_emotion = o.get("emotion", "")
        gold_polarity = o.get("polarity", "")

        # Run prediction
        pred = predict_distributions(
            model=MODEL,
            tokenizer=TOKENIZER,
            text=item["input"],
            aspect=aspect,
            emotion_names=EMOTION_NAMES,
            polarity_names=POLARITY_NAMES,
            device=META["device"],
        )

        # Overwrite topk length in output for UI if user changed it
        # (We keep TOPK global as default but allow UI override.)
        # We recompute topk slices here from the full distribution:
        pred["emotion"]["topk"] = pred["emotion"]["all"][:topk]
        pred["emotion"]["top1"] = pred["emotion"]["topk"][0][0]
        pred["emotion"]["top1_conf"] = pred["emotion"]["topk"][0][1]

        pred["polarity"]["topk"] = pred["polarity"]["all"][:topk]
        pred["polarity"]["top1"] = pred["polarity"]["topk"][0][0]
        pred["polarity"]["top1_conf"] = pred["polarity"]["topk"][0][1]

        results.append({
            "aspect": aspect,
            "topk": topk,
            "gold": {"emotion": gold_emotion, "polarity": gold_polarity},
            "pred": pred,
        })

    return render_template_string(
        TEMPLATE,
        meta=META,
        item=item,
        results=results,
        error=error,
        row_id=row_id,
        topk=topk
    )


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    print("\n[ABSA Demo] Loaded:")
    for k, v in META.items():
        if isinstance(v, list):
            continue
        print(f"  {k}: {v}")
    print("\nOpen: http://127.0.0.1:5000\n")

    # debug=True is handy locally (auto reload)
    # For a clean demo: set debug=False
    app.run(host="127.0.0.1", port=5000, debug=True)

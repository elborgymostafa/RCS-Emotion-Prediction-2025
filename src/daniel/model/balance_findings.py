#!/usr/bin/env python
# coding: utf-8

"""
balance_findings.py

Purpose
-------
Investigate dataset imbalance in:
1) Emotion labels (PRIMARY)
2) Polarity labels (SECONDARY)
3) Aspect labels (TERTIARY)
4) Emotion × Aspect interaction (EXPLANATORY)

Outputs
-------
- Printed summaries to terminal
- CSV files with counts and percentages
- TXT file with a clean, paper-ready narrative summary

No class merging. No modeling. Train/val/test kept separate.
"""

import json
from pathlib import Path
import pandas as pd

# ===============================
# 0) PROJECT ROOT (ROBUST)
# ===============================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(50):
        if (cur / ".git").exists():
            return cur
        if (cur / "src").is_dir() and (cur / "data").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError("Project root not found")

HERE = Path(__file__).resolve()
ROOT = find_project_root(HERE.parent)

DATA_ROOT = ROOT / "data" / "MAMS-ACSA" / "raw" / "data_jsonl" / "annotated"
OUT_ROOT = ROOT / "results" / "daniel" / "data_balance"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Project root:", ROOT)
print("Output dir  :", OUT_ROOT)

# ===============================
# 1) LOAD + EXPLODE JSONL
# ===============================
def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

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

splits = {}
for split in ["train", "val", "test"]:
    rows = load_jsonl(DATA_ROOT / f"{split}.jsonl")
    df = explode_rows(rows)
    splits[split] = df
    print(f"{split.upper():5s} samples:", len(df))

# ===============================
# 2) BALANCE ANALYSIS FUNCTIONS
# ===============================
def balance_table(df: pd.DataFrame, column: str):
    counts = df[column].value_counts(dropna=False)
    perc = (counts / counts.sum() * 100).round(2)
    out = pd.DataFrame({
        "count": counts,
        "percent": perc
    })
    return out

# ===============================
# 3) EMOTION BALANCE (PRIMARY)
# ===============================
emotion_tables = {}
for split, df in splits.items():
    emotion_tables[split] = balance_table(df, "emotion")
    print(f"\n=== EMOTION BALANCE ({split.upper()}) ===")
    print(emotion_tables[split])

    emotion_tables[split].to_csv(
        OUT_ROOT / f"emotion_balance_{split}.csv"
    )

# ===============================
# 4) POLARITY BALANCE (SECONDARY)
# ===============================
polarity_tables = {}
for split, df in splits.items():
    polarity_tables[split] = balance_table(df, "polarity")
    print(f"\n=== POLARITY BALANCE ({split.upper()}) ===")
    print(polarity_tables[split])

    polarity_tables[split].to_csv(
        OUT_ROOT / f"polarity_balance_{split}.csv"
    )

# ===============================
# 5) ASPECT BALANCE (TERTIARY)
# ===============================
aspect_tables = {}
for split, df in splits.items():
    aspect_tables[split] = balance_table(df, "aspect")
    print(f"\n=== ASPECT BALANCE ({split.upper()}) ===")
    print(aspect_tables[split])

    aspect_tables[split].to_csv(
        OUT_ROOT / f"aspect_balance_{split}.csv"
    )

# ===============================
# 6) EMOTION × ASPECT INTERACTION
# ===============================
for split, df in splits.items():
    cross = pd.crosstab(df["emotion"], df["aspect"])
    print(f"\n=== EMOTION × ASPECT ({split.upper()}) ===")
    print(cross)

    cross.to_csv(
        OUT_ROOT / f"emotion_aspect_matrix_{split}.csv"
    )

# ===============================
# 7) PAPER-READY TEXT SUMMARY
# ===============================
summary_lines = []
summary_lines.append("DATASET BALANCE ANALYSIS SUMMARY\n")
summary_lines.append("================================\n")

for split in ["train", "val", "test"]:
    et = emotion_tables[split]
    summary_lines.append(f"\n[{split.upper()} SPLIT]\n")

    min_count = et["count"].min()
    max_count = et["count"].max()
    ratio = round(max_count / max(min_count, 1), 2)

    summary_lines.append(
        f"- Emotion imbalance ratio (max/min): {ratio}\n"
    )

    rare = et[et["percent"] < 1.0]
    if not rare.empty:
        summary_lines.append(
            f"- Rare emotion classes (<1%): {', '.join(rare.index.tolist())}\n"
        )
    else:
        summary_lines.append(
            "- No emotion class below 1%\n"
        )

    pt = polarity_tables[split]
    summary_lines.append(
        f"- Polarity distribution: {', '.join([f'{k}={v}%' for k, v in pt['percent'].to_dict().items()])}\n"
    )

summary_path = OUT_ROOT / "balance_summary.txt"
summary_path.write_text("".join(summary_lines), encoding="utf-8")

print("\nWritten summary →", summary_path)

print("\nDONE.")

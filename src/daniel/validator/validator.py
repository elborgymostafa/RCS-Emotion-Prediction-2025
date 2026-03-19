import json
import os
from collections import Counter

# -----------------------------
# Paths
# -----------------------------
GOLD_PATH = os.path.join(data_root, "02_iteration_cleaned_300.jsonl")
PRED_PATH = os.path.join(results_root, "llama", "llama_output_updated.jsonl")
OUT_PATH  = os.path.join(results_root, "audit_aspect_polarity_mismatches.jsonl")

# -----------------------------
# Load JSONL
# -----------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

gold_data = load_jsonl(GOLD_PATH)
pred_data = load_jsonl(PRED_PATH)

assert len(gold_data) == len(pred_data), "Row count mismatch!"

# -----------------------------
# Validation
# -----------------------------
mismatches = []
valid_indices = []

for idx, (gold, pred) in enumerate(zip(gold_data, pred_data)):

    # Rule 1: input text must match exactly
    if gold["input"] != pred["input"]:
        mismatches.append({
            "index": idx,
            "reason": "input_mismatch",
            "gold_input": gold["input"],
            "pred_input": pred["input"]
        })
        continue

    gold_out = gold["output"]
    pred_out = pred["output"]

    # Rule 2: same number of triples
    if len(gold_out) != len(pred_out):
        mismatches.append({
            "index": idx,
            "reason": "output_count_mismatch",
            "input": gold["input"],
            "gold_count": len(gold_out),
            "pred_count": len(pred_out),
            "gold_output": gold_out,
            "pred_output": pred_out
        })
        continue

    # Rule 3: same (aspect, polarity) multiset
    gold_pairs = Counter((o["aspect"], o["polarity"]) for o in gold_out)
    pred_pairs = Counter((o["aspect"], o["polarity"]) for o in pred_out)

    if gold_pairs != pred_pairs:
        mismatches.append({
            "index": idx,
            "reason": "aspect_polarity_mismatch",
            "input": gold["input"],
            "gold_aspect_polarity": [
                {"aspect": a, "polarity": p, "count": c}
                for (a, p), c in gold_pairs.items()
            ],
            "pred_aspect_polarity": [
                {"aspect": a, "polarity": p, "count": c}
                for (a, p), c in pred_pairs.items()
            ],
        })
        continue

    # Passed all checks
    valid_indices.append(idx)

# -----------------------------
# Reporting
# -----------------------------
print("\n==============================")
print("VALIDATION SUMMARY")
print("==============================")
print(f"Total rows checked : {len(gold_data)}")
print(f"Valid rows         : {len(valid_indices)}")
print(f"Invalid rows       : {len(mismatches)}")

# -----------------------------
# Save mismatches
# -----------------------------
if mismatches:
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for m in mismatches:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\nInvalid rows written to:")
    print(OUT_PATH)

# -----------------------------
# Preview first mismatches
# -----------------------------
for m in mismatches[:10]:
    print("\n--- INVALID ROW ---")
    print("Index :", m["index"])
    print("Reason:", m["reason"])
    if "input" in m:
        print("Input :", m["input"])

print("\n==============================")
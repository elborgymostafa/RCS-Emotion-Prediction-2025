import json
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_lookup(data):
    """
    Builds:
    {
        input_text: {
            aspect: emotion
        }
    }
    """
    lookup = {}
    for item in data:
        aspect_map = {}
        for out in item["output"]:
            aspect_map[out["aspect"]] = out["emotion"]
        lookup[item["input"]] = aspect_map
    return lookup


def compute_emotion_f1(gold_path, pred_path):
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)

    gold_lookup = build_lookup(gold_data)
    pred_lookup = build_lookup(pred_data)

    y_true = []
    y_pred = []

    for input_text, gold_aspects in gold_lookup.items():
        if input_text not in pred_lookup:
            continue

        pred_aspects = pred_lookup[input_text]

        for aspect, gold_emotion in gold_aspects.items():
            if aspect not in pred_aspects:
                continue

            y_true.append(gold_emotion)
            y_pred.append(pred_aspects[aspect])

    print("Total compared labels:", len(y_true))

    print("\nMacro F1:",
          f1_score(y_true, y_pred, average="macro"))

    print("Micro F1:",
          f1_score(y_true, y_pred, average="micro"))

    print("Weighted F1:",
          f1_score(y_true, y_pred, average="weighted"))

    print("\nPer-emotion breakdown:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    compute_emotion_f1(
        gold_path="testdata.jsonl",
        pred_path="test_predictions_deberta_nopolarity.jsonl"
    )

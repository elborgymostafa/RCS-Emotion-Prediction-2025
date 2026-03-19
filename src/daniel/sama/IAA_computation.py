import json
import krippendorff
import numpy as np
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------

ASPECT_LIST = [
    "food", "service", "ambience", "price",
    "miscellaneous", "menu", "staff", "place"
]

EMOTION_MAP = {
    "satisfaction": 0,
    "admiration": 1,
    "gratitude": 2,
    "disappointment": 3,
    "annoyance": 4,
    "disgust": 5,
    "neutral": 6,
    "mentioned_only": 7,
    "mixed_emotions": 8
}

NUM_ANNOTATORS = 6


# -----------------------------
# LOAD MULTIPLE ANNOTATOR FILES
# -----------------------------

def load_annotations(file_paths):
    """
    file_paths: list of 6 JSONL files (same 50 reviews, different annotators)
    Returns structured annotation dictionary
    """

    annotator_data = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            annotator_reviews = [json.loads(line) for line in f]
        annotator_data.append(annotator_reviews)

    return annotator_data


# -----------------------------
# BUILD RELIABILITY MATRIX
# -----------------------------

def build_reliability_matrix(annotator_data):
    """
    Returns matrix shape:
    (num_annotators, num_aspect_instances)

    Each column = one review-aspect pair
    """

    num_reviews = len(annotator_data[0])

    # Collect all aspect instances
    aspect_instances = []

    for review_idx in range(num_reviews):
        for aspect in ASPECT_LIST:
            aspect_instances.append((review_idx, aspect))

    matrix = []

    for annotator in annotator_data:
        annotator_labels = []

        for review_idx, aspect in aspect_instances:
            outputs = annotator[review_idx]["output"]

            emotion_label = None
            for out in outputs:
                if out["aspect"].lower() == aspect:
                    emotion_label = EMOTION_MAP[out["emotion"].lower()]
                    break

            if emotion_label is None:
                annotator_labels.append(np.nan)  # missing
            else:
                annotator_labels.append(emotion_label)

        matrix.append(annotator_labels)

    return np.array(matrix)


# -----------------------------
# COMPUTE KRIPPENDORFF'S ALPHA
# -----------------------------

def compute_krippendorff_alpha(matrix):
    alpha = krippendorff.alpha(
        reliability_data=matrix,
        level_of_measurement="nominal"
    )
    return alpha


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    file_paths = [
        "iuliia.jsonl",
        "danila.jsonl",
        "dani.jsonl",
        "sama.jsonl",
        "mustafa.jsonl",
        "mane.jsonl"
    ]

    annotator_data = load_annotations(file_paths)

    matrix = build_reliability_matrix(annotator_data)

    alpha = compute_krippendorff_alpha(matrix)

    print("\nInter-Annotator Agreement (Krippendorff’s Alpha):")
    print(f"Alpha = {alpha:.4f}")
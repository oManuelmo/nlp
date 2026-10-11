import numpy as np
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import LABEL_NAMES


def compute_metrics(eval_prediction, label_names=LABEL_NAMES):
    """
    Compute seqeval NER metrics (precision, recall, f1, accuracy) from EvalPrediction.
    Ignores subwords marked with label ID -100.
    """
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        current_predictions = []
        current_labels = []

        for pred, lab in zip(prediction, label):
            if lab == -100:
                continue

            current_predictions.append(label_names[pred])
            current_labels.append(label_names[lab])

        true_predictions.append(current_predictions)
        true_labels.append(current_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

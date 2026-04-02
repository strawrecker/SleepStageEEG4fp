from __future__ import annotations

from typing import Dict, List

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        matrix[truth, pred] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class_f1: List[float] = []
    per_class_precision: List[float] = []
    per_class_recall: List[float] = []
    support: List[int] = []

    for klass in range(num_classes):
        tp = float(matrix[klass, klass])
        fp = float(matrix[:, klass].sum() - tp)
        fn = float(matrix[klass, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        support.append(int(matrix[klass, :].sum()))

    row_marginals = matrix.sum(axis=1).astype(np.float64)
    col_marginals = matrix.sum(axis=0).astype(np.float64)
    expected = float((row_marginals * col_marginals).sum() / max(total * total, 1))
    observed = accuracy
    kappa = float((observed - expected) / (1.0 - expected)) if abs(1.0 - expected) > 1e-8 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(per_class_f1)),
        "kappa": kappa,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "support": support,
        "confusion_matrix": matrix.tolist(),
    }

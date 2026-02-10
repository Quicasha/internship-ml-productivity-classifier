from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class Metrics:
    accuracy: float
    precision_0: float
    recall_0: float
    f1_0: float
    precision_1: float
    recall_1: float
    f1_1: float
    support_0: int
    support_1: int
    tn: int
    fp: int
    fn: int
    tp: int

    def to_row(self, model_name: str) -> Dict[str, Any]:
        return {
            "model": model_name,
            "accuracy": self.accuracy,
            "precision_0": self.precision_0,
            "recall_0": self.recall_0,
            "f1_0": self.f1_0,
            "precision_1": self.precision_1,
            "recall_1": self.recall_1,
            "f1_1": self.f1_1,
            "support_0": self.support_0,
            "support_1": self.support_1,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def compute_metrics(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p0 = float(precision_score(y_true, y_pred, pos_label=0))
    r0 = float(recall_score(y_true, y_pred, pos_label=0))
    f0 = float(f1_score(y_true, y_pred, pos_label=0))

    p1 = float(precision_score(y_true, y_pred, pos_label=1))
    r1 = float(recall_score(y_true, y_pred, pos_label=1))
    f1 = float(f1_score(y_true, y_pred, pos_label=1))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    support_0 = int(np.sum(y_true == 0))
    support_1 = int(np.sum(y_true == 1))

    return Metrics(
        accuracy=acc,
        precision_0=p0, recall_0=r0, f1_0=f0,
        precision_1=p1, recall_1=r1, f1_1=f1,
        support_0=support_0, support_1=support_1,
        tn=tn, fp=fp, fn=fn, tp=tp,
    )


def pretty_print(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, digits=2, zero_division=0))
"""Classification metrics for one test fold. Spec: docs/benchmarking_protocol.md section 2.

AUROC is the primary metric because it is threshold-free and robust to the 85/15 class imbalance.
The threshold metrics are reported twice: at the fixed 0.5 cut-off (comparable across models) and
at the Youden-optimal cut-off chosen on the inner validation set (never on the test fold, which
would leak).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def threshold_metrics(y_true, y_prob, threshold):
    """Metrics that depend on a cut-off. Abnormal = 1, so sensitivity is Abnormal recall."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def youden_threshold(y_true, y_prob):
    """Cut-off that maximises sensitivity + specificity - 1 (0.5 if only one class is present)."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[best], 0.0, 1.0))


def evaluate(y_true, y_prob, val_true=None, val_prob=None):
    """Full metric dict for one fold: AUROC, AUPRC, and threshold metrics at 0.5 and Youden."""
    y_true = np.asarray(y_true).astype(int)
    both = y_true.min() != y_true.max()
    result = {
        "n": int(len(y_true)),
        "n_abnormal": int(y_true.sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "at_0.5": threshold_metrics(y_true, y_prob, 0.5),
    }
    if val_true is not None and val_prob is not None:
        result["at_youden_val"] = threshold_metrics(y_true, y_prob, youden_threshold(val_true, val_prob))
    return result

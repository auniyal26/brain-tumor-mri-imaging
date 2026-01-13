from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    precision_score,
)

import torch
from torch.utils.data import DataLoader


def eval_argmax(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard eval: argmax on logits.
    Returns (y_true, y_pred) as numpy arrays.
    """
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            yp.append(pred)
            yt.append(np.array(y))
    return np.concatenate(yt), np.concatenate(yp)


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() * 0.6 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def metrics_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    num_classes = len(class_names)
    per_class_recall = recall_score(
        y_true, y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(num_classes)},
    }


def metrics_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["no_tumor", "tumor"],
) -> Dict:
    # y_true/y_pred in {0,1}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_no_tumor": float(precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "recall_no_tumor": float(recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "precision_tumor": float(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "recall_tumor": float(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
    }


def write_report_txt(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rep)


# --------- self-check ----------
if __name__ == "__main__":
    # import sanity + tiny CM render
    cm = np.array([[5, 1], [2, 7]])
    save_confusion_matrix_png(cm, ["no_tumor", "tumor"], "./_tmp_run/cm.png", "test cm")
    print("Wrote ./_tmp_run/cm.png")
    print("OK: eval.py sanity check passed")

# Compatibility for Python 3.9 type hints
from typing import List, Optional
# src/training/metrics.py

import torch
from sklearn.metrics import cohen_kappa_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_class_ids: Optional[List[int]] = None,
) -> float:
    """
    Pixel-wise accuracy computed on valid pixels only (excluding ignore_class_ids).

    Parameters
    ----------
    logits : torch.Tensor
        Shape (B, C, H, W)
    targets : torch.Tensor
        Shape (B, H, W)
    ignore_class_ids : list or None
        List of class IDs to ignore

    Returns
    -------
    float
        Accuracy in percentage.
    """
    if ignore_class_ids is None:
        ignore_class_ids = []

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)

        ignore_tensor = torch.tensor(
            ignore_class_ids,
            device=targets.device,
            dtype=targets.dtype,
        )

        valid_mask = ~torch.isin(targets, ignore_tensor)
        if valid_mask.sum() == 0:
            return 0.0

        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        acc = (correct / total) * 100.0 if total > 0 else 0.0
    return acc


def kappa_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_class_ids: Optional[List[int]] = None,
) -> float:
    """
    Cohen's Kappa computed on valid pixels only (excluding ignore_class_ids).

    logits: (B, C, H, W)
    targets: (B, H, W)
    """
    if ignore_class_ids is None:
        ignore_class_ids = []

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)

        ignore_tensor = torch.tensor(
            ignore_class_ids,
            device=targets.device,
            dtype=targets.dtype,
        )

        valid_mask = ~torch.isin(targets, ignore_tensor)
        if valid_mask.sum() == 0:
            return 0.0

        y_true = targets[valid_mask].cpu().numpy().ravel()
        y_pred = preds[valid_mask].cpu().numpy().ravel()

        kappa = cohen_kappa_score(y_true, y_pred)

        if kappa is None or not torch.isfinite(torch.tensor(kappa)):
            return None

        return float(kappa)


def f1_macro_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_class_ids: Optional[List[int]] = None,
) -> float:
    """
    Macro F1-score computed on valid pixels only (excluding ignore_class_ids).

    logits: (B, C, H, W)
    targets: (B, H, W)
    """
    if ignore_class_ids is None:
        ignore_class_ids = []

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)

        ignore_tensor = torch.tensor(
            ignore_class_ids,
            device=targets.device,
            dtype=targets.dtype,
        )

        valid_mask = ~torch.isin(targets, ignore_tensor)
        if valid_mask.sum() == 0:
            return 0.0

        y_true = targets[valid_mask].cpu().numpy().ravel()
        y_pred = preds[valid_mask].cpu().numpy().ravel()

        return float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

def confusion_matrix_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_class_ids: Optional[List[int]] = None,
    normalize: bool = False,
    labels: Optional[List[int]] = None,
):
    """
    Confusion matrix computed on valid pixels only (excluding ignore_class_ids).

    Parameters
    ----------
    logits : torch.Tensor
        Shape (B, C, H, W)
    targets : torch.Tensor
        Shape (B, H, W)
    normalize : bool
        If True, rows are normalized to sum to 1
    labels : list or None
        Explicit class labels (e.g. [1,2,...,10])
    ignore_class_ids : list or None
        List of class IDs to ignore

    Returns
    -------
    np.ndarray or None
        Confusion matrix (K x K) or None if no valid pixels
    """
    if ignore_class_ids is None:
        ignore_class_ids = []

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # (B,H,W) or already flat

        # --- make shapes robust ---
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        ignore_tensor = torch.tensor(
            ignore_class_ids,
            device=targets.device,
            dtype=targets.dtype,
        )

        valid_mask = ~torch.isin(targets, ignore_tensor)
        if valid_mask.sum() == 0:
            return None

        y_true = targets[valid_mask].cpu().numpy()
        y_pred = preds[valid_mask].cpu().numpy()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm = cm / row_sums

        return cm

def miou_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_class_ids: Optional[List[int]] = None,
    labels: Optional[List[int]] = None,
) -> float:
    """
    Mean Intersection over Union (mIoU) computed on valid pixels only (excluding ignore_class_ids).

    Returns
    -------
    float
        mIoU over classes present in labels
    """
    if ignore_class_ids is None:
        ignore_class_ids = []

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)

        ignore_tensor = torch.tensor(
            ignore_class_ids,
            device=targets.device,
            dtype=targets.dtype,
        )

        valid_mask = ~torch.isin(targets, ignore_tensor)
        if valid_mask.sum() == 0:
            return 0.0

        y_true = targets[valid_mask].cpu().numpy().ravel()
        y_pred = preds[valid_mask].cpu().numpy().ravel()

        if labels is None:
            labels = np.unique(y_true)

        ious = []
        for c in labels:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))

            denom = tp + fp + fn
            if denom == 0:
                continue

            ious.append(tp / denom)

        if len(ious) == 0:
            return 0.0

        return float(np.mean(ious))
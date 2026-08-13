# src/evaluation/metrics.py

import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import confusion_matrix, f1_score


def mask_valid_pixels(
    gt: np.ndarray,
    ignore_class_ids: List[int],
) -> np.ndarray:
    """
    Build validity mask based ONLY on ground truth.

    Returns
    -------
    valid_mask : np.ndarray (H, W), bool
        True for pixels to consider
    """
    valid_mask = np.ones_like(gt, dtype=bool)
    for cid in ignore_class_ids:
        valid_mask &= (gt != cid)
    return valid_mask


def compute_tile_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_class_ids: List[int],
    labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute tile-level metrics ignoring selected GT classes.

    Parameters
    ----------
    pred : np.ndarray (H, W)
        Predicted class IDs
    gt : np.ndarray (H, W)
        Ground truth class IDs
    ignore_class_ids : list[int]
        Class IDs to ignore (based on GT)
    labels : list[int], optional
        Explicit class labels to evaluate (excluding ignored ones)

    Returns
    -------
    metrics : dict
        OA, F1_macro, mIoU
    """
    valid_mask = mask_valid_pixels(gt, ignore_class_ids)

    if valid_mask.sum() == 0:
        return {"OA": 0.0, "F1_macro": 0.0, "mIoU": 0.0}

    gt_v = gt[valid_mask].ravel()
    pred_v = pred[valid_mask].ravel()

    if labels is None:
        labels = np.unique(gt_v)

    # ---------- Overall Accuracy ----------
    oa = float((pred_v == gt_v).mean())

    # ---------- Macro F1 ----------
    f1 = float(
        f1_score(
            gt_v,
            pred_v,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    # ---------- mIoU ----------
    cm = confusion_matrix(gt_v, pred_v, labels=labels)

    ious = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)

    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    return {
        "OA": oa,
        "F1_macro": f1,
        "mIoU": miou,
    }


def compute_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_class_ids: List[int],
    labels: List[int],
    normalize: bool = False,
) -> Optional[np.ndarray]:
    """
    Confusion matrix at tile level with GT-driven masking.
    """
    valid_mask = mask_valid_pixels(gt, ignore_class_ids)

    if valid_mask.sum() == 0:
        return None

    gt_v = gt[valid_mask].ravel()
    pred_v = pred[valid_mask].ravel()

    cm = confusion_matrix(gt_v, pred_v, labels=labels)

    if normalize:
        cm = cm.astype(np.float32)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm /= row_sum

    return cm
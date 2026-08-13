import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from src.data.dataset import S1Dataset
from src.data.legend import load_legend
from src.data.patching import patch_tensors
from src.models.swin_unetr import build_swin_unetr
from src.utils.config import load_config


PROJECT_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = Path(
    "/home/tlcrs/Philippines_Project/09_Luigi Russo/"
    "DATASET_PHILIPPINES/checkpoints_cc50_weighted_sampler_10x/"
    "philippines/best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "cc50_weighted_sampler_10x"
    / "evaluation"
)

REGION = "philippines"
SHRUBLAND_ID = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def metrics_from_confusion_matrix(
    cm: np.ndarray,
    evaluated_class_ids,
):
    per_class = {}

    for class_id in evaluated_class_ids:
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)
        support = int(cm[class_id, :].sum())
        predicted = int(cm[:, class_id].sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )
        iou = safe_divide(tp, tp + fp + fn)

        per_class[class_id] = {
            "support": support,
            "predicted_pixels": predicted,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    present_classes = [
        metrics
        for metrics in per_class.values()
        if metrics["support"] > 0
    ]

    if present_classes:
        macro_precision = float(
            np.mean([m["precision"] for m in present_classes])
        )
        macro_recall = float(
            np.mean([m["recall"] for m in present_classes])
        )
        macro_f1 = float(
            np.mean([m["f1"] for m in present_classes])
        )
        mean_iou = float(
            np.mean([m["iou"] for m in present_classes])
        )
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        mean_iou = 0.0

    total_pixels = int(cm.sum())
    correct_pixels = int(np.trace(cm))
    overall_accuracy = safe_divide(correct_pixels, total_pixels)

    summary = {
        "total_valid_pixels": total_pixels,
        "correct_pixels": correct_pixels,
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
    }

    return per_class, summary


def save_matrix_csv(
    matrix: np.ndarray,
    class_names,
    output_path: Path,
    normalized: bool,
):
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)

        column_names = [
            f"{class_id}:{class_names[class_id]}"
            for class_id in range(len(class_names))
        ]

        writer.writerow(["true\\pred"] + column_names)

        for class_id, row in enumerate(matrix):
            if normalized:
                values = [f"{float(value):.8f}" for value in row]
            else:
                values = [int(value) for value in row]

            writer.writerow(
                [f"{class_id}:{class_names[class_id]}"] + values
            )


def save_confusion_plot(
    matrix: np.ndarray,
    class_names,
    output_path: Path,
    title: str,
    annotate: bool,
):
    labels = [
        f"{class_id}\n{name}"
        for class_id, name in enumerate(class_names)
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def canonical_tile_id(gt_path: str) -> str:
    stem = Path(gt_path).stem
    return stem.replace("_GT_", "_", 1)


@torch.inference_mode()
def main():
    os.chdir(PROJECT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"CC50 checkpoint does not exist: {CHECKPOINT_PATH}"
        )

    dataset_cfg = load_config("configs/dataset.yaml")
    training_cfg = load_config("configs/training.yaml")
    legend = load_legend("configs/legend.yaml")

    class_names = legend.class_names
    num_classes = legend.num_classes

    ignore_class_ids = list(
        training_cfg["loss"]["ignore_class_ids"]
    )

    evaluated_class_ids = [
        class_id
        for class_id in range(num_classes)
        if class_id not in ignore_class_ids
    ]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== CC50 SAMPLER-10X BEST-CHECKPOINT EVALUATION ===")
    print("Project:", PROJECT_DIR)
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Output:", OUTPUT_DIR)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    region_root = (
        Path(dataset_cfg["dataset"]["base_path"])
        / REGION
    )

    split_path = (
        Path(dataset_cfg["derived"]["splits_base_dir"])
        / REGION
        / dataset_cfg["derived"]["tile_split_file"]
    )

    dataset = S1Dataset(
        root_s1=str(region_root / "s1"),
        root_gt=str(region_root / "ground_reference"),
        root_labels=str(region_root / "labels"),
        t_len=dataset_cfg["temporal"]["t_len"],
        legend=legend,
        out_size=tuple(dataset_cfg["spatial"]["out_size"]),
        polarizations=dataset_cfg["spatial"]["polarizations"],
        return_y_long=True,
        patch_size=None,
        stride=None,
        tile_split_path=str(split_path),
        augmentation_cfg=None,
        split="val",
    )

    print("Validation scenes:", len(dataset))

    patch_size = tuple(
        dataset_cfg["patching"]["patch_size"]
    )
    stride = int(dataset_cfg["patching"]["stride"])

    patch_batch_size = max(
        1,
        int(dataset_cfg["dataloader"]["batch_size"]),
    )

    in_channels = (
        dataset_cfg["temporal"]["t_len"]
        * len(dataset_cfg["spatial"]["polarizations"])
    )

    model = build_swin_unetr(
        img_size=patch_size,
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=48,
        spatial_dims=2,
        device=device,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint dictionary, "
            f"found {type(checkpoint)}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "model_state_dict was not found in checkpoint"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint monitor:", checkpoint.get("monitor"))
    print("Checkpoint best metric:", checkpoint.get("best_metric"))
    print("Patch size:", patch_size)
    print("Stride:", stride)
    print("Patch inference batch size:", patch_batch_size)

    global_cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    all_true = []
    all_pred = []
    per_tile_rows = []

    for dataset_index in tqdm(
        range(len(dataset)),
        desc="Evaluating validation scenes",
        ncols=100,
    ):
        x, y = dataset[dataset_index]

        if x.ndim != 3:
            raise RuntimeError(
                f"Expected X shape (C,H,W), found {tuple(x.shape)}"
            )

        if y.ndim != 2:
            raise RuntimeError(
                f"Expected Y shape (H,W), found {tuple(y.shape)}"
            )

        x_batch = x.unsqueeze(0)

        _, _, height, width = x_batch.shape
        patch_height, patch_width = patch_size

        coordinates = [
            (row, column)
            for row in range(
                0,
                height - patch_height + 1,
                stride,
            )
            for column in range(
                0,
                width - patch_width + 1,
                stride,
            )
        ]

        x_patches = patch_tensors(
            x_batch,
            patch_size,
            stride,
        )

        if len(coordinates) != x_patches.shape[0]:
            raise RuntimeError(
                "Patch-coordinate mismatch: "
                f"{len(coordinates)} coordinates versus "
                f"{x_patches.shape[0]} patches"
            )

        probability_sum = torch.zeros(
            (num_classes, height, width),
            dtype=torch.float32,
            device=device,
        )

        probability_count = torch.zeros(
            (1, height, width),
            dtype=torch.float32,
            device=device,
        )

        for start in range(
            0,
            x_patches.shape[0],
            patch_batch_size,
        ):
            end = min(
                start + patch_batch_size,
                x_patches.shape[0],
            )

            patch_batch = x_patches[start:end].to(
                device,
                non_blocking=False,
            )

            logits = model(patch_batch)
            probabilities = F.softmax(logits, dim=1)

            for local_index, (
                row,
                column,
            ) in enumerate(coordinates[start:end]):
                probability_sum[
                    :,
                    row:row + patch_height,
                    column:column + patch_width,
                ] += probabilities[local_index]

                probability_count[
                    :,
                    row:row + patch_height,
                    column:column + patch_width,
                ] += 1

        if float(probability_count.min().item()) <= 0:
            raise RuntimeError(
                "At least one tile pixel was not covered by patches"
            )

        probability_tile = (
            probability_sum / probability_count
        )

        prediction = torch.argmax(
            probability_tile,
            dim=0,
        ).cpu().numpy().astype(np.int64)

        ground_truth = y.cpu().numpy().astype(np.int64)

        valid_mask = ~np.isin(
            ground_truth,
            ignore_class_ids,
        )

        true_valid = ground_truth[valid_mask]
        pred_valid = prediction[valid_mask]

        tile_cm = confusion_matrix(
            true_valid,
            pred_valid,
            labels=list(range(num_classes)),
        )

        global_cm += tile_cm

        all_true.append(true_valid)
        all_pred.append(pred_valid)

        tile_per_class, tile_summary = (
            metrics_from_confusion_matrix(
                tile_cm,
                evaluated_class_ids,
            )
        )

        gt_path = dataset.pairs[dataset_index][0]
        tile_id = canonical_tile_id(gt_path)

        shrubland_metrics = tile_per_class[
            SHRUBLAND_ID
        ]

        per_tile_rows.append(
            {
                "tile_id": tile_id,
                "valid_pixels": tile_summary[
                    "total_valid_pixels"
                ],
                "overall_accuracy": tile_summary[
                    "overall_accuracy"
                ],
                "macro_f1": tile_summary["macro_f1"],
                "mean_iou": tile_summary["mean_iou"],
                "shrubland_support": shrubland_metrics[
                    "support"
                ],
                "shrubland_predicted_pixels": shrubland_metrics[
                    "predicted_pixels"
                ],
                "shrubland_true_positive": shrubland_metrics[
                    "true_positive"
                ],
                "shrubland_false_positive": shrubland_metrics[
                    "false_positive"
                ],
                "shrubland_false_negative": shrubland_metrics[
                    "false_negative"
                ],
                "shrubland_precision": shrubland_metrics[
                    "precision"
                ],
                "shrubland_recall": shrubland_metrics[
                    "recall"
                ],
                "shrubland_f1": shrubland_metrics["f1"],
                "shrubland_iou": shrubland_metrics["iou"],
            }
        )

        del (
            x_batch,
            x_patches,
            probability_sum,
            probability_count,
            probability_tile,
        )

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    per_class_metrics, global_metrics = (
        metrics_from_confusion_matrix(
            global_cm,
            evaluated_class_ids,
        )
    )

    global_metrics["cohen_kappa"] = float(
        cohen_kappa_score(
            all_true,
            all_pred,
        )
    )

    global_metrics[
        "predicted_nodata_on_valid_gt"
    ] = int((all_pred == 0).sum())

    row_sums = global_cm.sum(
        axis=1,
        keepdims=True,
    ).astype(np.float64)

    normalized_cm = np.divide(
        global_cm,
        row_sums,
        out=np.zeros_like(
            global_cm,
            dtype=np.float64,
        ),
        where=row_sums != 0,
    )

    per_class_json = {}

    for class_id in evaluated_class_ids:
        per_class_json[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_names[class_id],
            **per_class_metrics[class_id],
        }

    shrubland_metrics = per_class_metrics[
        SHRUBLAND_ID
    ]

    shrubland_tile_rows = [
        row
        for row in per_tile_rows
        if row["shrubland_support"] > 0
    ]

    summary = {
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": sha256_file(CHECKPOINT_PATH),
            "saved_epoch": checkpoint.get("epoch"),
            "monitor": checkpoint.get("monitor"),
            "best_training_metric": checkpoint.get(
                "best_metric"
            ),
        },
        "dataset": {
            "base_path": dataset_cfg["dataset"][
                "base_path"
            ],
            "region": REGION,
            "validation_scenes": len(dataset),
            "patch_size": list(patch_size),
            "stride": stride,
            "ignore_class_ids": ignore_class_ids,
        },
        "global_metrics_unique_validation_pixels": (
            global_metrics
        ),
        "per_class_metrics": per_class_json,
        "shrubland_summary": {
            "class_id": SHRUBLAND_ID,
            "class_name": class_names[SHRUBLAND_ID],
            **shrubland_metrics,
            "validation_tiles_with_shrubland": len(
                shrubland_tile_rows
            ),
        },
        "important_note": (
            "These are global unique-pixel metrics after "
            "reconstructing each 512x512 validation scene from "
            "overlapping patches. They are not expected to be "
            "numerically identical to the training log, which "
            "averages metrics over validation patches."
        ),
    }

    summary_path = OUTPUT_DIR / "evaluation_summary.json"

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    per_class_csv = OUTPUT_DIR / "per_class_metrics.csv"

    with per_class_csv.open("w", newline="") as file:
        fieldnames = [
            "class_id",
            "class_name",
            "support",
            "predicted_pixels",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "iou",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for class_id in evaluated_class_ids:
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    **per_class_metrics[class_id],
                }
            )

    per_tile_csv = OUTPUT_DIR / "per_tile_metrics.csv"

    with per_tile_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_tile_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_tile_rows)

    shrubland_tiles_csv = (
        OUTPUT_DIR / "shrubland_validation_tiles.csv"
    )

    with shrubland_tiles_csv.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_tile_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(shrubland_tile_rows)

    np.save(
        OUTPUT_DIR / "confusion_matrix_counts.npy",
        global_cm,
    )

    np.save(
        OUTPUT_DIR
        / "confusion_matrix_row_normalized.npy",
        normalized_cm,
    )

    save_matrix_csv(
        global_cm,
        class_names,
        OUTPUT_DIR / "confusion_matrix_counts.csv",
        normalized=False,
    )

    save_matrix_csv(
        normalized_cm,
        class_names,
        OUTPUT_DIR
        / "confusion_matrix_row_normalized.csv",
        normalized=True,
    )

    save_confusion_plot(
        global_cm,
        class_names,
        OUTPUT_DIR / "confusion_matrix_counts.png",
        "CC50 sampler-10x validation confusion matrix — pixel counts",
        annotate=False,
    )

    save_confusion_plot(
        normalized_cm,
        class_names,
        OUTPUT_DIR
        / "confusion_matrix_row_normalized.png",
        "CC50 sampler-10x validation confusion matrix — row normalized",
        annotate=True,
    )

    print("\n=== GLOBAL UNIQUE-PIXEL RESULTS ===")
    print(
        "Overall accuracy:",
        f"{global_metrics['overall_accuracy']:.6f}",
    )
    print(
        "Macro precision:",
        f"{global_metrics['macro_precision']:.6f}",
    )
    print(
        "Macro recall:",
        f"{global_metrics['macro_recall']:.6f}",
    )
    print(
        "Macro F1:",
        f"{global_metrics['macro_f1']:.6f}",
    )
    print(
        "Mean IoU:",
        f"{global_metrics['mean_iou']:.6f}",
    )
    print(
        "Cohen kappa:",
        f"{global_metrics['cohen_kappa']:.6f}",
    )

    print("\n=== SHRUBLAND ===")
    print("GT support:", shrubland_metrics["support"])
    print(
        "Predicted pixels:",
        shrubland_metrics["predicted_pixels"],
    )
    print(
        "True positives:",
        shrubland_metrics["true_positive"],
    )
    print(
        "False positives:",
        shrubland_metrics["false_positive"],
    )
    print(
        "False negatives:",
        shrubland_metrics["false_negative"],
    )
    print(
        "Precision:",
        f"{shrubland_metrics['precision']:.6f}",
    )
    print(
        "Recall:",
        f"{shrubland_metrics['recall']:.6f}",
    )
    print(
        "F1:",
        f"{shrubland_metrics['f1']:.6f}",
    )
    print(
        "IoU:",
        f"{shrubland_metrics['iou']:.6f}",
    )
    print(
        "Validation tiles containing Shrubland:",
        len(shrubland_tile_rows),
    )

    if shrubland_metrics["support"] != 3716:
        print(
            "\n[WARNING] Expected 3716 unique validation "
            "Shrubland pixels from the pretraining audit, "
            f"but evaluation found {shrubland_metrics['support']}."
        )
    else:
        print(
            "\n[CHECK] Shrubland GT support agrees with "
            "the pretraining audit: 3716 pixels."
        )

    print("\nEvaluation outputs:")
    for output_path in sorted(OUTPUT_DIR.iterdir()):
        print(" ", output_path)

    print("\nCC50 SAMPLER-10X BEST-CHECKPOINT EVALUATION COMPLETED")
    

if __name__ == "__main__":
    main()

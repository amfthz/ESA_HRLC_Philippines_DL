import argparse
import os
import json
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from src.utils.config import load_config
from src.data.dataset import S1Dataset
from src.data.ram_builder import build_patch_dataset_in_ram
from src.data.legend import load_legend
from src.models.swin_unetr import build_swin_unetr


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Swin-UNETR models on validation tiles")
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Override dataset.base_path from configs/dataset.yaml"
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="Override dataset.regions from configs/dataset.yaml (e.g. amazonia africa)"
    )
    return parser.parse_args()


# --------------------------------------------------
# Device
# --------------------------------------------------
def get_device(cfg):
    if cfg["device"]["prefer_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg["device"]["prefer_mps"] and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def compute_metrics(y_true, y_pred, num_classes, ignore_ids):
    mask = ~np.isin(y_true, ignore_ids)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Producer Accuracy = recall per class
    producer_acc = {}
    for c in range(num_classes):
        if c in ignore_ids:
            continue
        denom = cm[c, :].sum()
        producer_acc[c] = cm[c, c] / denom if denom > 0 else np.nan

    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return producer_acc, oa, kappa, f1


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    args = parse_args()

    dataset_cfg = load_config("configs/dataset.yaml")
    train_cfg = load_config("configs/training.yaml")
    legend = load_legend("configs/legend.yaml")

    if args.base_path is not None:
        dataset_cfg["dataset"]["base_path"] = args.base_path

    if args.regions is not None:
        dataset_cfg["dataset"]["regions"] = args.regions

    regions = dataset_cfg["dataset"].get("regions")
    if regions is None:
        raise ValueError("dataset.regions not found in dataset.yaml")

    device = get_device(train_cfg)
    print(f"Using device: {device}")

    ignore_class_ids = train_cfg["loss"]["ignore_class_ids"]
    num_classes = legend.num_classes

    results = {}

    # ==================================================
    # EVAL PER REGION
    # ==================================================
    for region in regions:
        print(f"\n[EVAL] Region: {region}")

        base_path = dataset_cfg["dataset"]["base_path"]
        region_root = os.path.join(base_path, region)

        root_s1 = os.path.join(region_root, "s1")
        root_gt = os.path.join(region_root, "ground_reference")
        root_labels = os.path.join(region_root, "labels")

        split_json = f"configs/splits/{region}/tile_split.json"

        # -------------------------
        # DATASET (VAL ONLY)
        # -------------------------
        val_dataset = S1Dataset(
            root_s1=root_s1,
            root_gt=root_gt,
            root_labels=root_labels,
            t_len=dataset_cfg["temporal"]["t_len"],
            legend=legend,
            out_size=tuple(dataset_cfg["spatial"]["out_size"]),
            polarizations=dataset_cfg["spatial"]["polarizations"],
            return_y_long=False,
            patch_size=tuple(dataset_cfg["patching"]["patch_size"])
                if dataset_cfg["patching"]["enabled"] else None,
            stride=dataset_cfg["patching"]["stride"]
                if dataset_cfg["patching"]["enabled"] else None,
            tile_split_path=split_json,
            split="val",
            augmentation_cfg=None,
        )

        X_val, y_val = build_patch_dataset_in_ram(val_dataset)

        valloader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=dataset_cfg["dataloader"]["batch_size"],
            shuffle=False,
            num_workers=dataset_cfg["dataloader"]["num_workers"],
        )

        # -------------------------
        # MODEL
        # -------------------------
        in_channels = dataset_cfg["temporal"]["t_len"] * len(dataset_cfg["spatial"]["polarizations"])

        model = build_swin_unetr(
            img_size=dataset_cfg["patching"]["patch_size"],
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
            spatial_dims=2,
            device=device,
        )

        checkpoint_dir = os.path.join(
            train_cfg["checkpoint"]["base_dir"],
            region,
        )

        ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[CHECKPOINT] Loaded epoch {checkpoint.get('epoch')} | best {checkpoint.get('monitor')} = {checkpoint.get('best_metric')}")
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        # -------------------------
        # INFERENCE
        # -------------------------
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for x, y in valloader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                y_true_all.append(y.cpu().numpy().ravel())
                y_pred_all.append(preds.cpu().numpy().ravel())

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        # -------------------------
        # METRICS
        # -------------------------
        pa, oa, kappa, f1 = compute_metrics(
            y_true_all,
            y_pred_all,
            num_classes=num_classes,
            ignore_ids=ignore_class_ids,
        )

        results[region] = {
            "producer_accuracy": pa,
            "OA": oa,
            "kappa": kappa,
            "F1_macro": f1,
        }

        print(f"[RESULTS] OA={oa:.4f} | Kappa={kappa:.4f} | F1={f1:.4f}")

    # ==================================================
    # SAVE RESULTS
    # ==================================================
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[EVAL DONE] Results saved to {out_path}")


if __name__ == "__main__":
    main()
import argparse
# --------------------------------------------------
# Argument parsing for CLI overrides
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-UNETR model for ESA CCI LC mapping")
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
        help="Override dataset.regions from configs/dataset.yaml (e.g. amazonia africa siberia)"
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default=None,
        help="Override training.checkpoint.base_dir from configs/training.yaml"
    )
    return parser.parse_args()
# src/train.py

import torch
import torch.nn as nn
from torch.optim import Adam

import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.utils.config import load_config
from src.data.dataset import S1Dataset
from src.data.ram_builder import build_patch_dataset_in_ram
from src.data.legend import load_legend
from src.data.run_tile_split import run_tile_split
from src.training.trainer import train


# --------------------------------------------------
# Device selection (CUDA > MPS > CPU)
# --------------------------------------------------
def get_device(cfg):
    if cfg["device"]["prefer_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg["device"]["prefer_mps"] and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    # --------------------------------------------------
    # LOAD CONFIGS
    # --------------------------------------------------
    dataset_cfg = load_config("configs/dataset.yaml")
    train_cfg = load_config("configs/training.yaml")
    legend = load_legend("configs/legend.yaml")

    # --------------------------------------------------
    # CLI overrides
    # --------------------------------------------------
    if args.base_path is not None:
        dataset_cfg["dataset"]["base_path"] = args.base_path

    if args.regions is not None:
        dataset_cfg["dataset"]["regions"] = args.regions

    # Override checkpoint base dir (training.yaml)
    if args.checkpoint_base_dir is not None:
        train_cfg.setdefault("checkpoint", {})
        train_cfg["checkpoint"]["base_dir"] = args.checkpoint_base_dir

    regions = dataset_cfg["dataset"].get("regions")
    if regions is None:
        raise ValueError("dataset.regions not found in dataset.yaml")

    # ==================================================
    # STEP 1 — PREPARE DATA ONCE (ALL REGIONS)
    # ==================================================
    print("[SETUP] Ensuring tile split and class weights for all regions...")
    run_tile_split(
        dataset_cfg=dataset_cfg,
        legend_cfg_path="configs/legend.yaml",
        force=False,
        regions=regions,
    )

    # ==================================================
    # STEP 2 — TRAIN PER-REGION
    # ==================================================
    for region in regions:
        print(f"\n[DATASET] Region: {region}")

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------
        seed = train_cfg["seed"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------
        device = get_device(train_cfg)
        print(f"Using device: {device}")

        # --------------------------------------------------
        # Dataset paths
        # --------------------------------------------------
        base_path = dataset_cfg["dataset"]["base_path"]
        region_root = os.path.join(base_path, region)

        root_s1 = os.path.join(region_root, "s1")
        root_gt = os.path.join(region_root, "ground_reference")
        root_labels = os.path.join(region_root, "labels")

        split_json = f'configs/splits/{region}/tile_split.json'

        # --------------------------------------------------
        # DATASETS
        # --------------------------------------------------
        train_dataset = S1Dataset(
            root_s1=root_s1,
            root_gt=root_gt,
            root_labels=root_labels,
            t_len=dataset_cfg["temporal"]["t_len"],
            legend=legend,
            out_size=tuple(dataset_cfg["spatial"]["out_size"]),
            polarizations=dataset_cfg["spatial"]["polarizations"],
            return_y_long=dataset_cfg["spatial"]["return_y_long"],
            patch_size=tuple(dataset_cfg["patching"]["patch_size"])
                if dataset_cfg["patching"]["enabled"] else None,
            stride=dataset_cfg["patching"]["stride"]
                if dataset_cfg["patching"]["enabled"] else None,
            tile_split_path=split_json,
            augmentation_cfg=dataset_cfg.get("augmentation"),
            split="train",
        )

        val_dataset = S1Dataset(
            root_s1=root_s1,
            root_gt=root_gt,
            root_labels=root_labels,
            t_len=dataset_cfg["temporal"]["t_len"],
            legend=legend,
            out_size=tuple(dataset_cfg["spatial"]["out_size"]),
            polarizations=dataset_cfg["spatial"]["polarizations"],
            return_y_long=dataset_cfg["spatial"]["return_y_long"],
            patch_size=tuple(dataset_cfg["patching"]["patch_size"])
                if dataset_cfg["patching"]["enabled"] else None,
            stride=dataset_cfg["patching"]["stride"]
                if dataset_cfg["patching"]["enabled"] else None,
            tile_split_path=split_json,
            augmentation_cfg=None,
            split="val",
        )

        print(
            f"[TILE SPLIT] Train tiles: {len(train_dataset)} | "
            f"Val tiles: {len(val_dataset)}"
        )

        # SAFETY CHECK (IMPORTANT)
        if len(train_dataset) == 0:
            raise RuntimeError(
                f"Empty training dataset for region '{region}'. "
                f"Check tile split and label JSONs."
            )
        
        X_train, y_train = build_patch_dataset_in_ram(train_dataset)
        X_val, y_val = build_patch_dataset_in_ram(val_dataset)

        print(f"Train patches in RAM: {X_train.shape[0]}")
        print(f"Val patches in RAM:   {X_val.shape[0]}")

        # --------------------------------------------------
        # DATALOADERS
        # --------------------------------------------------
        batch_size = dataset_cfg["dataloader"]["batch_size"]
        num_workers = dataset_cfg["dataloader"]["num_workers"]

        # --------------------------------------------------
        # WEIGHTED RANDOM SAMPLER — CC50 SHRUBLAND EXPERIMENT
        # --------------------------------------------------
        # A patch receives the boosted weight when it contains
        # at least one Shrubland pixel (class ID 5).
        shrubland_class_id = 5
        shrubland_boost = 10.0

        # y_train shape: (N, 1, H, W)
        y_train_long = y_train.long()

        has_shrubland = (
            (y_train_long == shrubland_class_id)
            .view(y_train_long.shape[0], -1)
            .any(dim=1)
        )

        sample_weights = torch.ones(
            y_train_long.shape[0],
            dtype=torch.float32,
        )

        sample_weights[has_shrubland] = shrubland_boost

        num_shrub_patches = int(has_shrubland.sum().item())
        num_total_patches = int(y_train_long.shape[0])

        original_shrub_fraction = (
            num_shrub_patches / num_total_patches
            if num_total_patches > 0
            else 0.0
        )

        expected_sampled_shrub_fraction = float(
            sample_weights[has_shrubland].sum()
            / sample_weights.sum()
        )

        print(
            f"[SAMPLER] Shrubland patches: "
            f"{num_shrub_patches}/{num_total_patches}"
        )
        print(
            f"[SAMPLER] Original Shrubland-patch fraction: "
            f"{original_shrub_fraction:.4f}"
        )
        print(
            f"[SAMPLER] Shrubland boost weight: "
            f"{shrubland_boost}"
        )
        print(
            f"[SAMPLER] Expected sampled Shrubland-patch "
            f"fraction: {expected_sampled_shrub_fraction:.4f}"
        )
        print(
            f"[SAMPLER] Samples per epoch: "
            f"{len(sample_weights)}"
        )
        print("[SAMPLER] Replacement: True")

        if num_shrub_patches == 0:
            raise RuntimeError(
                "No Shrubland-containing training patches were found"
            )

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        trainloader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        valloader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------
        from src.models.swin_unetr import build_swin_unetr
        in_channels = dataset_cfg["temporal"]["t_len"] * len(dataset_cfg["spatial"]["polarizations"])
        out_channels = legend.num_classes

        model = build_swin_unetr(
            img_size=dataset_cfg["patching"]["patch_size"],
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            spatial_dims=2,
            device=device,
        )

        print(f"\nModel instantiated via build_swin_unetr: {model.__class__.__name__}")

        # --------------------------------------------------
        # LOSS
        # --------------------------------------------------
        use_class_weights = train_cfg["loss"].get("use_class_weights", False)

        class_weights = None
        if use_class_weights:
            class_weights_path = os.path.join(
                dataset_cfg["derived"]["splits_base_dir"],
                region,
                "class_weights.json",
            )

            if not os.path.exists(class_weights_path):
                raise FileNotFoundError(
                    f"class_weights.json not found at {class_weights_path}"
                )

            import json
            with open(class_weights_path, "r") as f:
                cw = json.load(f)

            class_weights = torch.tensor(
                cw["weights"],
                dtype=torch.float32,
                device=device,
            )

            print("[LOSS] Using class-weighted CrossEntropyLoss")
        else:
            print("[LOSS] Using standard CrossEntropyLoss")

        
        ignore_class_ids = train_cfg["loss"]["ignore_class_ids"]
        primary_ignore_index = ignore_class_ids[0]
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=primary_ignore_index,
        )

        # --------------------------------------------------
        # OPTIMIZER
        # --------------------------------------------------
        optimizer = Adam(
            model.parameters(),
            lr=train_cfg["optimizer"]["learning_rate"],
            weight_decay=train_cfg["optimizer"]["weight_decay"],
        )

        # --------------------------------------------------
        # SCHEDULER (optional)
        # --------------------------------------------------
        scheduler = None
        sched_cfg = train_cfg["scheduler"]

        if sched_cfg["enabled"]:
            if sched_cfg["name"] == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=sched_cfg["t_max"],
                    eta_min=sched_cfg["eta_min"],
                )

            elif sched_cfg["name"] == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_cfg["step_size"],
                    gamma=sched_cfg["gamma"],
                )

            else:
                raise ValueError(
                    f"Unknown scheduler: {sched_cfg['name']}"
                )

        # --------------------------------------------------
        # CHECKPOINT DIRECTORY (region-based)
        # --------------------------------------------------
        checkpoint_dir = None
        if train_cfg["checkpoint"]["enabled"]:
            checkpoint_base = train_cfg["checkpoint"]["base_dir"]
            checkpoint_dir = os.path.join(
                checkpoint_base,
                region,
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            print(f"[CHECKPOINT] Saving best model to: {checkpoint_dir}")

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------
        train(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            cfg_train=train_cfg,
            checkpoint_dir=checkpoint_dir,
        )

        print(f"[DONE] Training completed for region: {region}")

if __name__ == "__main__":
    main()
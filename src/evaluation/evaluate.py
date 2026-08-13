# src/evaluation/evaluate.py

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from src.utils.config import load_config
from src.data.dataset import S1Dataset
from src.data.legend import load_legend
from src.data.patching import patch_tensors

from src.evaluation.metrics import compute_tile_metrics
from src.evaluation.io import (
    prepare_output_dirs,
    save_like_gt,
    save_probs_like_gt,
    save_qualitative_plot,
    save_metrics,
)

from src.training.utils.load_checkpoint import load_checkpoint

# --------------------------------------------------
# Patch → Tile reconstruction
# --------------------------------------------------
def reconstruct_from_patches(
    probs_patches: torch.Tensor,
    patch_coords: list,
    tile_shape: tuple,
    patch_size: tuple,
):
    C, H, W = tile_shape
    ph, pw = patch_size

    prob_sum = torch.zeros((C, H, W), device=probs_patches.device)
    prob_cnt = torch.zeros((1, H, W), device=probs_patches.device)

    for i, (y, x) in enumerate(patch_coords):
        prob_sum[:, y:y+ph, x:x+pw] += probs_patches[i]
        prob_cnt[:, y:y+ph, x:x+pw] += 1

    prob_cnt = torch.clamp(prob_cnt, min=1.0)
    return prob_sum / prob_cnt


# --------------------------------------------------
# MAIN
# --------------------------------------------------
@torch.inference_mode()
def run():

    cfg_data = load_config("configs/dataset.yaml")
    train_cfg = load_config("configs/training.yaml")
    legend = load_legend("configs/legend.yaml")

    regions = cfg_data["dataset"]["regions"]
    ignore_class_ids = train_cfg["loss"]["ignore_class_ids"]

    pols = cfg_data["spatial"]["polarizations"]
    if isinstance(pols, (list, tuple)):
        pol_suffix = "_".join(pols)
    else:
        pol_suffix = str(pols)

    patch_size = tuple(cfg_data["patching"]["patch_size"])
    stride = cfg_data["patching"]["stride"]

    for region in regions:
        print(f"\n[EVALUATE] Region: {region}")

        # -----------------------------
        # Dataset (VAL tiles only)
        # -----------------------------
        dataset = S1Dataset(
            root_s1=f"{cfg_data['dataset']['base_path']}/{region}/s1",
            root_gt=f"{cfg_data['dataset']['base_path']}/{region}/ground_reference",
            t_len=cfg_data["temporal"]["t_len"],
            legend=legend,
            out_size=tuple(cfg_data["spatial"]["out_size"]),
            polarizations=cfg_data["spatial"]["polarizations"],
            return_y_long=True,
            patch_size=None,
            stride=None,
            tile_split_path=f"{cfg_data['derived']['splits_base_dir']}/{region}/{cfg_data['derived']['tile_split_file']}",
            split="val",
            augmentation_cfg={"enabled": False},
        )

        if len(dataset) == 0:
            print(f"[WARNING] No validation tiles for {region}")
            continue

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # -----------------------------
        # Model
        # -----------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from src.models.swin_unetr import build_swin_unetr

        model = build_swin_unetr(
            img_size=tuple(cfg_data["spatial"]["out_size"]),
            in_channels = cfg_data["temporal"]["t_len"] * len(cfg_data["spatial"]["polarizations"]),
            out_channels=legend.num_classes,
            feature_size=48,
            spatial_dims=2,
            device=device,
        )

        model = load_checkpoint(
            model=model,
            cfg_data=cfg_data,
            train_cfg=train_cfg,
            region=region,
            device=device,
        )

        # -----------------------------
        # Output directories
        # -----------------------------
        out_dirs = prepare_output_dirs(
            base_dir=f"src/outputs/evaluation_{pol_suffix}",
            region=region,
        )

        metrics_all = []
        y_true_all, y_pred_all = [], []

        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.to(device)          # (1, T, H, W)
            y = y.squeeze(0).cpu().numpy()

            _, _, H, W = x.shape

            # --- extract patches ---
            x_patches = patch_tensors(x, patch_size, stride)

            coords = [
                (yy, xx)
                for yy in range(0, H - patch_size[0] + 1, stride)
                for xx in range(0, W - patch_size[1] + 1, stride)
            ]

            # --- patch inference ---
            logits_p = model(x_patches)
            probs_p = F.softmax(logits_p, dim=1)

            # --- reconstruct ---
            probs_tile = reconstruct_from_patches(
                probs_p,
                coords,
                (probs_p.shape[1], H, W),
                patch_size,
            )

            pred_tile = torch.argmax(probs_tile, dim=0).cpu().numpy()

            # --- metrics ---
            metrics = compute_tile_metrics(
                pred_tile,
                y,
                ignore_class_ids,
            )
            metrics_all.append(metrics)

            # accumulate for confusion matrix
            valid = ~np.isin(y, ignore_class_ids)
            y_true_all.append(y[valid])
            y_pred_all.append(pred_tile[valid])

            # --- qualitative + tif ---
            sar_mean = x.squeeze(0).mean(0).cpu().numpy()

            gt_path = dataset.pairs[idx][0]
            tile_id = os.path.splitext(os.path.basename(gt_path))[0]

            save_qualitative_plot(
                sar_mean,
                pred_tile,
                y,
                f"{out_dirs['plots']}/{tile_id}.png",
            )

            save_like_gt(
                pred_tile,
                gt_path,
                f"{out_dirs['tif']}/{tile_id}_pred.tif",
            )

            save_like_gt(
                y,
                gt_path,
                f"{out_dirs['tif']}/{tile_id}_gt.tif",
            )

            save_probs_like_gt(
                probs_tile.cpu().numpy(),
                gt_path,
                f"{out_dirs['tif']}/{tile_id}_probs.tif",
            )

        # -----------------------------
        # Aggregate metrics
        # -----------------------------
        metrics_mean = {
            k: float(np.mean([m[k] for m in metrics_all]))
            for k in metrics_all[0]
        }

        save_metrics(metrics_mean, f"{out_dirs['root']}/metrics.json")

        print(f"[DONE] {region}")
        print(metrics_mean)


if __name__ == "__main__":
    run()
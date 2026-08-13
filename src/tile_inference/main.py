# src/tile_inference/main.py
import os
import torch
import numpy as np
from typing import Optional

from src.tile_inference.config import load_inference_config
from src.tile_inference.window_reader import FeatureWindowReader
from src.tile_inference.tile_inference_engine import run_sliding_window_inference
from src.tile_inference.writer import (
    write_prediction_tile,
    write_posteriors_uint8_tif,
    write_posteriors_json,
)
from src.tile_inference.rescale_probabilities_to_uint8 import (
    rescale_probabilities_to_uint8
)

from src.data.legend import load_legend


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(cfg_override: Optional[dict] = None):

    # --------------------------------------------------
    # CONFIG & DEVICE
    # --------------------------------------------------
    if cfg_override is None:
        cfg = load_inference_config("src/tile_inference/inference.yaml")
    else:
        cfg = cfg_override
    device = get_device()
    print(f"[MAIN] Device: {device}")

    legend = load_legend("configs/legend.yaml")

    # --------------------------------------------------
    # OUTPUT PATH (from CLI via infer.py)
    # --------------------------------------------------
    out_dir = cfg.get("cli", {}).get("intermediate_dir")
    if out_dir is None:
        raise ValueError("Missing CLI intermediate_dir in config.")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # FEATURE DIRECTORY RESOLUTION
    # --------------------------------------------------
    # New pipeline passes `input_dir` (which contains `features/`).
    # Old pipeline passed `feature_dir` directly.
    # We support both for backward compatibility.

    cli_cfg = cfg.get("cli", {})

    feature_dir = cli_cfg.get("feature_dir")

    if feature_dir is None:
        input_dir = cli_cfg.get("input_dir")
        if input_dir is None:
            raise ValueError("Missing both 'feature_dir' and 'input_dir' in CLI config.")

        feature_dir = os.path.join(input_dir, "features")

    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    print(f"[MAIN] Using feature_dir: {feature_dir}")

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    from src.models.swin_unetr import build_swin_unetr

    model = build_swin_unetr(
        img_size=int(cfg["patching"]["patch_size"]),
        in_channels=int(cfg["n_channels"]),
        out_channels=legend.num_classes,
        feature_size=int(cfg["model"]["feature_size"]),
        spatial_dims=2,
        device=device,
    )

    pol = cfg["polarization_mode"]
    ckpt = cfg["model"].get("checkpoint")
    if ckpt is None:
        raise ValueError("Checkpoint must be provided via CLI.")

    print(f"[MAIN] Loading checkpoint ({pol}): {ckpt}")

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found for polarization {pol}: {ckpt}"
        )

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.to(device).eval()

    # --------------------------------------------------
    # FEATURE READER + INFERENCE
    # --------------------------------------------------
    with FeatureWindowReader(
        feature_dir=feature_dir,
        n_features=int(cfg["data"]["n_features"]),
    ) as reader:

        print(f"[MAIN] Tile size: {reader.H} x {reader.W}")
        print(
            f"[MAIN] Mode: {cfg['polarization_mode']} | "
            f"bands={cfg['bands']} | "
            f"channels={cfg['n_channels']}"
        )

        pred, prob = run_sliding_window_inference(
            reader=reader,
            model=model,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
        )

        # --------------------------------------------------
        # RESCALE POSTERIORS TO UINT8 (FINAL DELIVERABLE, IN-MEMORY)
        # --------------------------------------------------
        prob_u8 = rescale_probabilities_to_uint8(prob)

        # Free float probabilities to reduce memory footprint
        del prob

    # --------------------------------------------------
    # OUTPUT FILENAMES
    # --------------------------------------------------
    meta = cfg.get("cli", {})
    src = meta.get("source")
    tile = meta.get("tile")
    year = meta.get("year")
    area = meta.get("area")
    typ = meta.get("type")

    if not all([src, tile, year, area, typ]):
        raise ValueError("Missing CLI identifiers for output naming.")

    classified_name = f"{src}_{tile}_{year}_classified.tif"
    posteriors_name = f"{src}_{tile}_{year}_posteriors.tif"
    json_name = f"{src}_{tile}_{year}_posteriors.json"

    classified_path = os.path.join(out_dir, classified_name)
    posteriors_path = os.path.join(out_dir, posteriors_name)
    json_path = os.path.join(out_dir, json_name)

    # --------------------------------------------------
    # WRITE OUTPUTS
    # --------------------------------------------------
    write_prediction_tile(
        out_path=classified_path,
        pred=pred,
        reference_profile=reader.profile,
        feature_dir=feature_dir,
    )

    write_posteriors_uint8_tif(
        out_path=posteriors_path,
        prob_u8=prob_u8,
        reference_profile=reader.profile,
        feature_dir=feature_dir,
    )

    # --------------------------------------------------
    # POSTERIORS JSON
    # --------------------------------------------------
    # The posteriors contain one band per class in the legend,
    # regardless of their presence in the final argmax map
    all_classes = list(range(legend.num_classes))

    json_meta = {
        "year": year,
        "area": area,
        "tile": tile,
        "type": typ,
        "source": src,
        "file_name": posteriors_name,
        "SAR_class": all_classes,
        "channel_num": legend.num_classes,
    }

    write_posteriors_json(json_path, json_meta)

    print("[DONE] Tile inference completed")
    print(f"  → classified : {classified_path}")
    print(f"  → posteriors : {posteriors_path}")
    print(f"  → json       : {json_path}")

    return {
        "classified": classified_path,
        "posteriors": posteriors_path,
        "json": json_path,
        "out_dir": out_dir,
    }


if __name__ == "__main__":
    main()
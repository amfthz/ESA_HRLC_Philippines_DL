# src/tile_inference/tile_inference_engine.py
import os
import numpy as np
import torch
from tqdm import tqdm

from src.tile_inference.blending import hann2d
from src.utils.normalization import norm_s1_values
from src.data.preprocessing import nan_to_zero

def _starts(size: int, patch_size: int, stride: int):
    """
    Generate start indices so that the last patch touches the border.
    """
    if size <= patch_size:
        return [0]
    starts = list(range(0, size - patch_size + 1, stride))
    if starts[-1] != size - patch_size:
        starts.append(size - patch_size)
    return starts

def run_sliding_window_inference(
    reader,
    model,
    cfg: dict,
    device,
    out_dir: str,
):
    """
    Outputs:
      - pred_classes.tif will be written by main (after argmax)
    Returns:
      pred (H,W) uint8
      prob_final (K,H,W) float32 - the full probability cube
    """
    # --------------------------------------------------
    # Sliding window parameters
    # Priority:
    #   1) CLI overrides injected by infer.py
    #   2) YAML config defaults
    # --------------------------------------------------

    cli_cfg = cfg.get("cli", {})

    ps = int(cli_cfg.get("patch_size", cfg["patching"]["patch_size"]))
    st = int(cli_cfg.get("stride", cfg["patching"]["stride"]))

    # batch size can also be overridden from CLI
    batch_size = int(cli_cfg.get("batch_size", cfg["patching"].get("batch_size", 4)))

    K = int(cfg["model"]["num_classes"])
    band_ids = cfg["bands"]

    H, W = reader.H, reader.W
    r_starts = _starts(H, ps, st)
    c_starts = _starts(W, ps, st)

    # --- disk-backed accumulators to avoid RAM blow-up ---
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    prob_path = os.path.join(tmp_dir, "prob_accum.dat")
    wsum_path = os.path.join(tmp_dir, "weight_accum.dat")

    # float16 reduces disk+IO; you can change to float32 if you want max precision
    prob_accum = np.memmap(prob_path, mode="w+", dtype=np.float16, shape=(K, H, W))
    weight_accum = np.memmap(wsum_path, mode="w+", dtype=np.float32, shape=(H, W))
    prob_accum[:] = 0
    weight_accum[:] = 0

    w_patch = hann2d(ps).astype(np.float32)
    w_patch_b = w_patch[None, :, :]  # pre-broadcasted for faster multiplication

    model.eval()
    torch.set_grad_enabled(False)

    coords_batch = []
    x_batch = []

    total = len(r_starts) * len(c_starts)
    pbar = tqdm(total=total, desc="Sliding inference")

    def flush_batch():
        """Run the model on the current batch and accumulate results."""
        if not x_batch:
            return

        # allocate tensor directly on device to avoid large numpy stack
        B = len(x_batch)
        C = x_batch[0].shape[0]

        x = torch.empty((B, C, ps, ps), device=device, dtype=torch.float32)

        for i, patch in enumerate(x_batch):
            x[i] = torch.from_numpy(patch)

        with torch.no_grad():
            logits = model(x)  # (B,K,ps,ps)
            prob = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

        for (rr, cc), pb in zip(coords_batch, prob):
            r1, c1 = rr + ps, cc + ps

            # accumulate directly into global memmaps
            prob_accum[:, rr:r1, cc:c1] += (pb * w_patch_b).astype(np.float16)
            weight_accum[rr:r1, cc:c1] += w_patch

        coords_batch.clear()
        x_batch.clear()

    for r0 in r_starts:

        for c0 in c_starts:

            patch = reader.read_patch(r0, c0, ps, band_ids)
            if cfg["normalization"]["enabled"]:
                patch = norm_s1_values(patch)
            if cfg["normalization"].get("nan_to_zero", True):
                patch = nan_to_zero(patch)

            coords_batch.append((r0, c0))
            x_batch.append(patch)

            if len(x_batch) >= batch_size:
                flush_batch()

            pbar.update(1)

        # ---- flush remaining patches in this row ----
        flush_batch()

    pbar.close()

    # finalize probabilities and argmax
    w = np.maximum(weight_accum, 1e-6)[None, :, :]   # (1,H,W)
    prob_final = (prob_accum.astype(np.float32) / w) # (K,H,W)
    pred = np.argmax(prob_final, axis=0).astype(np.uint8)

    # cleanup memmaps (flush to disk)
    prob_accum.flush()
    weight_accum.flush()

    return pred, prob_final

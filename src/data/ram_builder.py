# src/data/ram_builder.py

import torch
from tqdm import tqdm
from typing import Tuple


def build_patch_dataset_in_ram(
    scene_dataset,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the full patch-level dataset in RAM.

    This function:
    - iterates over the scene-level dataset
    - extracts already-patched tensors (P, T, H, W) and (P, C, H, W)
    - concatenates everything into two tensors:
        X: (N_patches, T, H, W)
        Y: (N_patches, C, H, W) or (N_patches, H, W)

    Parameters
    ----------
    scene_dataset : torch.utils.data.Dataset
        Dataset returning (x_scene, y_scene), where:
        - x_scene shape: (P, T, ph, pw)
        - y_scene shape: (P, 1, ph, pw) or (P, ph, pw)

    verbose : bool
        Whether to print progress and final shapes.

    Returns
    -------
    X : torch.Tensor
        Patch-level input tensor in RAM.

    Y : torch.Tensor
        Patch-level ground-truth tensor in RAM.
    """

    X_all = []
    Y_all = []

    iterator = range(len(scene_dataset))
    if verbose:
        iterator = tqdm(iterator, desc="Building dataset", ncols=100)

    for idx in iterator:
        x_scene, y_scene = scene_dataset[idx]

        # Safety checks (fail-fast)
        if not torch.is_tensor(x_scene) or not torch.is_tensor(y_scene):
            raise TypeError("Dataset must return torch.Tensor objects")

        X_all.append(x_scene)
        Y_all.append(y_scene)

    # Concatenate all patches along patch dimension
    X = torch.cat(X_all, dim=0)
    Y = torch.cat(Y_all, dim=0)

    if verbose:
        print("\nRAM dataset built successfully")
        print(f"  X shape: {tuple(X.shape)}")
        print(f"  Y shape: {tuple(Y.shape)}")

        x_mem = X.element_size() * X.nelement() / 1e9
        y_mem = Y.element_size() * Y.nelement() / 1e9
        print(f"  Memory usage:")
        print(f"    X: {x_mem:.2f} GB")
        print(f"    Y: {y_mem:.2f} GB")
        print(f"    Total: {x_mem + y_mem:.2f} GB")

    return X, Y
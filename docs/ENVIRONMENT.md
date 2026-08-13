# Philippines Training Environment

## Reference environment

The Philippines CC50 baseline and WeightedRandomSampler experiments were run using the Conda environment:

`swinunet_philippines`

Verified runtime:

- Python: 3.8.20
- PyTorch: 1.12.1+cu113
- CUDA version used by the PyTorch build: 11.3
- MONAI: 1.2.0
- NumPy: 1.24.4
- Rasterio: 1.3.11
- CUDA available during verification: Yes

The Conda environment export is stored at:

`environment.yml`

## Original UNIPV environment

The environment file distributed with the original UNIPV Swin-UNETR package is preserved separately as:

`environment_original_unipv.yml`

That environment is substantially newer and includes, among other differences:

- Python 3.10
- PyTorch 2.5.x
- MONAI 1.4.x

It must therefore not be interpreted as the environment used for the Philippines CC50 training runs.

## MONAI compatibility

The Philippines version of `src/models/swin_unetr.py` contains compatibility handling for differences in the `SwinUNETR` constructor between MONAI versions.

The Philippines experiments were executed with MONAI 1.2.0.

The compatibility modification should be reviewed together with the original UNIPV implementation before defining the final stable reference version.

## Checkpoints

Large checkpoint binaries are kept outside the GitHub repository on laboratory storage.

Their exact paths, hashes, epochs and experiment status are documented in:

`docs/CHECKPOINT_INVENTORY.md`

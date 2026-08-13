# ESA HR Land Cover - Philippines DL Classification

This repository is the current handover version of the Swin-UNETR classification work used for the Philippines case study.

It was prepared from the original UNIPV package and the modifications made during the Philippines training work. The purpose of this version is to put the code, configurations, fixed data split, evaluation results and experiment notes in one place before the final reference version is reviewed together with Luigi.

At this stage, the CC50 clean training run is considered the current Philippines baseline. The WeightedRandomSampler run is kept separately as an experiment and should not be confused with the baseline.

## Current Philippines setup

The training dataset used by the current experiments is the CC50 version of the Philippines reference dataset.

Region:

`philippines`

Training/validation split:

- 249 training scenes
- 28 validation scenes

The split is fixed and stored in:

`configs/splits/philippines/tile_split.json`

SHA-256:

`f6988d7bc2604306c37e089aeed3a7682fe2e316be93f173c7246d8ceb6e0e4c`

The same split was used for the clean baseline and for the 10x WeightedRandomSampler experiment.

The Philippines class legend is:

| ID | Class |
|---|---|
| 0 | NoData |
| 1 | Built-up |
| 2 | Cropland |
| 3 | Forest |
| 4 | Grassland |
| 5 | Shrubland |
| 6 | Wetland |
| 7 | Water |
| 8 | Bareland |
| 9 | Mangrove Forest |

Class 0 is ignored during training and evaluation.

## Current baseline

The current baseline is the CC50 clean run using `src/train.py`.

No `WeightedRandomSampler` is used in this run.

Main validation results:

| Metric | Value |
|---|---:|
| Overall accuracy | 0.8333 |
| Mean IoU | 0.4339 |
| Macro F1 | 0.5460 |
| Shrubland recall | 0.0000 |

The best checkpoint was saved at epoch 53. Training stopped early at epoch 78.

The checkpoint itself is kept outside GitHub because of its size. Its path and SHA-256 are documented in:

`docs/CHECKPOINT_INVENTORY.md`

The full evaluation output is under:

`outputs/cc50_clean_baseline/`

## WeightedRandomSampler experiment

A second run was made to test whether oversampling patches containing Shrubland could improve the rare-class problem.

The training entry point is:

`src/train_cc50_sampler10x.py`

Patches containing at least one Shrubland pixel receive a sampling weight of 10.

This run is experimental and is not the current baseline.

Main validation results:

| Metric | Value |
|---|---:|
| Overall accuracy | 0.8038 |
| Mean IoU | 0.4038 |
| Macro F1 | 0.5122 |
| Shrubland recall | 0.0105 |

The sampler recovered 39 Shrubland true-positive pixels out of 3716 validation Shrubland pixels, but it also predicted 26056 pixels as Shrubland. The gain in Shrubland recall was therefore small and came with many false positives and lower overall performance.

Results are under:

`outputs/cc50_weighted_sampler_10x/`

A direct comparison with the clean baseline is under:

`outputs/cc50_comparison/`

## CC50 and Shrubland checks

The CC50 training dataset was created after the Shrubland issue found in the earlier Philippines runs.

The related utilities are:

`src/data/build_cc50_training_dataset.py`

and:

`src/data/check_cc50_shrubland_train_val_and_255.py`

The second script was used to check Shrubland presence in the fixed train/validation split and to verify the treatment of reference value 255.

The generated checks are stored in:

`outputs/cc50_pretraining_checks/`

These scripts still contain paths from the UNIPV laboratory workstation. They are kept here because they document the actual workflow used for the Philippines work. Path portability has not yet been refactored.

## Environment

The Philippines experiments were run with:

- Python 3.8.20
- PyTorch 1.12.1+cu113
- CUDA 11.3
- MONAI 1.2.0
- NumPy 1.24.4
- Rasterio 1.3.11

The exported environment is:

`environment.yml`

The environment shipped with the original UNIPV package is kept separately as:

`environment_original_unipv.yml`

More details are in:

`docs/ENVIRONMENT.md`

## Repository notes

The main files added or changed for the Philippines work are described in:

`docs/PHILIPPINES_CHANGES.md`

Machine-specific paths and the current portability limitations are described in:

`docs/PATHS_AND_PORTABILITY.md`

The checkpoint locations and hashes are in:

`docs/CHECKPOINT_INVENTORY.md`

A compact comparison of the current experiments is in:

`docs/EXPERIMENT_SUMMARY.csv`

The original training and inference notes are still available in:

`README_train_patch.md`

`README_infer_tile.md`

These documents come from the original package and have not yet been rewritten for the Philippines handover.

## About the current status

This repository should be read as the current Philippines working reference prepared for handover, not yet as the final production version.

The next review with Luigi should cover the remaining differences from the original UNIPV code, the machine-specific paths, the organization of the Philippines utilities and the checkpoint to use for the final inference workflow.

# Philippines Adaptation - Changes from Original UNIPV Package

## Purpose

This document summarizes the modifications made while adapting the original UNIPV Swin-UNETR land-cover classification package to the Philippines dataset.

The current repository is a handover version prepared for review. It distinguishes:

- generic/core compatibility changes,
- Philippines-specific configuration,
- CC50 reference preparation and rare-class analysis,
- the current clean baseline,
- experimental WeightedRandomSampler work.

The final stable reference version will be defined after review with Luigi.

---

## 1. Original UNIPV reference

The original package is preserved separately on the laboratory workstation at:

`/home/tlcrs/Philippines_Project/09_Luigi Russo/ESA_CCI_UNIPV_ORIGINAL`

Copies of the original main configuration files are also preserved in:

`configs/reference_original_unipv/`

The original environment definition is preserved as:

`environment_original_unipv.yml`

---

## 2. Generic/core code changes

### `src/train.py`

The Philippines reference training script uses:

`force=False`

when calling the tile-split generation routine.

This prevents an already established train/validation split from being regenerated unintentionally.

The authoritative Philippines split is stored in:

`configs/splits/philippines/tile_split.json`

Its SHA-256 is:

`f6988d7bc2604306c37e089aeed3a7682fe2e316be93f173c7246d8ceb6e0e4c`

The same split was used for the CC50 clean baseline and the 10x WeightedRandomSampler experiment.

### `src/training/trainer.py`

Epoch timing was added to the training loop.

This provides elapsed-time reporting during training and does not intentionally change the training algorithm.

### `src/eval.py`

Checkpoint loading was adapted to the Philippines training outputs.

Changes include support for:

- `.pt` checkpoint files,
- checkpoint dictionaries containing `model_state_dict`,
- saved epoch and monitored metric metadata,
- fallback loading of a plain model state dictionary.

### `src/models/swin_unetr.py`

Compatibility handling was added for differences in the MONAI `SwinUNETR` constructor.

The code first attempts construction with `img_size`. If that constructor signature is unsupported, it falls back to construction without `img_size`.

The Philippines experiments were run with MONAI 1.2.0.

`use_checkpoint=False` is used in the model constructor.

---

## 3. Philippines dataset configuration

The main configuration files under `configs/` correspond to the Philippines adaptation.

### Dataset

Current CC50 training dataset:

`/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES/training_cc50`

Region:

`philippines`

The current fixed split contains:

- 249 training scenes
- 28 validation scenes

### Philippines class legend

The Philippines legend uses contiguous class IDs:

- 0 - NoData
- 1 - Built-up
- 2 - Cropland
- 3 - Forest
- 4 - Grassland
- 5 - Shrubland
- 6 - Wetland
- 7 - Water
- 8 - Bareland
- 9 - Mangrove Forest

Class 0 is ignored during loss/metric computation.

Class 9 is a valid land-cover class in the Philippines configuration and is not ignored.

---

## 4. CC50 Shrubland reference preparation

The initial Philippines experiments revealed a severe Shrubland-class problem.

A conservative Shrubland reference rescue was therefore prepared.

The selected reference version is the CC50 version.

The CC50-specific dataset-building utility is:

`src/data/build_cc50_training_dataset.py`

This script is Philippines/reference-preparation specific and is not part of the generic data loader.

It currently contains laboratory-specific paths and should be treated as a reproducibility utility rather than a generic pipeline component.

---

## 5. Rare-class and pre-training analysis

The main CC50 Shrubland validation utility is:

`src/data/check_cc50_shrubland_train_val_and_255.py`

It checks the Shrubland distribution in the fixed training and validation sets and verifies handling of reference value 255.

Generated pre-training checks are stored in:

`outputs/cc50_pretraining_checks/`

These include scene-, tile-, and patch-level Shrubland distributions and loader/remapping checks.

The fixed validation set contains Shrubland in 5 validation scenes.

---

## 6. CC50 clean baseline

Training entry point:

`src/train.py`

The clean baseline does not use a WeightedRandomSampler.

Results are stored in:

`outputs/cc50_clean_baseline/`

Best checkpoint:

- epoch: 53
- early stopping: epoch 78
- best training val_miou: 0.3696774013914169

Unique-pixel validation evaluation:

- overall accuracy: 0.8332968375
- mean IoU: 0.4339485300
- macro F1: 0.5460316983

Shrubland:

- validation support: 3716 pixels
- predicted Shrubland pixels: 0
- recall: 0.0

The clean baseline is currently the Philippines baseline because it provides the stronger global validation performance.

It should not yet be interpreted as the final production reference checkpoint until review with Luigi is completed.

---

## 7. WeightedRandomSampler experiment

Experimental training entry point:

`src/train_cc50_sampler10x.py`

This experiment assigns a 10x sampling weight to training patches containing at least one Shrubland pixel.

Implementation:

`torch.utils.data.WeightedRandomSampler`

Main settings:

- Shrubland class ID: 5
- Shrubland-containing patch weight: 10.0
- replacement: True
- number of samples per epoch unchanged

Results are stored in:

`outputs/cc50_weighted_sampler_10x/`

Best checkpoint:

- epoch: 20
- early stopping: epoch 45
- best training val_miou: 0.3524741870332935

Unique-pixel validation evaluation:

- overall accuracy: 0.8038167051
- mean IoU: 0.4038107281
- macro F1: 0.5122222575

Shrubland:

- validation support: 3716 pixels
- predicted Shrubland pixels: 26056
- true positives: 39
- precision: 0.0014967762
- recall: 0.0104951561
- F1: 0.0026199113

The experiment achieved approximately 1.05% Shrubland recall but produced a large number of Shrubland false positives and reduced global performance.

For this reason it is retained as an experiment and is not the current baseline.

---

## 8. Evaluation utilities

Current CC50 evaluation scripts:

`src/evaluation/evaluate_cc50_best.py`

for the clean baseline, and:

`src/evaluation/evaluate_cc50_sampler10x.py`

for the 10x sampler experiment.

Evaluation results include:

- global metrics,
- per-class metrics,
- per-tile metrics,
- confusion matrices,
- Shrubland-specific validation statistics,
- checkpoint metadata and SHA-256 hashes.

Metrics in these evaluation outputs are computed after reconstructing unique validation-scene pixels from overlapping patches.

They are therefore not expected to be numerically identical to the validation metrics reported during training, which are averaged over validation patches.

---

## 9. Environment

The actual Philippines environment is documented in:

`docs/ENVIRONMENT.md`

and exported as:

`environment.yml`

The original UNIPV environment is preserved separately as:

`environment_original_unipv.yml`

---

## 10. Checkpoints

Large model checkpoints are not stored directly in GitHub.

The current baseline and experimental checkpoint locations, hashes and epochs are documented in:

`docs/CHECKPOINT_INVENTORY.md`

---

## 11. Experiment comparison

A compact machine-readable comparison is stored in:

`docs/EXPERIMENT_SUMMARY.csv`

Supporting comparison outputs are stored in:

`outputs/cc50_comparison/`

---

## 12. Items intentionally not included

The clean handover repository excludes:

- historical backup versions of training scripts,
- temporary debugging scripts,
- personal notes,
- email attachment duplicates,
- ZIP packages created for transfer,
- failed/duplicate launch logs,
- large checkpoint binaries.

The original working directories and safety backups remain preserved separately on the laboratory workstation.

---

## 13. Pending review

Before defining the final stable reference version, the following should be reviewed with Luigi:

1. generic compatibility changes versus the original UNIPV implementation,
2. portability of Philippines-specific paths,
3. organization of CC50/reference-preparation utilities,
4. final baseline checkpoint selection,
5. inference configuration and workflow.

No final production reference is declared by this handover version before that review.

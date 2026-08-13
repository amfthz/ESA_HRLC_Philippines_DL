# Paths and Portability

## Current status

This handover repository preserves the working Philippines CC50 implementation used on the UNIPV laboratory workstation.

Some configurations and Philippines-specific utilities therefore retain absolute laboratory paths.

These paths are intentionally preserved at this stage so that the handover version remains directly traceable to the verified training and evaluation runs.

They should be reviewed with Luigi before the final stable/reference version is defined.

## Active configuration paths

`configs/dataset.yaml`

currently points to:

`/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES/training_cc50`

`configs/training.yaml`

currently uses the laboratory checkpoint base directory:

`/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES/checkpoints`

These values are machine-specific.

## Philippines-specific utilities

The following scripts currently contain laboratory-specific paths:

- `src/data/build_cc50_training_dataset.py`
- `src/data/check_cc50_shrubland_train_val_and_255.py`
- `src/evaluation/evaluate_cc50_best.py`
- `src/evaluation/evaluate_cc50_sampler10x.py`

The CC50 dataset builder and Shrubland audit also refer to the MOLCA rare-class working directory on `/media/tlcrs/Disc_Data/`.

These scripts are retained as reproducibility/reference utilities rather than generic cross-region pipeline components.

## Historical outputs

Absolute paths present inside:

- training logs,
- evaluation logs,
- evaluation summaries,
- experiment configuration snapshots,

must not be interpreted as current portable defaults.

They record the exact filesystem locations used when the experiments were executed and are preserved as provenance.

## Tile split behavior

The main Philippines training entry point, `src/train.py`, calls the tile-split routine with:

`force=False`

so the established split is not regenerated during normal reference training.

The generic utility:

`src/data/run_tile_split.py`

contains `force=True` only in its standalone `__main__` execution block.

Running that utility directly is therefore intended to regenerate split/statistics products.

The authoritative Philippines split is:

`configs/splits/philippines/tile_split.json`

SHA-256:

`f6988d7bc2604306c37e089aeed3a7682fe2e316be93f173c7246d8ceb6e0e4c`

This same split was used by both the CC50 clean baseline and the 10x WeightedRandomSampler experiment.

## Pending portability review

For the final stable repository, possible improvements to review with Luigi include:

- command-line arguments for CC50 utility input/output paths,
- external configuration for checkpoint paths,
- removal of machine-specific defaults where appropriate,
- clearer separation between generic cross-region code and Philippines/CC50 utilities.

No such functional refactoring has been applied in this handover version before joint review.

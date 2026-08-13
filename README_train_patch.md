# ESA CCI UNIPV – Land Cover Mapping (SAR DL)

This repository contains the full pipeline for **training and inference** of a deep learning land cover classifier based on **Sentinel-1 SAR features**, developed in the context of the **ESA CCI HRLC Phase 2** project.

## Core Components

- Dataset indexing and loading  
- Deep learning training  
- Tile-based inference  
- Post-processing and merging (e.g. water contribution)

---

## 1. Training Dataset Structure

The training dataset must follow a **strict directory and naming convention**.

Each training sample is composed of:

- **1 Ground Truth (GT) raster**
- **28 SAR feature rasters** (temporal stack)

➡️ There is **exactly one GT for every group of 28 SAR features**.

---

### 1.1 Root Structure

```text
dataset/
└── <region>/
    ├── s1/
    │   ├── <tile_id>_<date>_<stat>_<row>_<col>.tif
    │   └── ...
    ├── ground_reference/
    │   ├── <tile_id>_GT_<row>_<col>.tif
    │   └── ...
    └── labels/
        ├── <tile_id>_<row>_<col>.json
        └── ...
```

Where:

- `<region>` ∈ `{amazonia, africa, siberia}`
- `<tile_id>` is the Sentinel-2 tile ID (e.g. `20KQD`)
- `<row>_<col>` identify the spatial patch inside the tile
- `<stat>` ∈ `{SI, LEE, MAX, MIN, MAXMIN, MEAN, MEDIAN}`
- Dates define the **temporal sequence**

---

### JSON-based Feature Association (Mandatory)

Each GT tile must have a corresponding JSON file in the `labels/` directory.

The JSON filename uniquely identifies the GT (e.g. `20KPF_12_02.json` corresponds to `20KPF_GT_12_02.tif`).

This JSON file contains the ordered list of the 28 SAR features under the key `corresponding_s1`.

The order specified in the JSON defines the exact temporal and feature stack order used during training.

Example JSON content:

```json
{
  "corresponding_s1": "20KPF_2021_0101_SI_12_02.tif;20KPF_2021_0102_LEE_12_02.tif;...;20KPF_2021_0407_MEDIAN_12_02.tif",
  "labels": "0;20"
}
```

**Note:** Filename-based sorting is no longer used. The JSON file is the single source of truth for feature association and ordering.

---

## 2. Example: One Training Sample

Below is a **real example** of how a single training sample is organized.

### Ground Truth

```text
20KQD_GT_17_17.tif
```

### Corresponding 28 SAR Features

```text
t=0:  20KQD_20210101_SI_17_17.tif
t=1:  20KQD_20210102_LEE_17_17.tif
t=2:  20KQD_20210103_MAX_17_17.tif
t=3:  20KQD_20210104_MIN_17_17.tif
t=4:  20KQD_20210105_MAXMIN_17_17.tif
t=5:  20KQD_20210106_MEAN_17_17.tif
t=6:  20KQD_20210107_MEDIAN_17_17.tif

t=7:  20KQD_20210201_SI_17_17.tif
t=8:  20KQD_20210202_LEE_17_17.tif
t=9:  20KQD_20210203_MAX_17_17.tif
t=10: 20KQD_20210204_MIN_17_17.tif
t=11: 20KQD_20210205_MAXMIN_17_17.tif
t=12: 20KQD_20210206_MEAN_17_17.tif
t=13: 20KQD_20210207_MEDIAN_17_17.tif

t=14: 20KQD_20210301_SI_17_17.tif
t=15: 20KQD_20210302_LEE_17_17.tif
t=16: 20KQD_20210303_MAX_17_17.tif
t=17: 20KQD_20210304_MIN_17_17.tif
t=18: 20KQD_20210305_MAXMIN_17_17.tif
t=19: 20KQD_20210306_MEAN_17_17.tif
t=20: 20KQD_20210307_MEDIAN_17_17.tif

t=21: 20KQD_20210401_SI_17_17.tif
t=22: 20KQD_20210402_LEE_17_17.tif
t=23: 20KQD_20210403_MAX_17_17.tif
t=24: 20KQD_20210404_MIN_17_17.tif
t=25: 20KQD_20210405_MAXMIN_17_17.tif
t=26: 20KQD_20210406_MEAN_17_17.tif
t=27: 20KQD_20210407_MEDIAN_17_17.tif
```

⚠️ **The ordering of the 28 features is fixed and mandatory.**

---

## 3. Dataset Indexing

Before training, the dataset must be indexed to verify consistency:

```bash
python -m src.data.indexing
```

This script:

- Iterates over the JSON files in the `labels/` directory  
- For each JSON, verifies the existence of the corresponding GT raster  
- Checks that exactly 28 SAR features exist and are readable in the order specified by the JSON  
- Skips tiles with missing or inconsistent associations  
- Prints a summary like:

```text
Totale tile: 96
[0] GT: 20KQD_GT_17_17.tif
    t=0: ...
```

❗ If indexing fails, **training must not be launched**.

---

This design guarantees robust, reproducible, and merge-safe dataset construction across regions.

---

## 4. Configuration Files

All default configurations are stored in `configs/`.

Main files:

- `training.yaml` → model, optimizer, epochs, batch size
- `dataset.yaml` → dataset structure and defaults
- `legend.yaml` → land cover classes and codes

If no CLI parameters are provided, **defaults from these YAML files are used**.

---

## 5. Training Script (`train.py`)

The training script now supports **CLI overrides** for the most important runtime parameters, allowing portable execution across laptop, workstation and HPC environments.

### 5.1 Supported CLI Arguments

The following parameters can be overridden from the command line:

```text
--base-path        → overrides dataset.base_path
--regions          → overrides dataset.regions
--checkpoint-dir   → overrides checkpoint.base_dir
```

If not provided, values from the YAML configuration files are used.

Relevant YAML defaults:

```yaml
dataset:
  base_path: <configs/dataset.yaml>
  regions:   <configs/dataset.yaml>

checkpoint:
  base_dir:  <configs/training.yaml>
```

CLI parameters **always take precedence** over YAML.

---

### 5.2 Example Training Commands

#### 5.2.1 Training on a single region

```bash
python -m src.train \
  --base-path ESA_CCI_UNIPV/DATASET_LC_MAPPING_JSTARS/training \
  --regions siberia \
  --checkpoint-dir ESA_CCI_PROJECT/training_checkpoints
```

Behaviour:
- Only the *siberia* region is processed
- Tile statistics, splits and class weights are computed (or reused) only for that region
- Checkpoints are saved in the provided checkpoint directory

---

#### 5.2.2 Training on multiple regions sequentially

```bash
python -m src.train \
  --base-path ESA_CCI_UNIPV/DATASET_LC_MAPPING_JSTARS/training \
  --regions amazonia africa siberia \
  --checkpoint-dir ESA_CCI_PROJECT/training_checkpoints
```

Behaviour:
- Dataset is expected under:
  `ESA_CCI_UNIPV/DATASET_LC_MAPPING_JSTARS/training/<region>/`
- Training runs **sequentially per region**
- Each region has independent:
  - tile statistics (tile_stats.json)
  - train/validation split (tile_split.json)
  - class weights (class_weights.json)
  - checkpoints and logs

If CLI parameters are not provided:
- `dataset.base_path` is read from `configs/dataset.yaml`
- `dataset.regions` is read from `configs/dataset.yaml`
- `checkpoint.base_dir` is read from `configs/training.yaml`

---

## 5.3 Tile-wise Train / Validation Split (Automatic)

Training and validation splitting is performed **at tile level**, not at patch level, to avoid spatial leakage and ensure statistically meaningful validation.

### Canonical Tile Definition (JSON-driven)

Tiles are defined **canonically** by the JSON files in the `labels/` directory:

```text
labels/
└── <tile_id>_<row>_<col>.json
```

Each JSON corresponds to exactly one GT tile:

```text
ground_reference/
└── <tile_id>_GT_<row>_<col>.tif
```

The JSON filenames are the **single source of truth** for:
- which tiles exist
- which tiles are eligible for training/validation
- how SAR features are associated to each GT

Tiles missing GTs or with invalid associations are **automatically ignored**.

---

### Tile-level Class Statistics

Before splitting, the pipeline computes **tile-level class distributions** from the GT rasters:

- For each tile, a normalized class-frequency vector is computed
- Only valid (non-NoData) pixels are considered
- Tiles with no valid pixels are skipped

The result is saved as:

```text
configs/splits/<region>/tile_stats.json
```

---

### Clustering-based Split Strategy

Tiles are grouped using **KMeans clustering** on their class-distribution vectors.

This ensures that:
- tiles with similar land-cover composition are grouped together
- train and validation sets are **stratified by semantic content**, not randomly

After clustering:
- singleton clusters are merged automatically
- a **stratified train/validation split** is applied at tile level

The resulting split is saved as:

```text
configs/splits/<region>/tile_split.json
```

Example content:

```json
{
  "n_tiles": 81,
  "n_clusters": 6,
  "val_fraction": 0.2,
  "train_tiles": ["20KPF_12_02", "..."],
  "val_tiles": ["20KQD_17_17", "..."]
}
```

---

### Automatic Class Weights Computation (Optional)

If specified in `training.yaml`, the pipeline automatically computes **class weights** based on the **training tiles only**.

The method used is:

- **inverse square-root frequency**
- computed from tile-level class distributions
- normalized over active classes

Ignored classes (e.g. background or NoData) are excluded via:

```yaml
loss:
  ignore_index: 0
  ignore_class_ids: [0]
```

The resulting weights are saved as:

```text
configs/splits/<region>/class_weights.json
```

These weights can be directly injected into the loss function during training.

---

### Execution and Reproducibility

All steps above are executed **automatically** when launching:

```bash
python -m src.train
```

By default:
- existing statistics, splits, and weights are reused
- results are fully reproducible across runs

Setting:

```python
force = True
```

forces **recomputation and overwrite** of:
- tile statistics
- train/validation split
- class weights

This is useful when:
- GT data changes
- label definitions are updated
- experimenting with different clustering parameters

---


---

## Training Patch Size (Important)

The training pipeline is **optimized for input patches of size 549 × 549 pixels**.

This choice follows the experimental setup described in:

Russo, L., Sorriso, A., Ullo, S. L., & Gamba, P. (2025). *A Deep Learning Architecture for Land Cover Mapping Using Spatio‑Temporal Sentinel‑1 Features*. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 18, 10562–10581. https://doi.org/10.1109/JSTARS.2025.3557687

The patch size is not arbitrary: it is designed to provide a **large spatial context** around each training pixel while remaining compatible with GPU memory constraints and the Swin‑UNETR receptive field.

Key motivations:

- Ensures sufficient spatial context for complex SAR backscatter patterns
- Matches the receptive field used in the JSTARS paper experiments
- Improves stability of class statistics inside each training sample
- Provides better representation of rare classes (e.g., built‑up and wetlands)

---

## 6. Key Assumptions (IMPORTANT)

- All rasters must be **aligned**  
  (same CRS, resolution, size — preferably UTM, 10 m)
- Features and GT must be **pixel-aligned**
- **One GT ↔ exactly 28 SAR features**
- Naming convention is **not optional**

---

## 7. Common Pitfalls

❌ Missing one feature → indexing fails  
❌ Different CRS between GT and features → wrong training  
❌ Wrong temporal ordering → invalid learning  
❌ Mixing regions in the same folder → undefined behavior  

---

## 8. Recommended Workflow

1. Prepare dataset following the exact structure  
2. Run dataset indexing  
3. Inspect indexing output  
4. Configure `training.yaml` if needed  
5. Launch `train.py`  
6. Monitor logs and checkpoints  
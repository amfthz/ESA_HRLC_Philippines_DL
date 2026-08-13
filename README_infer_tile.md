# ESA CCI UNIPV – SAR DL Tile Inference & Water Merging

This repository contains the **tile-based inference pipeline** for land‑cover mapping using **Sentinel‑1 SAR deep learning**, developed within the **ESA CCI HRLC Phase 2** project.

The pipeline enables **large‑scale, reproducible inference** and generation of
final deliverables compliant with ESA CCI production workflows.

---

## Key Capabilities

The framework supports:

- Sentinel‑1 SAR inference with:
  - VH polarization
- Sliding‑window inference on large tiles
- Generation of:
  - **Classified land‑cover maps**
  - **Per‑class posterior probability maps**
- Optional integration with external **water products**
  - water merging **with seasonality**
  - water merging **without seasonality**

---

## Pipeline Overview (Based on src/infer.py)

The entire pipeline is orchestrated by a **single entrypoint**:

```
python -m src.infer
```

This script is the **core controller of the full workflow**. It does not only run inference, but manages the entire processing chain automatically.

### What `src/infer.py` actually does

For each tile, the pipeline performs the following steps:

1. **Input normalization (`ensure_input_structure`)**
   - Accepts flexible input paths
   - Reorganizes data into:
     ```
     <tile>/S1/features/
     <tile>/S1/water_map/
     ```
   - Moves files automatically based on naming rules:
     - SAR features → start with tile ID
     - Water maps → contain "water"

2. **Input completion (`prepare_input_directory`)**
   - Fills missing SAR features (up to 28 expected)
   - Generates a water map if missing

3. **Deep Learning inference**
   - Executed via `src.tile_inference.main`
   - Uses sliding window:
     - patch size
     - stride
     - batch size
   - Produces:
     - classified map
     - posterior probabilities
     - JSON metadata

4. **Automatic water detection**
   - Reads the raster
   - Detects if values contain class `2`
   - Determines mode:
     - `seasonal`
     - `no_seasonal`

5. **Automatic merge_config generation**
   - No manual JSON required
   - Created dynamically from the detected water map

6. **Posterior merging**
   - Calls:
     ```
     src.tile_merging.merge_posteriors_DL_final
     ```
   - Injects water information into:
     - classification
     - posteriors

7. **Post-processing (UNIGE remapping)**
   - Renames outputs with `_DL`
   - Applies class remapping
   - Updates JSON using classified raster

8. **Cleanup**
   - Deletes `_intermediate` directory automatically

---

### Conceptual flow

```
Input tile (features + optional water)
        ↓
[ infer.py ]
        ↓
Structure fix + feature completion
        ↓
DL inference (sliding window)
        ↓
Water detection + auto-config
        ↓
Posterior merging
        ↓
Remapping + final outputs
```

---

### Key design principle

👉 **All logic is centralized inside `infer.py`**

This means:

- no manual preprocessing
- no manual merge_config
- no manual pipeline orchestration

Everything is automatic and reproducible.

---

## DL Inference Outputs

After running the deep learning model on each tile, the following outputs are generated:

- **Classified raster** → GeoTIFF with predicted class labels
- **Posterior raster** → GeoTIFF with per‑class probabilities
- **JSON metadata** → semantic description of posterior bands

Example output structure:

```
ESA_CCI_PROJECT/output/
└── static/2021/Amazon/22MCT/S1/
    ├── S1_22MCT_2021_classified.tif
    ├── S1_22MCT_2021_posteriors.tif
    └── S1_22MCT_2021_posteriors.json
```

Posterior encoding convention:

- `0` → NoData
- `1–255` → probability mapped internally to `[0,1]`

---

## Water Merging Logic

The pipeline supports merging the DL prediction with an **external water map**.

> **Water always wins over other land‑cover classes.**

This affects:
- classified raster
- posterior probabilities
- JSON metadata

The behaviour is controlled by a **tile‑specific JSON configuration file**.

---

## Merge Configuration JSON (Critical)

Each tile must provide its own JSON file for merging.

The most important field is:

```
water.image_path
```

This field must contain the **absolute path** to the water raster for that tile.

### Expected JSON structure

```json
{
  "water": {
    "image_path": "/absolute/path/to/water_raster.tif",
    "has_seasonal": true,
    "remove_permanent": true|false,
    "seasonal": {
      "1": <prob_when_water>,
      "0": <prob_when_not_water>,
      "position": <band_index>,
      "cls_value": <class_code>
    },
    "permanent": {
      "1": <prob_when_water>,
      "0": <prob_when_not_water>,
      "position": <band_index>,
      "cls_value": <class_code>
    }
  }
}
```


Seasonality and non‑seasonality configs share the same overall structure, but they differ in **two key flags that control the merging behaviour**:

- `has_seasonal` → indicates that the water raster contains seasonal information.
- `remove_permanent` → activates the *no‑seasonality* workflow when set to `true`.

### Seasonality merge_config (seasonal + permanent water)
Typical configuration example:

```json
{
  "water": {
    "image_path": "/absolute/path/to/<TILE>_seasonal_water_map.tif",
    "has_seasonal": true,
    "seasonal": { ... },
    "permanent": { ... }
  }
}
```

Characteristics:
- Uses **seasonal + permanent** water information
- `remove_permanent` is **absent or false**
- Two water contributions are injected into the posteriors

### No‑seasonality merge_config (single water layer)
Typical configuration example:

```json
{
  "water": {
    "image_path": "/absolute/path/to/<TILE>_water_map.tif",
    "has_seasonal": true,
    "remove_permanent": true,
    "seasonal": { ... },
    "permanent": { ... }
  }
}
```

Characteristics:
- Activates the **no‑seasonality workflow**
- Permanent water is removed from the pipeline
- Only a single water contribution is injected into the posteriors

⚠️ Important: each tile must provide its own merge JSON and the field `water.image_path` must point to the **exact raster of that tile**.

---

## Running the Inference Pipeline

The entire pipeline is executed through the CLI entry point:

```
python -m src.infer
```

This script performs **end-to-end processing for a single tile**, including:

1. Input validation and automatic restructuring
2. Filling missing SAR features (up to 28)
3. Water map handling (creation if missing)
4. Automatic detection of water type (seasonal vs non-seasonal)
5. Automatic generation of `merge_config` JSON
6. Deep Learning inference (sliding window)
7. Posterior merging
8. UNIGE remapping + JSON update

---

### Run inference on a single tile

Minimal example:

```bash
python -m src.infer \
  -c configs/inference.yaml \
  --input-dir ESA_CCI_PROJECT/tiles_io/input/static/2021/Amazon/22MCT/S1 \
  --polarization VH \
  --checkpoint ESA_CCI_PROJECT/training_checkpoints/amazonia/best_model.pt \
  --output-base-dir ESA_CCI_PROJECT/tiles_io/output \
  --type static \
  --year 2021 \
  --area Amazon \
  --tile 22MCT \
  --source S1 \
  --patch-size 256 \
  --stride 128 \
  --batch-size 4
```

### Required arguments

- `--input-dir` → path to tile folder (can be flexible, pipeline auto-fixes structure)
- `--checkpoint` → trained DL model
- `--output-base-dir` → base output directory
- `--type`, `--year`, `--area`, `--tile`, `--source` → define output structure

### Optional overrides

These parameters override YAML config at runtime:

- `--polarization` → VH | VV | VH_VV
- `--patch-size` → sliding window size
- `--stride` → sliding window stride
- `--batch-size` → inference batch size

---

### Input directory flexibility (important)

The pipeline is robust to different input layouts. You can pass:

- `<TILE>/S1/`
- `<TILE>/`
- `<TILE>/features/`
- `<TILE>/water_map/`

The function `ensure_input_structure()` will automatically:

- create `S1/features/` and `S1/water_map/`
- move SAR features (matching `<TILE>_*`)
- move water maps (containing "water")

---

### Output structure

Outputs are always written as:

```
<output-base-dir>/<type>/<year>/<area>/<tile>/<source>/
```

Inside you will find:

- `_intermediate/` → temporary DL outputs (auto-deleted)
- `seasonality/` or `no_seasonality/` → final outputs

Final products:

- `*_classified_DL.tif`
- `*_posteriors_DL.tif`
- `*_posteriors_DL.json`

---

## Running inference on multiple tiles

For batch processing, use the provided script:

```
scripts/run_inference_2019_amazon.sh
```

### Make executable (first time only)

```bash
chmod +x scripts/run_inference_2019_amazon.sh
```

### Run

```bash
bash scripts/run_inference_2019_amazon.sh
```

---

### How the script works

The script:

1. Activates the correct conda environment (e.g. `torch` or `torch-gpu`)
2. Defines a list of tiles:

```bash
TILES=(
  "18MYS" "19LEL" ... "24MVV"
)
```

3. Loops over tiles and runs:

```bash
python -m src.infer ...
```

Each tile is processed independently.

If one tile fails:

```bash
Tile <TILE> failed. Continuing...
```

→ the script continues with the next tile.

---

### Key advantage

The script is **minimal** because all intelligence is inside `src/infer.py`:

- no manual merge_config needed
- no manual water handling
- no manual feature completion

Everything is handled automatically by the pipeline.

---

## Key Assumptions

All SAR features and water maps must be:

- UTM projected
- perfectly pixel‑aligned

---

## Recommended Workflow (Practical Usage)

### Single tile inference (manual run)

Use `src/infer.py` directly when you want full control:

```bash
python -m src.infer \
  -c configs/inference.yaml \
  --input-dir <PATH_TO_TILE>/S1 \
  --checkpoint <MODEL_PATH> \
  --output-base-dir <OUTPUT_BASE> \
  --type static \
  --year <YEAR> \
  --area <AREA> \
  --tile <TILE_ID> \
  --source S1
```

Use this when:
- debugging a tile
- testing parameters
- running experiments

---

### Multi-tile inference (automated)

Use the script:

```
scripts/run_inference_2019_amazon.sh
```

This script is a **thin wrapper over `infer.py`**.

#### What it does

1. Activates conda environment (`torch` / `torch-gpu`)
2. Defines a list of tiles
3. Runs:

```bash
python -m src.infer ...
```

for each tile

---

#### Why this design

- `infer.py` = **core logic (single tile)**
- `run_inference.sh` = **automation layer (multiple tiles)**

This separation ensures:

- modularity
- reproducibility
- easy debugging

---

### Typical production workflow

1. Prepare input tiles (features may be incomplete)
2. Launch batch script:

```bash
bash scripts/run_inference_2019_amazon.sh
```

3. Pipeline automatically:
   - fixes inputs
   - runs DL
   - merges water
   - produces final outputs

4. Inspect outputs in:

```
<output-base-dir>/<type>/<year>/<area>/<tile>/<source>/
```

---

### Important takeaway

👉 You NEVER need to:

- manually create merge_config
- manually align features
- manually manage pipeline steps

👉 You ONLY need to:

- provide input directory
- provide model checkpoint
- run `infer.py` (directly or via script)

---
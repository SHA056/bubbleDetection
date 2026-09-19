# bubbleDetection

A pipeline that turns published superbubble catalogues and a JWST/MIRI mosaic into a trained segmentation model and a per-size-class detection report.

Given a FITS mosaic and one or more catalogues of elliptical annotations, the code merges and cleans the catalogues, rasterises them into a three-valued segmentation target, extracts training patches under a leakage-safe spatial split, trains a U-Net per fold, and grades the result against a spatial null model. A command-line tool applies a trained model to an unseen mosaic and emits a candidate catalogue in sky coordinates.

## Requirements

Python 3.10 or newer. No environment file is committed; install directly:

```
pip install tensorflow numpy pandas scipy scikit-image astropy reproject \
            matplotlib jupyter nbformat nbclient
```

## Inputs

Two things must be present before anything runs:

- `data/raw/fits/` — a Level-3 calibrated (i2d) FITS mosaic
- `data/raw/catalogue/` — one or more catalogues of elliptical annotations, as
  whitespace-delimited text with sky coordinates, semi-major and semi-minor
  axes in parsecs, average radius and position angle

Neither is redistributed in this repository. The pipeline as configured expects the F770W mosaic of NGC 628 and the three annotator catalogues associated with it.

## Running the pipeline

Seven notebooks in `notebooks/`, in this order. The first four are run once and produce the cleaned catalogue and the masks; the last three are run once per label set.

| # | Notebook | What it does |
|---|---|---|
| 1 | `catalogueBCorrection` | Repairs an hour/degree notation defect in one catalogue |
| 2 | `bubbleStrattification` | Merges the catalogues, assigns a global ID, derives six size classes from radius percentiles |
| 3 | `catalogueSegregation` | Applies a manual exclusion list, writes the cleaned catalogue |
| 4 | `galaxyMaskGeneration` | Rasterises annotations into a foreground mask and an ignore mask |
| 5 | `modelTraining` | Patch extraction, class balancing, augmentation, U-Net training per fold, threshold sweep |
| 6 | `perClassEvaluation` | Post-processing, per-class detection, precision, chance-coverage control |
| 7 | `baselineTraining` | Reduced-capacity model for comparison |

**Selecting a label set.** Notebooks 4–7 read an `ACTIVE_STAGE` constant in their first cell. A label set is a filter on the `SIZE_CLASS` column of a single catalogue, not a separate file, so switching stages requires no data duplication:

| stage | active classes |
|---|---|
| `stage00` | largest class only |
| `stage01` | two largest |
| `stage02` | three largest |
| `stage05` | all six |

Annotations outside the active set are marked *ignored* rather than background, and are excluded from the loss.

**Fold definitions** come from `scripts/makeFolds.py`, which partitions the mosaic into contiguous blocks. Patches straddling a block boundary are dropped from both the training and validation sets, so no training patch shares pixels
with a validation patch.

## Applying a trained model

```
python scripts/whereismybubble.py \
    --image path/to/mosaic.fits \
    --model outputs/models/stage02fold3bubble_unet.keras 
    --output results/name_of_galaxy/reduced
```
Writes a probability map, a candidate region table in pixel and sky
coordinates, and a DS9 region file. `--threshold`, `--min-blob-px` and
`--stride` override the defaults. The scipt assumes that the distance in pc is set at 9.77e6.

## Layout

```
data/
  raw/                 FITS mosaic and source catalogues (not redistributed)
  processed/           merged and cleaned catalogue, masks, patches, arrays
notebooks/             the seven pipeline notebooks
  archival/            superseded notebooks from earlier pipeline versions
scripts/
  makeFolds.py         partitions the mosaic into spatial folds
  mergeCatalogues.py   catalogue merge helper
  whereismybubble.py   probabily map writer
outputs/
  models/              trained weights, prefixed by label set and fold
  history/             training curves, run configurations, selected thresholds
  predictions/         per-bubble coverage, per-class evaluation, chance control
  figures/             images and publication plots
```
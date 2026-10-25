# Pork rasher packaging inspection

MMA3001 (Numerical Methods and Machine Learning) individual project.

A computer-vision pipeline that looks at a camera frame of a packaged pork-rasher tray on a conveyor and finds packaging and product-placement defects. The aim is to support automated quality inspection, so that faulty trays can be pulled before distribution without relying on manual inspection.

> **Status: v0.1, data audit and preparation.** Training, the baseline comparison and evaluation are still to come.

## Problem, inputs and outputs

| | |
|---|---|
| **Input** | One RGB image, 720 × 540 px, 8-bit (JPEG), of a tray on a dark mesh conveyor belt, taken from above. Trays are often only partly in view. |
| **Output** | A list of detected defects. Each has a class, a bounding box (normalised `x, y, w, h` in [0, 1]) and a confidence in [0, 1]. Also a per-frame flag, *defect present*, which is true if any detection is above a confidence threshold. |
| **Defect classes** | `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`. (`wrinkle` is dropped; see below.) |
| **Out of scope** | Tray-level pass/fail. A single frame often shows only part of a tray, so "no defect visible" does not mean "tray OK". |

## Data

[Pork Rasher Error (Packaging), v4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4), by Roboflow user "Hello", licence **CC BY 4.0**. The data is **not** stored in this repository.

To get it: download **v4** in **YOLOv8** format from the link above and place the zip (or its unzipped folder) at `data/raw/`.

### Audit results (`results/data_audit.json`)

| | Train | Valid | Test |
|---|---|---|---|
| Files | 3,349 | 120 | 80 |
| Distinct images | 1,117 | 120 | 80 |
| Images with no defect box | 228 | 14 | 12 |
| `packaging-error` boxes | 571 | 34 | 23 |
| `unsealed` boxes | 352 | 81 | 49 |
| `twisted-meat` boxes | 24 | 3 | 4 |
| `loose-meat` boxes | 22 | 3 | 3 |
| `wrinkle` boxes | 6 | 0 | 0 |

Box counts are per distinct image, so augmentation copies aren't counted twice.

Findings:

- **The 3,549 files are 1,317 distinct images.** Roboflow made 3 exposure-augmented copies of each training image.
- **File names are not unique ids.** 45 frame numbers are each shared by two *different* images. Copies are therefore matched by image content: the correlation of normalised greyscale thumbnails. Copies score 0.95–1.0 and different images score below 0.8, with no pairs in between.
- **No image appears in more than one official split,** so the official split has no duplicate leakage and is used as is.
- **Only 3 images have copies with conflicting labels.**
- **Neighbouring frame numbers show different trays,** and belt brightness varies randomly from frame to frame with no drift over time. Time blocks explain 5% of its variance, and the lag-1 autocorrelation is 0.00 (`results/brightness_over_time.png`).
- **`wrinkle` is dropped:** it has 6 training images and none in validation or test, so it can be neither learned reliably nor evaluated.
- **Rare classes and a small test set:** `loose-meat` and `twisted-meat` have 3–4 test boxes each, so their scores will carry wide uncertainty.

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Audit the raw export and write the training-ready dataset (wrinkle removed)
python scripts/prepare_data.py --raw data/raw --out data/processed

# 2. Run the tests
pytest
```

`data/processed/data.yaml` can be passed straight to Ultralytics YOLO.

## Repository layout

```
src/porkinspect/    package code
  labels.py         read and validate YOLO label files, parse Roboflow file names
  index.py          group files into distinct images; leakage and label-conflict checks
  export.py         write the dataset with dropped classes removed
scripts/            command-line entry points
tests/              pytest tests
results/            small result files (JSON, figures) that are committed
data/               raw and processed data (ignored by git)
docs/               HTML API documentation (pdoc)
AI_USE.md           log of AI assistance
```

## Licence

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, © its Roboflow Universe author. It is not redistributed here.

## AI use

AI assistance is logged in [`AI_USE.md`](AI_USE.md).

# Pork rasher packaging inspection

MMA3001 (Numerical Methods and Machine Learning) individual project.

A computer-vision pipeline that looks at a camera frame of a packaged pork-rasher tray on a conveyor and finds packaging and product-placement defects. The aim is to support automated quality inspection, so that faulty trays can be pulled before distribution without relying on manual inspection.

> **Status: v0.2.** Data audit, YOLOv8n training, evaluation and the baseline comparison are done. Sensitivity runs (image size, model size, brightness) are in progress.

## Results so far (official test split: 80 images, 68 with a defect, 12 without)

The threshold is chosen on validation (the highest one that still catches at least 95% of defect frames) and applied once to test. Brackets show bootstrap 95% intervals.

| | YOLOv8n, 640 px | Baseline: HOG + RBF SVM |
|---|---|---|
| Frame-level ROC AUC | **0.92** | 0.74 |
| Defect frames caught | **64/68 (94%)** [88–99%] | 54/68 (79%) [70–88%] |
| Defect-free frames flagged | 3/12 [0–50%] | 5/12 |
| CPU time per image (1 thread, end to end) | 92 ms | 14 ms |

- **AUC difference** (YOLO − baseline): +0.18 [0.06, 0.31], paired bootstrap.
- **YOLO test mAP50 per class:**
  - unsealed 0.84;
  - packaging-error 0.53;
  - loose-meat 0.67 and twisted-meat 0.06, each from only 3–4 test boxes.
- **Model size and speed:** 3.1 M parameters, 8.75 GFLOPs, about 5 ms inference per image on a T4 GPU.

Details are in `results/yolov8n_640/`, `results/baseline/` and `results/comparison/`.

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

### Training (GPU)

Open [`notebooks/01_train_yolo.ipynb`](notebooks/01_train_yolo.ipynb) in Google Colab: [Open in Colab](https://colab.research.google.com/github/mmal0078/36137316-pork-rasher-inspection/blob/main/notebooks/01_train_yolo.ipynb). It clones this repo, prepares the data, then runs:

```bash
pip install ultralytics
python scripts/train_yolo.py --data data/processed/data.yaml --model yolov8n.pt \
    --epochs 60 --imgsz 640 --batch 16 --device 0 --out runs --name yolov8n_640
```

[`notebooks/02_sensitivity_runs.ipynb`](notebooks/02_sensitivity_runs.ipynb) trains three variants with identical settings: YOLOv8n at 320 px and 480 px, and YOLOv8s at 640 px.

### Analysis (CPU)

```bash
# Frame-level threshold sweep, bootstrap intervals and per-class box recall for one run
python scripts/analyse_run.py --run runs/yolov8n_640 --data data/processed

# Baseline: hand-crafted features + decision tree / SVM / MLP, chosen by validation AUC
python scripts/baseline.py --data data/processed --out results/baseline

# YOLO vs baseline on the same frames, with CPU timing
python scripts/compare.py --run runs/yolov8n_640 --data data/processed --baseline results/baseline
```

Training outputs go to `runs/<name>/`: best weights, the training log, `metrics.json` (validation and test mAP, model size, GFLOPs, timings), and every prediction at confidence ≥ 0.001 on the validation and test images (`predictions_*.json`). Thresholds can then be analysed without re-running the model.

## Repository layout

```
src/porkinspect/    package code
  labels.py         read and validate YOLO label files, parse Roboflow file names
  index.py          group files into distinct images; leakage and label-conflict checks
  export.py         write the dataset with dropped classes removed
  evaluate.py       IoU matching, frame-level metrics, threshold sweep, bootstrap intervals
  features.py       hand-crafted colour/shape and HOG features for the baseline
scripts/            prepare_data, train_yolo, analyse_run, baseline, compare
notebooks/          Colab notebooks: 01 training, 02 sensitivity runs
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

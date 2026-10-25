# Pork rasher packaging inspection

MMA3001 (Numerical Methods and Machine Learning) individual project.

**Docs site:** https://mmal0078.github.io/36137316-pork-rasher-inspection/ (API reference and test report)

A computer-vision pipeline that looks at a camera frame of a packaged pork-rasher tray on a conveyor and finds packaging and product-placement defects. The aim is to support automated quality inspection, so that faulty trays can be pulled before distribution without relying on manual inspection.

> **Status: v0.4.** Done: data audit, YOLOv8n training, evaluation, baseline comparison, brightness robustness, image-size and model-size sensitivity, input validation, docs and tests. Final model: YOLOv8n at 640 px.

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
- **Model size and speed:** 3.1 M parameters, 8.75 GFLOPs per image. On a T4 GPU, inference takes 4.7 ms per image in batches of 16, and 10–15 ms end to end one frame at a time. On one CPU thread it takes 86 ms.
- **Brightness:** with test images scaled from 0.4× to 2× and the threshold fixed, mAP50 stayed between 0.49 and 0.52, and 62–63 of the 68 defect frames were still caught.
- **Line speed:** at up to 200 packs/min (3.3 packs/s), the GPU model has time for about 20–30 frames per pack, and one CPU thread for about 3.

Details are in `results/yolov8n_640/`, `results/baseline/` and `results/comparison/`.

### Sensitivity: image size and model size (`results/sensitivity/`)

All four runs used identical settings (60 epochs, batch 16, seed 0, T4 GPU). Each threshold was chosen on validation (recall ≥ 95%).

| Run | GFLOPs | T4 inference (batch 16) | Val frame AUC | Test frame AUC [95% CI] | Defect frames caught | False alarms |
|---|---|---|---|---|---|---|
| YOLOv8n, 320 px | 2.2 | 1.9 ms | 0.89 | 0.91 [0.77–0.99] | 61/68 | 1/12 |
| YOLOv8n, 480 px | 4.9 | 2.8 ms | 0.90 | 0.91 [0.78–0.99] | 58/68 | 2/12 |
| **YOLOv8n, 640 px (chosen)** | 8.8 | 4.7 ms | 0.87 | 0.92 [0.79–0.99] | **64/68** | 3/12 |
| YOLOv8s, 640 px | 28.6 | 8.1 ms | 0.87 | 0.89 [0.71–0.99] | 58/68 | 2/12 |

**What the runs show**
- **No configuration is detectably better.** The paired test-AUC differences against v8n 640 all have 95% intervals that include 0.
- **Overall mAP50 (0.36–0.58) is not used to rank the runs.** It is dominated by loose-meat and twisted-meat, which have only 3–4 test boxes each.
- **The larger YOLOv8s overfits.** Its validation loss bottomed out at epoch 21, and it gains nothing for 3.3× the compute.

**Choice: YOLOv8n at 640 px.**
- **Speed is not a binding constraint.** Every configuration runs more than 60× faster than the 300 ms per pack available at 200 packs/min.
- **A missed defect is the costly error,** while a false alarm costs one recheck. The 640 px model gives small packaging creases the most resolution, and it caught the most defect frames.
- **The difference from 320 px is not significant.** 640 px caught 3 frames that 320 px missed, and 320 px caught none that 640 px missed (McNemar p = 0.25). Validation alone slightly favoured the smaller inputs.
- **A plant with limited hardware could use 320 px** at a quarter of the compute.

### Profiling and arithmetic intensity (`results/profiling/`, `scripts/profile_pipeline.py`)

**Where the time goes.** Profiled with cProfile on one CPU thread, one frame at a time, with files already in memory:

| Stage | Share of pipeline time |
|---|---|
| Network forward pass | 89% (about 82 ms) |
| Our input checks (`check_image`) | 4% (about 3 ms) |
| Ultralytics preprocessing | 4% |
| Post-processing (non-maximum suppression) | 1% |

Only the network is worth optimising, and reducing the input size is the effective lever (see the 320 px run). The first read of each file from disk added about 65 ms per image in this sandbox. A line camera delivers frames in memory, so this is excluded.

**Arithmetic intensity** of one 640 px forward pass:
- **Work:** 8.75 GFLOPs.
- **Memory traffic (lower-bound model):** 12.6 MB of weights, a 4.9 MB input, and 184 MB of layer outputs, each written once and read once, all as float32. That is 385 MB in total.
- **AI ≈ 22.7 FLOP/byte.** This is just below the T4's ridge point of 8.1 TFLOPS ÷ 320 GB/s ≈ 25 FLOP/byte, so the model sits near the memory-bound/compute-bound boundary on the T4.
- **Most traffic is activations, not weights.** This is typical of a small CNN on a large image.

**Achieved throughput:**
- T4 at batch 16: 8.75 GFLOP ÷ 4.7 ms ≈ 1.86 TFLOP/s, about 23% of FP32 peak. The roofline allows about 7.3 TFLOP/s, so small layers and launch overheads keep the GPU underused.
- One CPU thread: about 105 GFLOP/s.

## Problem, inputs and outputs

**Engineering context.** Pork rashers are packed in clear trays on an automated line, and faulty trays are currently found by eye. The defects that matter:
- unsealed or creased lid film, which causes spoilage and is a food-safety and recall risk;
- rashers that are loose or twisted, which customers reject.

This project asks whether a camera and a computer-vision model can detect and locate these defects accurately enough, and fast enough, to support line inspection. Published line speeds range from 90 to 200 packs per minute.

### Input

| | |
|---|---|
| What | One overhead camera frame of a tray on the conveyor |
| Format | RGB image (JPEG or PNG), 8 bits per channel (0–255) |
| Size | 720 × 540 px, as in the training data. The detector letterboxes it to 640 px. |
| Units | Pixels. There is no mm-per-pixel calibration, so sizes are fractions of the image. |
| Valid domain | The same camera view: dark mesh belt, clear trays, pork rashers. Belt brightness 17–109 of 255, the range seen in training. |
| Settings | Confidence threshold (default 0.16, chosen on validation) and detector input size (default 640) |

Every image in the dataset shows only **part** of a tray: the meat reaches the image edge in 100% of images and covers a median 4% of the frame.

### Output (`porkinspect.inspection.inspect_image`)

| Field | Type | Meaning |
|---|---|---|
| `defect_present` | bool | True if any detection has confidence ≥ the threshold. This would trigger a reject or alert. |
| `detections` | list | Each has `class` (`loose-meat`, `packaging-error`, `twisted-meat` or `unsealed`), `confidence` (0–1, a ranking score rather than a calibrated probability) and `box_xywhn` (box centre, width and height as fractions 0–1 of the image) |
| `max_confidence` | float | Highest confidence in the frame (0 if none) |
| `warnings` | list | Out-of-domain input warnings (see below) |
| `inference_ms` | float | Model time for this frame |

`defect_present = False` means *no defect visible in this frame*, not *tray OK*, because part of the tray is out of view. Tray-level pass/fail is out of scope. The `wrinkle` class is dropped (6 training images, none for testing).

### Invalid, missing or unsupported inputs

| Case | Handling |
|---|---|
| File missing, not an image, or truncated or corrupt | `ValueError` with the reason. The command-line tool reports it and exits with code 1. It is never treated as "no defect". |
| Greyscale, RGBA or palette image | Converted to RGB, with a warning |
| Size other than 720 × 540 | Accepted (resized), with a warning that accuracy was only measured at 720 × 540 |
| Belt brightness outside 17–109 | Accepted, with a warning. Accuracy was stable from 0.4× to 2× brightness in testing (`results/yolov8n_640/brightness_test.png`). |
| Malformed label file (training) | `ValueError` naming the bad line (wrong field count, class id out of range, coordinate outside [0, 1], zero-size box). A missing label file is an error, not "no defect". |
| Different product, camera or site | Not detectable from the image alone. The model will still output something, so results are only valid for this setup. |

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
pip install ultralytics   # needed for training, inspection and profiling
```

**Versions used for the reported results:**
- Training on Google Colab (Tesla T4, 15 GB): Ultralytics 8.4.188, PyTorch 2.11.0+cu130, Python 3.13.16.
- Analysis on Linux (x86-64 CPU): Python 3.13.17, Ultralytics 8.4.188, PyTorch 2.14.1, NumPy 2.5.3, scikit-learn 1.9.1, scikit-image 0.26.0, Pillow 12.3.0, matplotlib 3.11.2, pytest 9.1.1, pdoc 16.0.0.

The final trained model is included at `models/yolov8n_640_best.pt` (6.5 MB), so inspection and the analysis scripts run without retraining. See `models/README.md` for its licence.

## Usage

```bash
# 1. Audit the raw export and write the training-ready dataset (wrinkle removed)
python scripts/prepare_data.py --raw data/raw --out data/processed

# 2. Run the tests (and write the HTML test report)
pytest --html=docs/test_report.html --self-contained-html

# 3. Inspect images with the trained model (one JSON line per image)
python scripts/inspect_image.py --weights models/yolov8n_640_best.pt path/to/frame.jpg

# 4. Rebuild the docs site (pdoc API reference + test report + landing page)
python scripts/build_docs.py
```

Documentation is published at **https://mmal0078.github.io/36137316-pork-rasher-inspection/**: the API reference generated by pdoc from the docstrings, and the latest pytest report. GitHub Pages serves it from the `docs/` folder on `main`.

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

# Brightness robustness (threshold fixed at the validation-chosen value)
python scripts/brightness_test.py --run runs/yolov8n_640 --data data/processed --threshold 0.16

# Profile the per-frame pipeline and estimate arithmetic intensity
python scripts/profile_pipeline.py --weights models/yolov8n_640_best.pt --data data/processed

# Image-size / model-size comparison across runs
python scripts/compare_runs.py --runs runs/yolov8n_640 runs/yolov8n_320 runs/yolov8n_480 runs/yolov8s_640 --data data/processed
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
  inspection.py     run the detector on one image, with input validation
scripts/            prepare_data, train_yolo, analyse_run, baseline, compare, compare_runs, brightness_test,
                    profile_pipeline, inspect_image, build_docs
notebooks/          Colab notebooks: 01 training, 02 sensitivity runs
  executed/         the notebooks as actually run, with their outputs
models/             final trained detector (AGPL-3.0, see models/README.md)
tests/              pytest tests
results/            small result files (JSON, figures) that are committed
data/               raw and processed data (ignored by git)
docs/               docs site: landing page, pdoc API reference, pytest report (GitHub Pages)
AI_USE.md           log of AI assistance
```

## Licence

- **Code:** MIT (see `LICENSE`).
- **Trained weights** (`models/`): AGPL-3.0, because they are derived from Ultralytics YOLOv8.
- **Dataset:** CC BY 4.0, © its Roboflow Universe author. It is not redistributed here.
- **Ultralytics** (used as a dependency) is AGPL-3.0.

## AI use

AI assistance is logged in [`AI_USE.md`](AI_USE.md).

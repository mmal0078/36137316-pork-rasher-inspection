# Data folder

Not tracked by git.

- `raw/`: the Roboflow export, **v4, YOLOv8 format**, from https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4 (CC BY 4.0). Unzip it here, or pass the zip path to `scripts/prepare_data.py`.
- `processed/`: written by `scripts/prepare_data.py`. It keeps the official split and removes the `wrinkle` class.

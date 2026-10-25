"""porkinspect: defect detection for packaged pork rashers (MMA3001 project).

Modules:
    labels: read YOLO label files and Roboflow file names.
    index: group exported files into distinct images; detect leakage and label conflicts.
    export: write a training-ready dataset on the official split with dropped classes removed.
    evaluate: IoU matching, frame-level metrics, threshold sweeps and bootstrap intervals.
    features: hand-crafted colour/shape and HOG features for the baseline classifier.
    inspection: run the trained detector on one image, with input validation.
"""

__version__ = "0.4.0"

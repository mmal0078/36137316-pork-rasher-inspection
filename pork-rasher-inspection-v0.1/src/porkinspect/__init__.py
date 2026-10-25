"""porkinspect: defect detection for packaged pork rashers (MMA3001 project).

Modules:
    labels: read YOLO label files and Roboflow file names.
    index: group exported files into distinct images; detect leakage and label conflicts.
    export: write a training-ready dataset on the official split with dropped classes removed.
"""

__version__ = "0.1.0"

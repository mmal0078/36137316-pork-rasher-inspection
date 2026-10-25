# Trained model

`yolov8n_640_best.pt` is the final detector: YOLOv8n fine-tuned at 640 px for 60 epochs (seed 0), the best epoch (50) chosen by validation fitness. It was produced by `notebooks/01_train_yolo.ipynb` (executed copy in `notebooks/executed/`). Its metrics are in `results/yolov8n_640/`.

- Classes (ids 0-3): `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`
- Input: RGB image, letterboxed to 640 px
- Recommended frame threshold: 0.16, chosen on validation so that at least 95% of defect frames are caught

```bash
python scripts/inspect_image.py --weights models/yolov8n_640_best.pt path/to/frame.jpg
```

## Licence (different from the rest of this repository)

These weights were fine-tuned from Ultralytics' pretrained YOLOv8n weights using the Ultralytics library, which is licensed under **AGPL-3.0**. Treat this file as a derivative covered by AGPL-3.0, not by the MIT licence that covers the code in this repository. The training data is CC BY 4.0 (see the main README).

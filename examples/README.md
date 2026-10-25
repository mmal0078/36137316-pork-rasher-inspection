# Example images

These are 8 images from the **test split**, which the model never saw during training. They are here so the model can be tried without downloading the dataset. `examples.csv` gives each image's source file name, the labelled defects (boxes as normalised `x, y, w, h`) and what the final model did with it on the test set.

They were chosen to show both strengths and failures. Each outcome was computed by matching the model's boxes to the labels (IoU ≥ 0.5, threshold 0.16):
- **Successes:** unsealed, packaging-error and loose-meat, each found with the correct type; and a defect-free frame correctly passed.
- **Wrong type:** a twisted-meat frame that is flagged but called `packaging-error`. No twisted-meat box in the test set was given the right type.
- **Partly found:** a frame with two defects where only one is found.
- **Missed:** a twisted-meat frame that is missed entirely.
- **False alarm:** the lid film looks wrinkled, but no defect was labelled.

**Source and licence:** [Pork Rasher Error (Packaging), v4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4) by Roboflow Universe user "Hello", licensed **CC BY 4.0**. The images are unmodified (720 × 540, as exported) and renamed only.

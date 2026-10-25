# AI use log

The brief requires disclosure of all material AI use. This file is a running log, kept as the work happens; the reflection in the report is written from it.

**Tool:** Claude (Anthropic), used in a chat session that could run code on a copy of the data.

## Session 1: 25 Oct 2026, data audit and repo v0.1

**What AI did**
- Read the project brief and the dataset slide.
- Wrote the data-audit code (`src/porkinspect/`, `scripts/prepare_data.py`) and its tests.
- Ran the audit on the dataset and drafted this README.

**My decisions**
- Chose Dataset 3.
- Scoped the output to frame-level detection rather than tray pass/fail.
- Dropped `wrinkle`.
- Used the official split.
- Asked AI to verify its duplicate claim ("are you sure?"), which led to the correction below.

**AI errors caught**

| # | What AI claimed | How it was checked | Result |
|---|---|---|---|
| 1 | Official test frames are near-copies of training frames, because their frame numbers are only 2–3 apart. | Looked at neighbouring frames side by side, and compared pixel differences. | Wrong: neighbouring frame numbers show different trays. |
| 2 | Lighting drifts over the recording, so testing on a separate time period would be a fairer test. | Measured belt brightness across all frames. | Wrong: brightness varies randomly frame to frame and does not drift (time blocks explain 5% of variance). |
| 3 | 13 images appear in two splits, and 46 have conflicting labels. | I asked whether it was sure; AI then compared the actual pixels rather than the file names. | Wrong: the same file name is used by different images. In reality 0 images are in two splits and 3 have conflicting labels. |

**Lesson:** all three errors came from AI inferring from file names or frame numbers instead of checking the images. Each claim had to be tested against the data before it was used. The final audit code checks image content directly, and tests cover the shared-file-name case.

**How I checked AI output**
- Ran the code.
- Looked at the sample images and figures.
- Asked for evidence before accepting claims.

## Session 1 (continued): training script and notebook

**What AI did**
- Wrote `scripts/train_yolo.py` and `notebooks/01_train_yolo.ipynb`.
- Smoke-tested the script on CPU before handing it over: 1 epoch, 5% of the data, 160 px.

**Bugs the smoke test caught (before I ran anything)**
- The generated `data.yaml` used a relative `path: .`. Ultralytics resolves that from the working directory, so the images weren't found. The unit test had passed because it only checked the file's text. The fix writes an absolute path, and a test now checks for it.
- `model.info()` returns nothing in Ultralytics 8.4, so the parameter and FLOPs count crashed. The fix computes them directly. The result, 8.75 GFLOPs at 640 px, matches the published YOLOv8n figure (8.7).

**Another claim corrected by looking at the data**
- AI first described `packaging-error` as "torn, misplaced or folded film", guessing from the class name.
- Zoomed crops of 12 random boxes per class showed it actually marks small creases or pinches in the lid film, almost always where a rasher presses against the film near the tray edge.
- `unsealed` marks a large area of loose, wrinkled film. `twisted-meat` marks a rasher on its edge or flipped. `loose-meat` marks a rasher out of line or a stray piece.

## Session 1 (continued): baseline and comparison

**What AI did**
- Wrote `features.py` (20 colour/shape features, HOG), `baseline.py` (tree, SVM and MLP grid) and `compare.py`, with tests.

**Errors caught along the way**
- **The hue feature was wrong.** Red sits where the hue scale wraps from 1 back to 0, so the plain mean came out at a meaningless 0.705. Fixed by shifting the hue half a turn; a test now checks it.
- **The first model-selection metric was misleading.** Average precision was used first, but 88% of frames contain a defect, so random scores already get about 0.88 AP. Switched to ROC AUC, where random is 0.5, and reran.
- **No cherry-picking.** On test, the colour/shape SVM scored higher (AUC 0.87) than the configuration chosen on validation (HOG SVM, test AUC 0.74). The validation choice is reported anyway, because choosing by test score would bias the comparison.

**Notebook 02 hid the training log**
- AI ran training with `subprocess.run`, so Colab didn't show the child process's output. The cell looked stuck while it was actually training.
- I noticed and asked. The fix streams the output line by line.

**Input-validation bug found by testing the command-line tool**
- A corrupt file crashed the tool with `ModuleNotFoundError` instead of a clear error. Ultralytics replaces `PIL.Image.open` when it is imported, and its fallback raises a different exception.
- The fix catches it. A regression test now checks this with Ultralytics imported.
- The brightness limit was first set too narrow: it flagged a real test image the model handled correctly. It is now set from the measured training range (17-109).

## Sensitivity runs (image size, model size)

**AI's prediction was not supported**
- AI predicted that 320 px images would lose the small packaging-error creases.
- Test box recall for packaging-error was 14/23 at 320 px vs 15/23 at 640 px: no detectable loss.

**A misleading plot, caught before use**
- The first accuracy-vs-cost plot had a y-axis spanning only 0.888-0.915 and no error bars. That made differences of about 0.01 look large.
- Redrawn on a 0.6-1.0 axis with 95% bootstrap and Wilson intervals. These show all four runs overlap.

**Overall test mAP50 needed care**
- The overall figure (0.36-0.58 across runs) moves mainly because of loose-meat and twisted-meat, with 3-4 test boxes each. It is not used to rank the models.

**My decision on the final model**
- AI laid out two options:
  - (A) YOLOv8n at 320 px, chosen on validation with the Week 5 "equally good, so pick the simplest" rule;
  - (B) YOLOv8n at 640 px.
- I asked what would be best for a real packing plant, and chose **B** on engineering grounds:
  - a missed defect costs far more than a false alarm;
  - speed is not a constraint at real line speeds of 90-200 packs/min.
- I had already seen the test results when choosing, and the report says so. The difference from 320 px is not significant (p = 0.25).

## Final checks: profiling, arithmetic intensity, reproducibility

**What AI did**
- Added `profile_pipeline.py` (cProfile, arithmetic intensity, roofline), `build_docs.py` and the trained weights with an AGPL-3.0 note.
- Recorded the exact library versions and added the executed notebooks.

**AI errors caught**
- **First profiling read:** AI's first reading blamed our input check (35% of the time) on Ultralytics' patched `Image.open`. Timing it directly showed 3 ms per image either way. The cost was the files being read from disk for the first time. The profile now excludes the cold disk pass, which is reported separately.
- **Mislabelled GPU time:** AI had been quoting "about 5 ms per image on the T4" without saying this came from batch-16 validation. One frame at a time it is 10-15 ms end to end. The README and profiling output now label both.
- **Licence:** AI had not mentioned that Ultralytics and the fine-tuned weights are AGPL-3.0, not MIT, until the final audit.

## Try-the-model notebook (03)

**What AI did**
- Wrote `draw_report` (with a test), chose 8 example test images and wrote notebook 03.
- Ran the notebook end to end before handing it over.

**AI errors caught**
- **Misleading example labels.** AI first labelled examples "caught" when only the frame was flagged. Running the notebook showed that the twisted-meat and packaging-error examples were flagged as the wrong defect type. The examples were re-picked (4 genuine successes, 4 labelled failures), and the outcomes are now computed by box matching rather than written by hand.
- **Wrong arguments in AI's example script.** `match_image(dets, gts, T, True)` passed `True` as the IoU threshold (meaning IoU = 1) instead of switching on class matching, which garbled the first computed outcomes.
  - The fix makes `iou_thr` and `class_aware` keyword-only, with a test.
  - Earlier results were unaffected: every other caller passed the IoU threshold correctly.

## My reflection

The code was written primarily by Claude, and Claude drafted the README and the report, which I read through, edited and shortened. The analysis was roughly even: Claude often presented options as if it had already made up its mind, and I challenged it, asking it to weigh them objectively and whether they were relevant to the engineering problem. Many of these challenges changed the outcome, such as keeping the official split and choosing the 640 px model. Most of Claude's errors came from assumptions: it would start looking at the data and then extrapolate instead of confirming, so I questioned every response and file it sent me. I understood the theory of CNNs from the unit, but only once I made Claude slow down and explain not just what it was doing but why did I connect the dots and see that theory in a real engineering setting. The full reflection is in Section 7 of the report.

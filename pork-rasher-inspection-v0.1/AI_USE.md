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

*(To complete: my own reflection on how AI use affected my understanding and workflow.)*

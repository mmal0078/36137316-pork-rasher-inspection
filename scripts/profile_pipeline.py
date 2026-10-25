"""Profile the inspection pipeline and estimate the detector's arithmetic intensity.

Three parts, all on this machine's CPU:

1. **cProfile** of ``inspect_image`` on test images, batch size 1 (the way a line
   camera feeds frames), after one unprofiled pass so files are already in memory. Shows where wall time goes: our input checks, image
   decoding, preprocessing, the network, post-processing (NMS).
2. **Arithmetic intensity** (AI = FLOPs / bytes of memory traffic) of one
   forward pass at 640 px.
   - FLOPs come from Ultralytics/thop, which counts 2 FLOPs per multiply-accumulate.
   - Memory traffic is a simple lower-bound model: every weight read once, the
     input read once, and every layer output written once and read once,
     all in float32 (4 bytes).
   - The model ignores caches and re-reads, so the true AI is somewhat lower
     than the estimate.
   - Comparing AI with a processor's "ridge point" (peak FLOP/s divided by
     memory bandwidth) says whether the model is compute-bound or memory-bound.
3. **Achieved throughput**: FLOPs divided by measured inference time, for this
   CPU (batch 1) and for the Colab T4 time recorded in ``results/<run>/metrics.json``
   (Ultralytics validation, batch 16, so per-image GPU time is lower than at batch 1).

Usage::

    python scripts/profile_pipeline.py --weights runs/yolov8n_640/weights/best.pt \
        --data data/processed --metrics results/yolov8n_640/metrics.json --out results/profiling
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porkinspect.inspection import inspect_image  # noqa: E402

#: NVIDIA T4 datasheet figures (Lenovo Press LP1926): FP32 peak and memory bandwidth.
T4_FP32_FLOPS = 8.1e12
T4_BANDWIDTH = 320e9


def activation_bytes(torch_model, imgsz: int) -> tuple[int, int, int]:
    """Bytes of weights, input and all leaf-layer outputs for one float32 forward pass.

    Returns:
        ``(weight_bytes, input_bytes, activation_bytes)``.
    """
    import torch

    sizes = []

    def hook(_m, _inp, out):
        for t in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(t, torch.Tensor):
                sizes.append(t.numel() * 4)

    leaves = [m for m in torch_model.modules() if not list(m.children())]
    handles = [m.register_forward_hook(hook) for m in leaves]
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        torch_model.float().eval()(x)
    for h in handles:
        h.remove()
    weights = sum(p.numel() for p in torch_model.parameters()) * 4
    return weights, x.numel() * 4, int(sum(sizes))


def main() -> None:
    """Profile, estimate arithmetic intensity, and write ``profiling.json``."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--metrics", type=Path, default=ROOT / "results" / "yolov8n_640" / "metrics.json")
    ap.add_argument("--images", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "profiling")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(a.weights)
    paths = sorted((a.data / "test" / "images").iterdir())[: a.images]
    # Cold pass: the first read of each file comes from disk. A line camera delivers frames in
    # memory, so disk reads are timed separately and excluded from the profile below.
    t0 = time.perf_counter()
    for p in paths:
        inspect_image(p, a.weights, model=model, imgsz=a.imgsz)
    cold_ms = (time.perf_counter() - t0) * 1000 / len(paths)

    # 1. cProfile over the whole per-image pipeline.
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    for p in paths:
        inspect_image(p, a.weights, model=model, imgsz=a.imgsz)
    prof.disable()
    wall_ms = (time.perf_counter() - t0) * 1000 / len(paths)

    # Per-stage times reported by Ultralytics for the same calls (re-run without the profiler overhead).
    stage = {"preprocess": [], "inference": [], "postprocess": []}
    for p in paths:
        r = model.predict(str(p), imgsz=a.imgsz, conf=0.001, device="cpu", verbose=False)[0]
        for k in stage:
            stage[k].append(r.speed[k])
    stage_ms = {k: round(float(np.mean(v)), 2) for k, v in stage.items()}

    s = io.StringIO()
    st = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    st.print_stats(25)
    (a.out / "cprofile_inspect_image.txt").write_text(s.getvalue())

    # Fraction of profiled time in named parts of the pipeline (cumulative time per function).
    stats = st.stats  # {(file, line, func): (cc, nc, tt, ct, callers)}
    total = sum(v[2] for v in stats.values())  # total own time

    def cum(pred):
        return sum(v[3] for k, v in stats.items() if pred(k))

    parts = {
        "check_image (our input validation)": cum(lambda k: k[2] == "check_image"),
        "Ultralytics preprocess": cum(lambda k: k[2] == "preprocess" and "ultralytics" in k[0]),
        "network forward pass": cum(lambda k: k[2] == "inference" and "ultralytics" in k[0]),
        "Ultralytics postprocess (NMS)": cum(lambda k: k[2] == "postprocess" and "ultralytics" in k[0]),
    }
    parts = {k: round(100 * v / total, 1) for k, v in parts.items()}

    # 2. Arithmetic intensity at the deployed input size.
    gflops = float(get_flops(model.model, imgsz=a.imgsz))
    wb, ib, ab = activation_bytes(model.model, a.imgsz)
    traffic = wb + ib + 2 * ab  # activations written once and read once
    ai = gflops * 1e9 / traffic
    ridge_t4 = T4_FP32_FLOPS / T4_BANDWIDTH

    # 3. Achieved throughput.
    m = json.loads(a.metrics.read_text())
    t4_ms = m["test"]["speed_ms_per_image"]["inference"]
    out = {
        "images_profiled": len(paths), "torch_threads": torch.get_num_threads(),
        "cpu_wall_ms_per_image_cold_files_first_pass": round(cold_ms, 1),
        "cpu_wall_ms_per_image_warm_with_profiler": round(wall_ms, 1),
        "cpu_stage_ms_per_image": stage_ms,
        "share_of_profiled_time_percent": parts,
        "GFLOPs_per_image": round(gflops, 3),
        "bytes": {"weights": wb, "input": ib, "activations_once": ab, "modelled_traffic": traffic},
        "arithmetic_intensity_flop_per_byte": round(ai, 1),
        "t4_ridge_point_flop_per_byte": round(ridge_t4, 1),
        "bound_on_t4": "compute-bound" if ai > ridge_t4 else "memory-bound",
        "achieved_GFLOPs_per_s": {
            "cpu_inference": round(gflops / (stage_ms["inference"] / 1000), 1),
            "t4_inference_batch16": round(gflops / (t4_ms / 1000), 1),
        },
        "note": "T4 time is Ultralytics validation inference per image at batch 16 (FP32); CPU time is batch 1, 1 thread",
        "t4_fraction_of_fp32_peak_batch16": round(gflops * 1e9 / (t4_ms / 1000) / T4_FP32_FLOPS, 3),
        "roofline_t4_attainable_GFLOPs_per_s": round(min(T4_FP32_FLOPS, ai * T4_BANDWIDTH) / 1e9, 0),
    }
    (a.out / "profiling.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

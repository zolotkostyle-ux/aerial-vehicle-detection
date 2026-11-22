"""mAP@0.5 of the frozen auto-labeller against the reviewed labels on the held-out eval clips.

Single class `vehicle`, COCO-style 101-point interpolated AP at IoU 0.5. Predictions are the frozen
auto-labels with their confidence scores; ground truth is the reviewed export.

What counts, following the COCO convention for ignored objects:
  - ground truth that is `truncated: yes` or lies outside the band is ignored: a prediction that
    matches it is neither a true nor a false positive, and it is not a miss;
  - a prediction whose centre is in an `ignore` zone is dropped;
  - for a band, an unmatched prediction counts as a false positive only if its own box falls in that band.

usage: map_eval.py [clip ...]   (default: eval eval_far)
"""
import sys
from collections import defaultdict

import numpy as np

from common import band_of, distance_m, frame_sizes, iou, read_rows, zone_polygon
from qc_compare import FRAMES_V1, REVIEWED, box_in_ignore, load_baseline, load_reviewed

IOU_THRESHOLD = 0.5
RANGES = {"0-200 m": ("0-200 m",), "200-400 m": ("200-400 m",), "0-400 m (both bands)": ("0-200 m", "200-400 m")}


def band(box, size, size_class):
    return band_of(distance_m(box, size, size_class))


def average_precision(predictions, gt_count):
    """predictions: list of (confidence, is_true_positive)."""
    if gt_count == 0:
        return float("nan")
    predictions.sort(key=lambda p: -p[0])
    hits = np.array([p[1] for p in predictions], dtype=float)
    tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    points = np.linspace(0, 1, 101)
    index = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in index]))


def evaluate(frames, auto, reviewed, ignore, sizes, keep_bands):
    scored, gt_count = [], 0
    for key in frames:
        size = sizes[key]
        zones = [zone_polygon(z) for z in ignore.get(key, [])]
        gts = [(g["box"], g["truncated"] == "no" and band(g["box"], size, g["size_class"]) in keep_bands)
               for g in reviewed.get(key, [])]
        gt_count += sum(counted for _, counted in gts)
        preds = sorted((p for p in auto.get(key, []) if not box_in_ignore(p["box"], zones)), key=lambda p: -p["confidence"])
        used = set()
        for p in preds:
            best, best_iou = None, IOU_THRESHOLD
            for gi, (gbox, _) in enumerate(gts):
                if gi in used:
                    continue
                value = iou(p["box"], gbox)
                if value >= best_iou:
                    best, best_iou = gi, value
            if best is not None:
                used.add(best)
                if gts[best][1]:
                    scored.append((p["confidence"], True))
            elif band(p["box"], size, p["size_class"]) in keep_bands:
                scored.append((p["confidence"], False))
    return average_precision(scored, gt_count), gt_count, len(scored)


def main():
    clips = sys.argv[1:] or ["eval", "eval_far"]
    sizes = frame_sizes()
    frames = sorted(k for k in ((r["clip"], r["file"]) for r in read_rows(FRAMES_V1)) if k[0] in clips)
    auto = load_baseline(set(frames), sizes)
    reviewed, ignore, _ = load_reviewed(REVIEWED)
    print(f"mAP@0.5, frozen auto-labeller vs reviewed labels, clips {', '.join(clips)}, {len(frames)} frames")
    print(f"{'range':22}{'GT':>6}{'scored preds':>14}{'AP@0.5':>9}")
    for name, keep in RANGES.items():
        ap, gt, n = evaluate(frames, auto, reviewed, ignore, sizes, keep)
        print(f"{name:22}{gt:>6}{n:>14}{ap:>9.3f}")


if __name__ == "__main__":
    main()

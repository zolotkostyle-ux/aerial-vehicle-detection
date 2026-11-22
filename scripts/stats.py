"""Dataset statistics per clip and per distance band.

usage: stats.py [labels.csv] [--all-frames]
Defaults to labels/reviewed_v1.csv on the 69 review frames in labels/frames_v1.csv.
Pass labels/cv_all_frames.csv to inspect the raw auto-labeller output across all 574 frames.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (
    BANDS, CV_LABELS, FRAMES_V1, MANIFEST, REVIEWED,
    box_of, focal_px, frame_sizes, read_rows, row_band, vehicle_length_px, zone_polygon,
)

ROUND_ORDER = ["trainA", "trainB", "trainC", "trainD", "trainE", "trainF", "eval", "eval_far"]


def ignore_area_share(clip_frames, ignore_by_file, sizes):
    total_px = covered_px = 0
    for clip, fname in clip_frames:
        w, h = sizes[(clip, fname)]
        total_px += w * h
        zones = ignore_by_file.get((clip, fname))
        if not zones:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        for row in zones:
            pts = np.round(np.array(zone_polygon(row), dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        covered_px += int(mask.sum())
    return (100.0 * covered_px / total_px) if total_px else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="?", default=str(REVIEWED))
    parser.add_argument("--all-frames", action="store_true",
                        help="count empty frames against all frames in frame_manifest.csv")
    args = parser.parse_args()

    path = Path(args.labels)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    rows = read_rows(path)
    sizes = frame_sizes()
    use_all = args.all_frames or path.resolve() == CV_LABELS.resolve()
    frame_rows = read_rows(MANIFEST if use_all else FRAMES_V1)
    frames_by_clip = defaultdict(list)
    for r in frame_rows:
        frames_by_clip[r["clip"]].append((r["clip"], r["file"]))
    allowed_frames = {(r["clip"], r["file"]) for r in frame_rows}

    for size in sorted(set(sizes.values())):
        focal = focal_px(*size)
        print(f"{size[0]}x{size[1]}: focal length {focal:.0f} px; a 4.5 m car is "
              f"{focal * 4.5 / 200:.1f} px long at 200 m and {focal * 4.5 / 400:.1f} px at 400 m")
    print(f"Scope: {path.name} on {len(allowed_frames)} frames ({'all extracted' if use_all else 'frames_v1.csv'})\n")

    per_clip = defaultdict(Counter)
    lengths = defaultdict(list)
    frames_with_veh = defaultdict(set)
    ignore_by_file = defaultdict(list)
    ignore_count = Counter()
    large_count = Counter()
    trunc_count = Counter()
    total_veh = Counter()

    for row in rows:
        key = (row["clip"], row["file"])
        if key not in allowed_frames:
            continue
        if row.get("source_class") == "two_wheeler":
            continue
        if row.get("label") == "ignore":
            ignore_by_file[key].append(row)
            ignore_count[row["clip"]] += 1
            continue
        clip = row["clip"]
        total_veh[clip] += 1
        frames_with_veh[clip].add(row["file"])
        size_class = row.get("size_class") or "car"
        if size_class == "large" or (not row.get("size_class") and row.get("source_class") in ("truck", "bus")):
            large_count[clip] += 1
        if row.get("truncated") == "yes":
            trunc_count[clip] += 1
            continue
        box = box_of(row)
        band = row_band({**row, "size_class": size_class}, sizes)
        per_clip[clip][band] += 1
        lengths[clip].append(vehicle_length_px(box, size_class))

    names = [b[0] for b in BANDS]
    clips = [c for c in ROUND_ORDER if c in frames_by_clip] + sorted(set(frames_by_clip) - set(ROUND_ORDER))
    has_ignore = sum(ignore_count.values()) > 0
    hdr = (f"{'clip':9}{'frames':>7}{'empty':>6}{'veh':>6}"
           + "".join(f"{n:>13}" for n in names)
           + f"{'large':>7}{'trunc':>7}{'med px':>8}"
           + (f"{'ign':>6}{'ign%':>7}" if has_ignore else ""))
    print(hdr)
    print("-" * len(hdr))

    tot_frames = tot_empty = tot_veh_sum = tot_large = tot_trunc = tot_ign = 0
    tot_bands = Counter()
    all_lengths = []
    all_clip_frames = []

    for clip in clips:
        n_frames = len(frames_by_clip[clip])
        n_empty = n_frames - len(frames_with_veh[clip])
        sides = sorted(lengths[clip])
        median = sides[len(sides) // 2] if sides else 0.0
        ign_pct = ignore_area_share(frames_by_clip[clip], ignore_by_file, sizes) if has_ignore else 0.0
        line = (f"{clip:9}{n_frames:>7}{n_empty:>6}{total_veh[clip]:>6}"
                + "".join(f"{per_clip[clip][n]:>13}" for n in names)
                + f"{large_count[clip]:>7}{trunc_count[clip]:>7}{median:>8.0f}"
                + (f"{ignore_count[clip]:>6}{ign_pct:>6.1f}%" if has_ignore else ""))
        print(line)
        tot_frames += n_frames
        tot_empty += n_empty
        tot_veh_sum += total_veh[clip]
        tot_large += large_count[clip]
        tot_trunc += trunc_count[clip]
        tot_ign += ignore_count[clip]
        tot_bands.update(per_clip[clip])
        all_lengths.extend(lengths[clip])
        all_clip_frames.extend(frames_by_clip[clip])

    all_lengths.sort()
    all_med = all_lengths[len(all_lengths) // 2] if all_lengths else 0.0
    tot_ign_pct = ignore_area_share(all_clip_frames, ignore_by_file, sizes) if has_ignore else 0.0
    print("-" * len(hdr))
    print(f"{'Total':9}{tot_frames:>7}{tot_empty:>6}{tot_veh_sum:>6}"
          + "".join(f"{tot_bands[n]:>13}" for n in names)
          + f"{tot_large:>7}{tot_trunc:>7}{all_med:>8.0f}"
          + (f"{tot_ign:>6}{tot_ign_pct:>6.1f}%" if has_ignore else ""))


if __name__ == "__main__":
    main()

"""Compare reviewed labels against the frozen YOLO-World baseline (spec sections 3, 5, 7, 9).

usage:
  python scripts/qc_compare.py                             # all clips present in labels/reviewed_v1.csv
  python scripts/qc_compare.py trainA                      # only round 1 (trainA)
  python scripts/qc_compare.py --reviewed path.csv --log docs/review_log.csv

Auto-labels are loaded from labels/cv_all_frames.csv restricted to labels/frames_v1.csv, with the
exact pre-filled hints pushed by cvat_sync.py (truck/bus -> large, within 2 px of edge -> truncated).
Matching is greedy 1-to-1 by IoU >= 0.30; a matched pair is Keep if all four coordinates are within
max(2 px, 10% of the reviewed box's long side) and both attributes match, otherwise Fix.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (
    BANDS, CV_LABELS, FRAMES_V1, REVIEWED, ROOT,
    box_of, frame_sizes, iou, point_in_polygon, read_rows, row_band, zone_polygon,
)

ROUND_ORDER = ["trainA", "trainB", "trainC", "trainD", "trainE", "trainF", "eval", "eval_far"]
VEHICLE_MAP = {"car": "car", "truck": "large", "bus": "large"}
EDGE_PX = 2.0
MATCH_IOU = 0.30
IGNORE_TRAIN_LIMIT_PCT = 25.0


def load_baseline(v1_frames, sizes):
    by_cf = defaultdict(list)
    for row in read_rows(CV_LABELS):
        key = (row["clip"], row["file"])
        if key not in v1_frames or row["source_class"] not in VEHICLE_MAP:
            continue
        x1, y1, x2, y2 = box_of(row)
        w, h = sizes[key]
        at_edge = x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= w - EDGE_PX or y2 >= h - EDGE_PX
        by_cf[key].append({
            "clip": row["clip"],
            "file": row["file"],
            "box": (x1, y1, x2, y2),
            "size_class": VEHICLE_MAP[row["source_class"]],
            "truncated": "yes" if at_edge else "no",
            "confidence": float(row["confidence"]),
        })
    return by_cf


def load_reviewed(path):
    veh_by_cf = defaultdict(list)
    ign_by_cf = defaultdict(list)
    clips_seen = set()
    for row in read_rows(path):
        key = (row["clip"], row["file"])
        clips_seen.add(row["clip"])
        if row.get("label") == "ignore":
            ign_by_cf[key].append(row)
        else:
            veh_by_cf[key].append({
                "clip": row["clip"],
                "file": row["file"],
                "box": box_of(row),
                "size_class": row.get("size_class") or "car",
                "truncated": row.get("truncated") or "no",
            })
    return veh_by_cf, ign_by_cf, clips_seen


def match_frame(auto_items, rev_items):
    """Greedy 1-to-1 matching by highest IoU >= MATCH_IOU."""
    pairs = []
    for ai, a in enumerate(auto_items):
        for ri, r in enumerate(rev_items):
            score = iou(a["box"], r["box"])
            if score >= MATCH_IOU:
                pairs.append((score, ai, ri))
    pairs.sort(reverse=True)
    used_a, used_r, matched = set(), set(), []
    for score, ai, ri in pairs:
        if ai not in used_a and ri not in used_r:
            used_a.add(ai)
            used_r.add(ri)
            matched.append((ai, ri, score))
    unmatched_a = [i for i in range(len(auto_items)) if i not in used_a]
    unmatched_r = [i for i in range(len(rev_items)) if i not in used_r]
    return matched, unmatched_a, unmatched_r


def within_tolerance(auto_box, rev_box):
    long_side = max(rev_box[2] - rev_box[0], rev_box[3] - rev_box[1])
    tol = max(2.0, 0.10 * long_side)
    return max(abs(a - r) for a, r in zip(auto_box, rev_box)) <= tol


def box_in_ignore(box, ignore_polys):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return any(point_in_polygon(cx, cy, poly) for poly in ignore_polys)


def frame_ignore_pct(zones, size):
    if not zones:
        return 0.0, 0, size[0] * size[1]
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    for row in zones:
        pts = np.round(np.array(zone_polygon(row), dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    covered = int(mask.sum())
    return 100.0 * covered / (w * h), covered, w * h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*", help="clips to compare (default: all clips present in reviewed CSV)")
    parser.add_argument("--reviewed", default=str(REVIEWED), help="path to reviewed CSV")
    parser.add_argument("--log", default=str(ROOT / "docs" / "review_log.csv"), help="optional timing log CSV")
    args = parser.parse_args()

    rev_path = Path(args.reviewed)
    if not rev_path.exists():
        raise SystemExit(f"reviewed file not found: {rev_path} (run scripts/cvat_sync.py pull after a round)")

    sizes = frame_sizes()
    v1_rows = read_rows(FRAMES_V1)
    v1_by_clip = defaultdict(list)
    for r in v1_rows:
        v1_by_clip[r["clip"]].append(r["file"])
    v1_set = {(r["clip"], r["file"]) for r in v1_rows}

    auto_by_cf = load_baseline(v1_set, sizes)
    rev_by_cf, ign_by_cf, clips_in_rev = load_reviewed(rev_path)

    if args.clips:
        clips = args.clips
    else:
        clips = [c for c in ROUND_ORDER if c in clips_in_rev] + sorted(clips_in_rev - set(ROUND_ORDER))
    if not clips:
        raise SystemExit(f"no shapes found in {rev_path}")

    # Counters
    clip_stats = defaultdict(Counter)
    band_gt = Counter()
    band_tp = Counter()
    band_fp = Counter()
    band_size_err = Counter()
    rev_bands_by_clip = defaultdict(Counter)
    ign_area_by_clip = defaultdict(lambda: [0, 0])
    over_limit_frames = []

    for clip in clips:
        for fname in sorted(v1_by_clip[clip]):
            key = (clip, fname)
            a_items = auto_by_cf.get(key, [])
            r_items = rev_by_cf.get(key, [])
            i_rows = ign_by_cf.get(key, [])
            i_polys = [zone_polygon(z) for z in i_rows]

            pct, cov_px, tot_px = frame_ignore_pct(i_rows, sizes[key])
            ign_area_by_clip[clip][0] += cov_px
            ign_area_by_clip[clip][1] += tot_px
            if clip.startswith("train") and pct > IGNORE_TRAIN_LIMIT_PCT:
                over_limit_frames.append((clip, fname, pct))

            cs = clip_stats[clip]
            cs["frames"] += 1
            cs["auto"] += len(a_items)
            cs["rev"] += len(r_items)
            cs["ign_zones"] += len(i_rows)

            for r in r_items:
                if r["size_class"] == "large":
                    cs["rev_large"] += 1
                if r["truncated"] == "yes":
                    cs["rev_trunc"] += 1
                else:
                    b = row_band({"clip": clip, "file": fname, "x1": r["box"][0], "y1": r["box"][1],
                                  "x2": r["box"][2], "y2": r["box"][3], "size_class": r["size_class"]}, sizes)
                    rev_bands_by_clip[clip][b] += 1
                    band_gt[b] += 1

            matched, un_a, un_r = match_frame(a_items, r_items)
            for ai, ri, _ in matched:
                a, r = a_items[ai], r_items[ri]
                box_ok = within_tolerance(a["box"], r["box"])
                size_ok = a["size_class"] == r["size_class"]
                trunc_ok = a["truncated"] == r["truncated"]
                if box_ok and size_ok and trunc_ok:
                    cs["keep"] += 1
                else:
                    cs["fix"] += 1
                    if not box_ok:
                        cs["fix_box"] += 1
                    if not size_ok:
                        cs["fix_size"] += 1
                    if not trunc_ok:
                        cs["fix_trunc"] += 1
                if r["truncated"] != "yes":
                    b = row_band({"clip": clip, "file": fname, "x1": r["box"][0], "y1": r["box"][1],
                                  "x2": r["box"][2], "y2": r["box"][3], "size_class": r["size_class"]}, sizes)
                    band_tp[b] += 1
                    if not size_ok:
                        band_size_err[b] += 1

            for ai in un_a:
                a = a_items[ai]
                cs["remove"] += 1
                if box_in_ignore(a["box"], i_polys):
                    cs["remove_ign"] += 1
                else:
                    cs["remove_fp"] += 1
                    if a["truncated"] != "yes":
                        b = row_band({"clip": clip, "file": fname, "x1": a["box"][0], "y1": a["box"][1],
                                      "x2": a["box"][2], "y2": a["box"][3], "size_class": "car"}, sizes)
                        band_fp[b] += 1

            cs["add"] += len(un_r)

    # Table 1: Reviewer edits vs frozen auto-labels
    print("=== 1. Reviewer decisions vs frozen auto-labels (spec §7) ===")
    h1 = (f"{'clip':9}{'fr':>4}{'auto':>6}{'rev':>6}{'keep':>6}{'fix':>6}"
          f"{'(box/sz/tr)':>13}{'rem':>6}{'(in_ign)':>9}{'add':>6}{'kept%':>7}")
    print(h1)
    print("-" * len(h1))
    tot = Counter()
    for clip in clips:
        c = clip_stats[clip]
        tot.update(c)
        kept_pct = 100.0 * c["keep"] / c["auto"] if c["auto"] else 0.0
        sub = f"{c['fix_box']}/{c['fix_size']}/{c['fix_trunc']}"
        print(f"{clip:9}{c['frames']:>4}{c['auto']:>6}{c['rev']:>6}{c['keep']:>6}{c['fix']:>6}"
              f"{sub:>13}{c['remove']:>6}{c['remove_ign']:>9}{c['add']:>6}{kept_pct:>6.1f}%")
    print("-" * len(h1))
    kept_pct = 100.0 * tot["keep"] / tot["auto"] if tot["auto"] else 0.0
    sub = f"{tot['fix_box']}/{tot['fix_size']}/{tot['fix_trunc']}"
    print(f"{'Total':9}{tot['frames']:>4}{tot['auto']:>6}{tot['rev']:>6}{tot['keep']:>6}{tot['fix']:>6}"
          f"{sub:>13}{tot['remove']:>6}{tot['remove_ign']:>9}{tot['add']:>6}{kept_pct:>6.1f}%")

    # Table 2: Reviewed coverage & ignore area
    print("\n=== 2. Reviewed coverage by range band & ignore share (spec §5, §6) ===")
    bnames = [b[0] for b in BANDS]
    h2 = (f"{'clip':9}{'fr':>4}{'veh':>6}" + "".join(f"{n:>13}" for n in bnames)
          + f"{'large':>7}{'trunc':>7}{'ign':>5}{'ign%':>7}")
    print(h2)
    print("-" * len(h2))
    tot_rb = Counter()
    tot_cov = tot_px_all = 0
    for clip in clips:
        c = clip_stats[clip]
        cov, tpx = ign_area_by_clip[clip]
        tot_cov += cov
        tot_px_all += tpx
        tot_rb.update(rev_bands_by_clip[clip])
        pct = 100.0 * cov / tpx if tpx else 0.0
        print(f"{clip:9}{c['frames']:>4}{c['rev']:>6}"
              + "".join(f"{rev_bands_by_clip[clip][n]:>13}" for n in bnames)
              + f"{c['rev_large']:>7}{c['rev_trunc']:>7}{c['ign_zones']:>5}{pct:>6.1f}%")
    print("-" * len(h2))
    pct_all = 100.0 * tot_cov / tot_px_all if tot_px_all else 0.0
    print(f"{'Total':9}{tot['frames']:>4}{tot['rev']:>6}"
          + "".join(f"{tot_rb[n]:>13}" for n in bnames)
          + f"{tot['rev_large']:>7}{tot['rev_trunc']:>7}{tot['ign_zones']:>5}{pct_all:>6.1f}%")
    if over_limit_frames:
        print("\nWARNING: training frames exceeding 25% ignore area limit (spec §5):")
        for clip, fname, pct in over_limit_frames:
            print(f"  {clip}/{fname}: {pct:.1f}%")

    # Table 3: Auto-labeller quality by distance band (excluding truncated & boxes inside ignore)
    print("\n=== 3. Frozen auto-labeller quality by band (non-truncated, outside ignore) ===")
    h3 = f"{'band':14}{'GT':>6}{'TP':>6}{'FN':>6}{'FP':>6}{'Recall':>9}{'Prec':>9}{'size_err':>10}"
    print(h3)
    print("-" * len(h3))
    gt_sum = tp_sum = fp_sum = se_sum = 0
    for name in bnames:
        gt, tp, fp, se = band_gt[name], band_tp[name], band_fp[name], band_size_err[name]
        fn = gt - tp
        rec = tp / gt if gt else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        serr = 100.0 * se / tp if tp else 0.0
        print(f"{name:14}{gt:>6}{tp:>6}{fn:>6}{fp:>6}{rec:>9.2f}{prec:>9.2f}{serr:>9.1f}%")
        gt_sum += gt; tp_sum += tp; fp_sum += fp; se_sum += se
    print("-" * len(h3))
    fn_sum = gt_sum - tp_sum
    rec_sum = tp_sum / gt_sum if gt_sum else 0.0
    prec_sum = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    serr_sum = 100.0 * se_sum / tp_sum if tp_sum else 0.0
    print(f"{'All bands':14}{gt_sum:>6}{tp_sum:>6}{fn_sum:>6}{fp_sum:>6}{rec_sum:>9.2f}{prec_sum:>9.2f}{serr_sum:>9.1f}%")

    # Table 4: Optional timing log summary
    log_path = Path(args.log)
    if log_path.exists():
        timed = []
        for row in read_rows(log_path):
            if row["clip"] in clips and row.get("active_min", "").strip():
                timed.append((row["clip"], float(row["active_min"])))
        if timed:
            print(f"\n=== 4. Review timing (from {log_path.name}) ===")
            h4 = f"{'clip':9}{'min':>7}{'min/fr':>8}{'sec/rev':>9}{'sec/edit':>10}"
            print(h4)
            print("-" * len(h4))
            tot_min = tot_fr_t = tot_rev_t = tot_ed_t = 0.0
            for clip, mins in timed:
                c = clip_stats[clip]
                edits = c["fix"] + c["remove"] + c["add"]
                print(f"{clip:9}{mins:>7.1f}{mins / max(c['frames'], 1):>8.2f}"
                      f"{60 * mins / max(c['rev'], 1):>9.1f}{60 * mins / max(edits, 1):>10.1f}")
                tot_min += mins; tot_fr_t += c["frames"]; tot_rev_t += c["rev"]; tot_ed_t += edits
            print("-" * len(h4))
            print(f"{'Total':9}{tot_min:>7.1f}{tot_min / max(tot_fr_t, 1):>8.2f}"
                  f"{60 * tot_min / max(tot_rev_t, 1):>9.1f}{60 * tot_min / max(tot_ed_t, 1):>10.1f}")


if __name__ == "__main__":
    main()

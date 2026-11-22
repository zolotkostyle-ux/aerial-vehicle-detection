"""Pick frames by how many vehicles in them have not been seen yet.

Neighbouring frames of a clip show the same vehicles moved by a few metres. Counting frames therefore
says little; counting distinct vehicles does. This script links the auto-labeller's boxes across the
extracted frames of each clip into tracks (camera motion removed by a background homography, vehicles
matched by position and appearance), so one physical vehicle gets one track id. It then walks each
clip in time order and keeps a frame only when at least NOVELTY of its vehicles, and at least MIN_NEW, are not in any
frame kept before, and the frame is at least MIN_GAP_S after the last kept one.

Inputs: labels/frame_manifest.csv, labels/cv_all_frames.csv (auto-labeller output, never edited).
Output: labels/frames_v1.csv and a per-clip summary on stdout.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
MANIFEST = ROOT / "labels" / "frame_manifest.csv"
CV_LABELS = ROOT / "labels" / "cv_all_frames.csv"
OUT = ROOT / "labels" / "frames_v1.csv"

VEHICLE_CLASSES = {"car", "truck", "bus"}
NOVELTY = 0.5            # share of a frame's vehicles that must be new for the frame to be kept
MIN_NEW = 2              # and at least this many new vehicles, so one re-detected car cannot keep a frame
MIN_GAP_S = 1.5          # and at least this long after the last kept frame
# Review budget: a clip with very dense traffic keeps only the frames that bring the most new vehicles.
MAX_FRAMES = {"trainE": 8}
SEARCH_LENGTHS = 4.0     # how far a vehicle may move between two extracted frames, in its own lengths
MIN_SEARCH_PX = 40
MIN_APPEARANCE = 0.45    # normalised correlation of the two vehicle patches
GAP = 2                  # a vehicle may be missed by the labeller in one frame and keep its id
PATCH = (24, 24)
REGISTER_SCALE = 0.5


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def homography(a, b, sift):
    """Maps points of frame a onto frame b, estimated on the background."""
    small = lambda im: cv2.cvtColor(cv2.resize(im, None, fx=REGISTER_SCALE, fy=REGISTER_SCALE), cv2.COLOR_BGR2GRAY)
    ka, da = sift.detectAndCompute(small(a), None)
    kb, db = sift.detectAndCompute(small(b), None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return np.eye(3)
    pairs = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
    if len(pairs) < 20:
        return np.eye(3)
    matrix, _ = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in pairs]),
                                   np.float32([kb[m.trainIdx].pt for m in pairs]), cv2.RANSAC, 3.0)
    if matrix is None:
        return np.eye(3)
    scale = np.diag([REGISTER_SCALE, REGISTER_SCALE, 1.0])
    return np.linalg.inv(scale) @ matrix @ scale


def patch(image, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = image[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
    if crop.size == 0:
        return None
    grey = cv2.cvtColor(cv2.resize(crop, PATCH), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grey -= grey.mean()
    norm = np.linalg.norm(grey)
    return grey / norm if norm > 0 else None


def match(boxes, patches, ids, earlier, matrix, taken):
    """Give unmatched boxes the track id of the best-matching box of an earlier frame."""
    e_boxes, e_patches, e_ids = earlier
    if not e_boxes:
        return
    centres = cv2.perspectiveTransform(np.float32([[[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in e_boxes]]), matrix)[0]
    candidates = []
    for i, box in enumerate(boxes):
        if ids[i] is not None or patches[i] is None:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        radius = max(MIN_SEARCH_PX, SEARCH_LENGTHS * max(box[2] - box[0], box[3] - box[1]))
        for j, (px, py) in enumerate(centres):
            if e_ids[j] in taken or e_patches[j] is None:
                continue
            distance = float(np.hypot(cx - px, cy - py))
            if distance <= radius:
                score = float((patches[i] * e_patches[j]).sum())
                if score >= MIN_APPEARANCE:
                    candidates.append((score - 0.2 * distance / radius, i, j))
    for _, i, j in sorted(candidates, reverse=True):
        if ids[i] is None and e_ids[j] not in taken:
            ids[i] = e_ids[j]
            taken.add(e_ids[j])


def track_clip(files, boxes_by_file):
    """One id per physical vehicle. A box is matched to the frame before, then to the one before that,
    so a vehicle the labeller missed in one frame keeps its id."""
    sift = cv2.SIFT_create(3000)
    next_id, tracks_by_file, history = 0, {}, []
    for name in files:
        image = cv2.imread(str(FRAMES / name.rsplit("_", 1)[0] / name))
        boxes = boxes_by_file.get(name, [])
        patches = [patch(image, b) for b in boxes]
        ids, taken = [None] * len(boxes), set()
        steps = []
        for back, (e_image, e_boxes, e_patches, e_ids) in enumerate(reversed(history[-GAP:])):
            steps.append(homography(e_image, image, sift))
            match(boxes, patches, ids, (e_boxes, e_patches, e_ids), steps[-1], taken)
        for i in range(len(ids)):
            if ids[i] is None:
                ids[i] = next_id
                next_id += 1
        tracks_by_file[name] = ids
        history.append((image, boxes, patches, ids))
    return tracks_by_file, next_id


def main():
    manifest = defaultdict(list)
    for row in read_rows(MANIFEST):
        manifest[row["clip"]].append(row)
    boxes_by_file = defaultdict(list)
    for row in read_rows(CV_LABELS):
        if row["source_class"] in VEHICLE_CLASSES:
            boxes_by_file[row["file"]].append(tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")))

    selected = []
    print(f"{'clip':9}{'frames':>7}{'boxes':>7}{'vehicles':>9}{'boxes/vehicle':>15}{'kept':>6}{'vehicles in kept':>18}")
    for clip in sorted(manifest):
        rows = sorted(manifest[clip], key=lambda r: int(r["frame_index"]))
        files = [r["file"] for r in rows]
        tracks, distinct = track_clip(files, boxes_by_file)
        seen, kept = set(), []
        for row in rows:
            ids = tracks[row["file"]]
            new = [t for t in ids if t not in seen]
            fresh = ids and len(new) / len(ids) >= NOVELTY and len(new) >= min(MIN_NEW, len(ids))
            spaced = not kept or float(row["time_s"]) - float(kept[-1]["time_s"]) >= MIN_GAP_S
            if not kept or (fresh and spaced):
                kept.append({**row, "vehicles_in_frame": len(ids), "new_vehicles": len(new)})
                seen.update(ids)
        if clip in MAX_FRAMES and len(kept) > MAX_FRAMES[clip]:
            best = sorted(kept, key=lambda r: -r["new_vehicles"])[:MAX_FRAMES[clip]]
            kept = [r for r in kept if r in best]
            seen = {t for r in kept for t in tracks[r["file"]]}
        boxes = sum(len(v) for v in tracks.values())
        print(f"{clip:9}{len(files):>7}{boxes:>7}{distinct:>9}{boxes / max(distinct, 1):>15.1f}{len(kept):>6}"
              f"{len(seen):>9} ({len(seen) / max(distinct, 1):.0%})")
        selected += kept
    if "--write" in sys.argv:
        with OUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        print(f"wrote {len(selected)} frames to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Sample frames from the clips and drop near-duplicates.

Train clips are sampled sparsely (variety matters, neighbouring frames add nothing).
The eval clip is sampled densely because time-to-first-detection needs a per-frame timeline.
Writes data/frames/<clip>/<clip>_<frame index>.jpg and labels/frame_manifest.csv.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"
FRAMES = ROOT / "data" / "frames"

SAMPLE_FPS = {"trainA": 2, "trainB": 2, "trainC": 2, "trainD": 2, "trainE": 2, "trainF": 2, "eval": 5, "eval_far": 5}
DUPLICATE_THRESHOLD = 0.3  # only frozen frames; 2.0 dropped 21 of 39 trainA frames where just the cars moved
JPEG_QUALITY = 95


def thumbnail(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (160, 160), interpolation=cv2.INTER_AREA).astype(np.float32)


def extract(clip, writer):
    capture = cv2.VideoCapture(str(VIDEOS / f"{clip}.mp4"))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {clip}.mp4")
    fps = capture.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / SAMPLE_FPS[clip]))
    out_dir = FRAMES / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = index = 0
    previous = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            thumb = thumbnail(frame)
            difference = None if previous is None else float(np.abs(thumb - previous).mean())
            is_duplicate = clip.startswith("train") and difference is not None and difference < DUPLICATE_THRESHOLD
            if is_duplicate:
                dropped += 1
            else:
                name = f"{clip}_{index:05d}.jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                height, width = frame.shape[:2]
                writer.writerow([clip, name, index, f"{index / fps:.3f}", width, height, f"{fps:.3f}",
                                 "" if difference is None else f"{difference:.2f}"])
                previous = thumb
                kept += 1
        index += 1
    capture.release()
    print(f"{clip}: {index} frames at {fps:.2f} fps, step {step}, kept {kept}, dropped as duplicates {dropped}")


def main():
    clips = sys.argv[1:] or [c for c in SAMPLE_FPS if (VIDEOS / f"{c}.mp4").exists()]
    FRAMES.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "labels" / "frame_manifest.csv"
    rows = []
    if manifest.exists():
        with manifest.open() as handle:
            rows = [r for r in csv.reader(handle)][1:]
        rows = [r for r in rows if r[0] not in clips]
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "file", "frame_index", "time_s", "width", "height", "fps", "diff_to_previous_kept"])
        writer.writerows(rows)
        for clip in clips:
            extract(clip, writer)


if __name__ == "__main__":
    main()

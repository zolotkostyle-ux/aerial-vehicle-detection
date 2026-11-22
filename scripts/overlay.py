"""Draw labels on frames for review.

usage: overlay.py <labels.csv> <out_dir> [clip ...] [--every N] [--zoom]
Boxes are coloured by distance band; --zoom adds a numbered 3x crop strip of every box.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import FRAMES, box_of, clips_in_manifest, frame_sizes, read_rows, row_band, zone_polygon

BAND_COLOUR = {"0-200 m": (60, 200, 60), "200-400 m": (0, 165, 255), "beyond 400 m": (255, 80, 200)}
IGNORE_COLOUR = (200, 60, 200)
CROP = 96


def draw(image, rows, sizes):
    canvas = image.copy()
    for number, row in enumerate(rows):
        x1, y1, x2, y2 = (int(round(v)) for v in box_of(row))
        if row.get("label") == "ignore":
            outline = np.array(zone_polygon(row), np.int32)
            shade = canvas.copy()
            cv2.fillPoly(shade, [outline], IGNORE_COLOUR)
            canvas = cv2.addWeighted(shade, 0.25, canvas, 0.75, 0)
            cv2.polylines(canvas, [outline], True, IGNORE_COLOUR, 3)
            cv2.putText(canvas, "ignore", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            continue
        colour = BAND_COLOUR[row_band(row, sizes)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3 if row.get("size_class") == "large" else 1)
        tag = str(number) + (f" {float(row['confidence']):.2f}" if row.get("confidence") else "") + (" L" if row.get("size_class") == "large" else "")
        cv2.putText(canvas, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(canvas, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return canvas


def crop_strip(image, rows, width):
    tiles = []
    for number, row in enumerate(rows):
        if row.get("label") == "ignore":
            continue
        x1, y1, x2, y2 = box_of(row)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 0.9 + 8
        xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
        xb, yb = int(min(image.shape[1], cx + half)), int(min(image.shape[0], cy + half))
        tile = cv2.resize(image[ya:yb, xa:xb], (CROP, CROP), interpolation=cv2.INTER_CUBIC)
        cv2.putText(tile, str(number), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(tile, str(number), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(tile)
    per_row = max(1, width // CROP)
    strips = []
    for start in range(0, len(tiles), per_row):
        strip = np.zeros((CROP, width, 3), np.uint8)
        for i, tile in enumerate(tiles[start:start + per_row]):
            strip[:, i * CROP:(i + 1) * CROP] = tile
        strips.append(strip)
    return np.vstack(strips) if strips else np.zeros((0, width, 3), np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("out_dir")
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--max-side", type=int, default=1920)
    args = parser.parse_args()

    by_file = defaultdict(list)
    for row in read_rows(args.labels):
        by_file[(row["clip"], row["file"])].append(row)
    sizes = frame_sizes()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = args.clips or clips_in_manifest()
    for clip in clips:
        for path in sorted((FRAMES / clip).glob("*.jpg"))[::args.every]:
            image = cv2.imread(str(path))
            rows = by_file.get((clip, path.name), [])
            canvas = draw(image, rows, sizes)
            if args.zoom:
                canvas = np.vstack([canvas, crop_strip(image, rows, canvas.shape[1])])
            scale = args.max_side / max(canvas.shape[:2])
            if scale < 1:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / path.name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()

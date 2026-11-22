"""Auto-label frames with an open-vocabulary detector (rule 2: labels start from a model).

Each frame gets one full-frame pass plus overlapping tiles that are upscaled 2x, because
far vehicles are 10-30 px long and the labeller misses them at native scale. Tiles are also
run rotated by 90 degrees: the first review showed the labeller missing small top-down
vehicles that drive along the vertical image axis.
Output: labels/cv_all_frames.csv, one row per box, in pixels of the original frame. Never edited by hand.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
RAW = ROOT / "labels" / "cv_all_frames.csv"

# YOLO-World is pretrained on Objects365, GoldG and CC3M, with no aerial dataset (rule 3).
# The first run used COCO YOLO11x and missed most top-down vehicles, see labels/labeller_check_coco_yolo11x.csv.
LABELLER = "yolov8x-worldv2.pt"
# Two-wheeler prompts are there to pull motorcycles away from "car"; QC removes those boxes.
PROMPTS = ["car", "van", "pickup truck", "truck", "bus", "motorcycle", "bicycle"]
VEHICLE_CLASSES = {0: "car", 1: "car", 2: "car", 3: "truck", 4: "bus", 5: "two_wheeler", 6: "two_wheeler"}
MIN_CONFIDENCE = 0.10
FULL_FRAME_SIZE = 1920
TILE = 480
TILE_OVERLAP = 120
TILE_INFERENCE_SIZE = 960
EDGE_MARGIN = 3  # px; a tile box touching an inner tile edge is a cut-off vehicle
MERGE_IOU = 0.5
FIELDS = ["clip", "file", "x1", "y1", "x2", "y2", "confidence", "source_class", "source_pass"]


def tile_origins(length):
    stride = TILE - TILE_OVERLAP
    origins = list(range(0, max(length - TILE, 0) + 1, stride))
    if origins[-1] + TILE < length:
        origins.append(length - TILE)
    return origins


def predict(model, image, size):
    result = model.predict(image, imgsz=size, conf=MIN_CONFIDENCE,
                           verbose=False, device="mps")[0]
    boxes = result.boxes
    return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)


def label_frame(model, image):
    height, width = image.shape[:2]
    rows = []
    xyxy, conf, cls = predict(model, image, FULL_FRAME_SIZE)
    rows += [(*b, c, k, "full") for b, c, k in zip(xyxy, conf, cls)]
    for y0 in tile_origins(height):
        for x0 in tile_origins(width):
            tile = image[y0:y0 + TILE, x0:x0 + TILE]
            found = [(b, c, k, "tile") for b, c, k in zip(*predict(model, tile, TILE_INFERENCE_SIZE))]
            rotated = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            for (u1, v1, u2, v2), c, k in zip(*predict(model, rotated, TILE_INFERENCE_SIZE)):
                found.append(((v1, tile.shape[0] - u2, v2, tile.shape[0] - u1), c, k, "tile_rotated"))
            for (x1, y1, x2, y2), c, k, source in found:
                cut_off = ((x1 < EDGE_MARGIN and x0 > 0) or (y1 < EDGE_MARGIN and y0 > 0)
                           or (x2 > TILE - EDGE_MARGIN and x0 + TILE < width)
                           or (y2 > TILE - EDGE_MARGIN and y0 + TILE < height))
                if not cut_off:
                    rows.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, c, k, source))
    if not rows:
        return []
    boxes = torch.tensor(np.array([r[:4] for r in rows], dtype=np.float32))
    scores = torch.tensor(np.array([r[4] for r in rows], dtype=np.float32))
    keep = nms(boxes, scores, MERGE_IOU).tolist()
    return [rows[i] for i in keep]


def main():
    clips = sys.argv[1:] or sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
    RAW.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if RAW.exists():
        with RAW.open() as handle:
            existing = [r for r in csv.DictReader(handle) if r["clip"] not in clips]
    model = YOLOWorld(LABELLER)
    model.set_classes(PROMPTS)
    with RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        for clip in clips:
            total = 0
            files = sorted((FRAMES / clip).glob("*.jpg"))
            for path in files:
                for x1, y1, x2, y2, c, k, source in label_frame(model, cv2.imread(str(path))):
                    writer.writerow({"clip": clip, "file": path.name, "x1": f"{x1:.1f}", "y1": f"{y1:.1f}",
                                     "x2": f"{x2:.1f}", "y2": f"{y2:.1f}", "confidence": f"{c:.3f}",
                                     "source_class": VEHICLE_CLASSES[k], "source_pass": source})
                    total += 1
            print(f"{clip}: {len(files)} frames, {total} raw boxes", flush=True)


if __name__ == "__main__":
    main()

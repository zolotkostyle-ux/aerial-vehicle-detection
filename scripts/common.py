"""Shared paths, label IO and the distance model."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "frames"
MANIFEST = ROOT / "labels" / "frame_manifest.csv"
LABELS = ROOT / "labels"
CV_LABELS = LABELS / "cv_all_frames.csv"   # auto-labeller output, never edited
FRAMES_V1 = LABELS / "frames_v1.csv"      # frames that go through review
REVIEWED = LABELS / "reviewed_v1.csv"     # boxes after the review rounds

# Distance model. Pexels clips carry no camera metadata, so the field of view is an assumption:
# 70 degrees along the long image side, typical for a consumer drone's wide camera in video mode.
ASSUMED_FOV_DEG = 70.0
# Reference length and length-to-width ratio depend on the reviewed size class, never on the
# labeller's class guess. A box with no reviewed size class is treated as a car.
REFERENCE_LENGTH_M = {"car": 4.5, "large": 12.0}
LENGTH_TO_WIDTH = {"car": 2.4, "large": 4.7}
BANDS = [("0-200 m", 0.0, 200.0), ("200-400 m", 200.0, 400.0), ("beyond 400 m", 400.0, math.inf)]


def focal_px(frame_width, frame_height):
    return (max(frame_width, frame_height) / 2) / math.tan(math.radians(ASSUMED_FOV_DEG / 2))


def vehicle_length_px(box, size_class="car"):
    """Vehicle length recovered from an axis-aligned box.

    A vehicle that sits diagonally in the frame gets a box whose long side is shorter than the
    vehicle. For a rectangle of known length-to-width ratio r turned by any angle, the box sides are
    w = W(r cos + sin) and h = W(r sin + cos), which gives the width W in closed form below.
    When the box is already more elongated than r the vehicle is taken as axis-aligned.
    """
    x1, y1, x2, y2 = box
    long_side, short_side = max(x2 - x1, y2 - y1), min(x2 - x1, y2 - y1)
    ratio = LENGTH_TO_WIDTH[size_class or "car"]
    if long_side >= ratio * short_side:
        return long_side
    width = math.sqrt(((long_side + short_side) / (ratio + 1)) ** 2 / 2 + ((long_side - short_side) / (ratio - 1)) ** 2 / 2)
    return ratio * width


def distance_m(box, frame_size, size_class="car"):
    """Slant range from the vehicle's length in pixels, seen top-down or side-on."""
    return focal_px(*frame_size) * REFERENCE_LENGTH_M[size_class or "car"] / vehicle_length_px(box, size_class)


def band_of(distance):
    for name, low, high in BANDS:
        if low <= distance < high:
            return name
    raise ValueError(distance)


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def frame_sizes():
    """(clip, file) -> (width, height), measured from the extracted frames."""
    return {(r["clip"], r["file"]): (int(r["width"]), int(r["height"])) for r in read_rows(MANIFEST)}


def clips_in_manifest():
    return sorted({r["clip"] for r in read_rows(MANIFEST)})


def box_of(row):
    return tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))


def row_band(row, sizes):
    return band_of(distance_m(box_of(row), sizes[(row["clip"], row["file"])], row.get("size_class")))


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def zone_polygon(row):
    """Corner points of an ignore zone: its polygon if it has one, otherwise its rectangle."""
    if row.get("points"):
        return [tuple(point) for point in json.loads(row["points"])]
    x1, y1, x2, y2 = box_of(row)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_area(polygon):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def point_in_polygon(x, y, polygon):
    inside = False
    for (xa, ya), (xb, yb) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            inside = not inside
    return inside

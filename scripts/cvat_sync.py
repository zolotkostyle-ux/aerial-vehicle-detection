"""Push the review frames to a local CVAT and pull the reviewed labels back.

  cvat_sync.py push   create the project and one task per clip (one review round each),
                      with the frozen auto-labels as the starting boxes
  cvat_sync.py pull   download the reviewed boxes of every task into labels/reviewed_v1.csv

The starting boxes are the auto-labeller's output and nothing else. Two attributes are pre-filled
from it as hints for the reviewer: size_class from the labeller's class guess (truck and bus give
`large`), and truncated for a box that touches the frame edge.

Credentials are read from data/cvat_credentials.txt (user on line 1, password on line 2).
"""
import csv
import json
import sys
from collections import defaultdict

from cvat_sdk import make_client, models
from cvat_sdk.core.proxies.tasks import ResourceType

from common import CV_LABELS, FRAMES, FRAMES_V1, REVIEWED, ROOT, frame_sizes, read_rows

HOST = "http://localhost:8080"
PROJECT = "aerial-vehicle-v1"
ROUND_ORDER = ["trainA", "trainB", "trainC", "trainD", "trainE", "trainF", "eval", "eval_far"]
VEHICLE_CLASSES = {"car": "car", "truck": "large", "bus": "large"}
EDGE_PX = 2
FIELDS = ["clip", "file", "label", "x1", "y1", "x2", "y2", "size_class", "truncated", "points", "source"]


def connect():
    user, password = (ROOT / "data" / "cvat_credentials.txt").read_text().split()[:2]
    return make_client(HOST, credentials=(user, password))


def attribute_ids(task):
    vehicle = next(l for l in task.get_labels() if l.name == "vehicle")
    ignore = next(l for l in task.get_labels() if l.name == "ignore")
    attributes = {a.name: a.id for a in vehicle.attributes}
    return vehicle.id, ignore.id, attributes


def push(client):
    if any(p.name == PROJECT for p in client.projects.list()):
        raise SystemExit(f"project {PROJECT} already exists; delete it in CVAT before pushing again")
    project = client.projects.create(models.ProjectWriteRequest(name=PROJECT, labels=[
        models.PatchedLabelRequest(name="vehicle", color="#ffcc00", type="any", attributes=[
            models.AttributeRequest(name="size_class", mutable=False, input_type="select",
                                    values=["car", "large"], default_value="car"),
            models.AttributeRequest(name="truncated", mutable=False, input_type="select",
                                    values=["no", "yes"], default_value="no")]),
        models.PatchedLabelRequest(name="ignore", color="#9b59b6", type="any")]))
    frames = defaultdict(list)
    for row in read_rows(FRAMES_V1):
        frames[row["clip"]].append(row["file"])
    boxes = defaultdict(list)
    for row in read_rows(CV_LABELS):
        if row["source_class"] in VEHICLE_CLASSES:
            boxes[(row["clip"], row["file"])].append(row)
    sizes = frame_sizes()
    for number, clip in enumerate([c for c in ROUND_ORDER if c in frames], 1):
        names = sorted(frames[clip])
        task = client.tasks.create_from_data(
            spec=models.TaskWriteRequest(name=f"round{number}_{clip}", project_id=project.id),
            resource_type=ResourceType.LOCAL,
            resources=[str(FRAMES / clip / name) for name in names],
            data_params={"image_quality": 95, "sorting_method": "lexicographical"})
        vehicle_id, _, attributes = attribute_ids(task)
        shapes = []
        for frame, name in enumerate(names):
            width, height = sizes[(clip, name)]
            for row in boxes.get((clip, name), []):
                x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                at_edge = x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= width - EDGE_PX or y2 >= height - EDGE_PX
                shapes.append(models.LabeledShapeRequest(
                    type="rectangle", frame=frame, label_id=vehicle_id, source="auto", points=[x1, y1, x2, y2],
                    attributes=[models.AttributeValRequest(spec_id=attributes["size_class"], value=VEHICLE_CLASSES[row["source_class"]]),
                                models.AttributeValRequest(spec_id=attributes["truncated"], value="yes" if at_edge else "no")]))
        task.set_annotations(models.LabeledDataRequest(shapes=shapes))
        print(f"round {number}: {clip}, {len(names)} frames, {len(shapes)} starting boxes  {HOST}/tasks/{task.id}")


def pull(client):
    project = next(p for p in client.projects.list() if p.name == PROJECT)
    rows = []
    for task in sorted(project.get_tasks(), key=lambda t: t.id):
        clip = task.name.split("_", 1)[1]
        vehicle_id, ignore_id, attributes = attribute_ids(task)
        names = [frame.name.split("/")[-1] for frame in task.get_frames_info()]
        shapes = task.get_annotations().shapes
        for shape in shapes:
            label = "vehicle" if shape.label_id == vehicle_id else "ignore" if shape.label_id == ignore_id else None
            kind = shape.type.value
            if (label, kind) not in (("vehicle", "rectangle"), ("ignore", "rectangle"), ("ignore", "polygon")):
                raise SystemExit(f"unexpected shape in task {task.name}: {kind} {label}")
            values = {a.spec_id: a.value for a in shape.attributes}
            xs, ys = shape.points[0::2], shape.points[1::2]
            rows.append({"clip": clip, "file": names[shape.frame], "label": label,
                         "x1": f"{min(xs):.1f}", "y1": f"{min(ys):.1f}", "x2": f"{max(xs):.1f}", "y2": f"{max(ys):.1f}",
                         "size_class": values.get(attributes["size_class"], "") if label == "vehicle" else "",
                         "truncated": values.get(attributes["truncated"], "") if label == "vehicle" else "",
                         "points": json.dumps([[round(x, 1), round(y, 1)] for x, y in zip(xs, ys)]) if kind == "polygon" else "",
                         "source": "auto" if shape.source == "auto" else "reviewer"})
        print(f"{task.name}: {len(shapes)} shapes on {len(names)} frames")
    rows.sort(key=lambda r: (r["clip"], r["file"], float(r["x1"]), float(r["y1"])))
    with REVIEWED.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} shapes to {REVIEWED.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("push", "pull"):
        raise SystemExit(__doc__)
    with connect() as cvat:
        {"push": push, "pull": pull}[sys.argv[1]](cvat)

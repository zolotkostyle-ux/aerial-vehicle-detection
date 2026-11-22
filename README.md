# Aerial vehicle detection: the dataset half of the task

Test task for the Dataset PM role at The Fourth Law. The brief is written for an ML Engineer; for this role the recruiter asked me to put the effort into the data rather than the model. So this repository builds and checks the dataset, and no detector is trained.

## Summary

- **69 frames, 3,002 reviewed vehicles**, chosen from 574 extracted frames by how many new vehicles each frame brings, not by count.
- **The auto-labeller cannot be trusted on its own.** Of its 2,443 boxes I kept 58% unchanged, fixed 595 and removed 438, and added 997 vehicles it missed. On the eval clips it finds two thirds of the vehicles at 200–400 m.
- **Review effort follows edits, not frames.** One frame took between 15 seconds and 12.5 minutes, while one edit took about 7 seconds everywhere. Plan annotation budgets per expected edit.
- **Low sun is the most expensive condition.** Two clips at the same height and scale: in diffuse light 86% of boxes were right, under low sun 35%, because boxes wrapped the shadows.
- **I added three clips** to close measured gaps: one eval clip for the 200–400 m band (rule 4) and two top-down training clips, because the brief's training clips had no top-down vehicles at that range.

AI use: the scripts were written with Claude and Gemini, and the auto-labels come from YOLO-World. Every review decision in CVAT is mine: 4 h 09 min over 8 timed rounds.

## The brief's rules

| Rule | Status |
|---|---|
| 1. Eval clip out of every decision | Kept. One exception, stated honestly: I switched labellers after the COCO model failed on the eval clip, then confirmed the choice on training clips |
| 2. Labels come from a model, corrections counted | Done, counts above and in `docs/qc_report.md` |
| 3. No ready-made aerial datasets | Kept. YOLO-World has no aerial training data |
| 4. Both distance bands need data | Done. `eval_far` added: every vehicle in the brief's eval clip is closer than 200 m |
| 5. Detector under 500,000 parameters | Not done, outside this role. No weights or demo video for the same reason |
| 6. README of 4 pages at most | This file. The full rulebook and QC tables are in `docs/` |

## Data

Eight Pexels clips, listed with links in `videos.csv`. Five come from the brief. I added:

- `eval_far` (pexels.com/video/11504182), for the 200–400 m eval band;
- `trainE` (5356780) and `trainF` (29491666), top-down interchanges. The brief's training clips show far vehicles only from an angle, and the one top-down clip, `trainD`, is close range (median car 61 px). The eval clip at range is top-down (median car 25 px), so a detector would have been tested on a view it never trained on. The new clips have cars of 24–27 px, the same size as in eval.

None of the added clips comes from the same shoot as a task clip. I dropped a third candidate because its freeway looked too much like `eval_far`.

**How frames were chosen.** Neighbouring frames show the same cars moved by a few metres, so labelling them twice costs time and adds nothing. `scripts/select_frames.py` follows each auto-labelled vehicle across frames, with camera motion removed, and keeps a frame only if at least half of its vehicles have not been seen in a frame already kept, and at least 1.5 s after the last one. `trainE` alone passed on 15 frames with about 77 cars each, so I capped it at the 8 that bring the most new vehicles.

| Clip | Split | View | Frames extracted | Distinct vehicles (est.) | Frames reviewed | Vehicles reviewed |
|---|---|---|---|---|---|---|
| trainA | train | high angle, fixed camera | 39 | 205 | 8 | 289 |
| trainB | train | low angle along the road | 63 | 169 | 12 | 123 |
| trainC | train | angled highway | 34 | 148 | 6 | 110 |
| trainD | train | top-down junction | 49 | 489 | 12 | 285 |
| trainE | train | top-down interchange | 80 | 1,308 | 8 | 1,169 |
| trainF | train | top-down interchange, parking | 44 | 363 | 6 | 284 |
| eval | eval | top-down, follows one car | 153 | 47 | 9 | 29 |
| eval_far | eval | top-down freeway | 112 | 916 | 8 | 713 |
| **Total** | | | **574** | **about 3,640** | **69** | **3,002** |

What is still missing: every clip is daylight, clear weather, paved roads. And close range in eval is thin: 0–100 m rests on 22 vehicles.

## Labels

**Rules.** `docs/annotation_spec.md` is the rulebook an annotator works from. One class, `vehicle`, as the brief asks, with two attributes. `size_class` is `car` or `large` and sets the reference length for distance, 4.5 m or 12 m; a truck and its trailer are one box. `truncated` marks a vehicle cut off by the frame edge or by a bridge, tree or another vehicle; it stays a positive example but is left out of the distance bands, because its length in pixels is not its real length. A second label, `ignore`, marks what cannot be judged, such as a road running to the horizon. Motorcycles and bicycles are background. Version 1.0 was frozen before the first round. Two rules were clarified during review and recorded as version 1.1, with the affected frames checked again.

**Auto-labeller.** COCO-trained models barely see cars from straight above: on three training clips the COCO YOLO11x found 1,064 small vehicles against 1,818 for YOLO-World, and 114 against 322 on the top-down `trainD`. So the labels come from YOLO-World (`yolov8x-worldv2`), run on the whole frame and on overlapping tiles enlarged twice, each tile also turned by 90°; the turned tiles found 29% of all boxes. Asking it for motorcycles and bicycles as well kept 107 two-wheelers from being called cars. Its output for every frame was frozen before review in `labels/cv_all_frames.csv`.

**Review.** All 69 frames were checked in CVAT, one clip per round, the eval clips last, with a stopwatch per round (`docs/review_log.csv`). The reviewed labels are in `labels/reviewed_v1.csv`.

| Starting boxes | Kept as is | Fixed | Removed | Added | Final |
|---|---|---|---|---|---|
| 2,443 | 1,410 (58%) | 595 | 438 | 997 | 3,002 |

Fixes were mostly box geometry (472), then the `truncated` flag (98) and the size class (82). Removals were mostly scooters and road markings in `trainD`, roofs and fences in `trainB`, and bridge parts in `trainE`.

## Training

Not run, as agreed for this role. The labels are ready for it: train on overlapping 640–960 px crops so that 15–30 px vehicles keep their size, mask the loss inside `ignore` zones, and keep truncated boxes as positives.

## Eval setup

Distance is worked out as the brief describes, from box size, a reference length and the camera's field of view. The clips carry no camera data, so I assumed a 70° horizontal view: a 4.5 m car is then 31 px long at 200 m and 15 px at 400 m. A car seen at an angle gets a box shorter than the car, so the length is recovered from both box sides (`scripts/common.py`). If the real lens is 55° or 84°, all distances move by +35% or −22%. The bands are therefore exact as bands of vehicle size, and approximate in metres; what matters for training is that train and eval now hold vehicles of the same size seen from the same angle.

## Results

**The dataset by band.** Truncated boxes are not in the bands.

| Split | Frames | Vehicles | 0–200 m | 200–400 m | Beyond 400 m | Review time |
|---|---|---|---|---|---|---|
| Train | 52 | 2,260 | 448 | 1,261 | 266 | 214 min |
| Eval | 17 | 742 | 112 | 572 | 9 | 35 min |
| **Total** | **69** | **3,002** | **560** | **1,833** | **275** | **249 min** |

**The brief's metrics.** There is no detector, so the table measures the auto-labeller against the reviewed eval labels. It shows how far raw model output is from ground truth, and the same scripts will score a detector once one exists.

| Metric | 0–200 m | 200–400 m |
|---|---|---|
| Detection rate | 0.74 (83 of 112) | 0.66 (377 of 572) |
| Precision | 0.80 (83 of 104) | 0.94 (377 of 399) |
| False alarms × 60 / frames | 74.1 | 77.6 |
| Time to first detection | not measured | not measured |
| mAP@0.5 (bonus); 0.65 over both bands | 0.69 | 0.64 |

**Frames with predictions and ground truth** are in `docs/examples/`: each image shows the auto-labeller's boxes on top (cyan) and the reviewed labels below, coloured by band. Two are eval frames, two are training frames where the auto-labels were worst (shadows in `trainE`, roofs and fences in `trainB`). The poor top halves are the point: they show what the review fixed.

The brief's false-alarm formula gives alarms per minute only at one frame per second; these frames are at least 1.5 s apart, so read it as alarms per frame × 60. Time to first detection needs consecutive frames and a detector. The reviewed boxes started from the labeller's own predictions, so the mAP is an upper bound. The labeller's detection rate on training clips is 0.77 and 0.66, close to eval, so the two splits are equally hard.

**What the review taught me.**

- **Budget annotation by edits.** Time per frame ranged from 0.24 to 12.5 minutes, a 52-fold spread. Time per edit stayed between 6 and 14 seconds, 7.4 on average. Choosing frames by new vehicles cut the work about eightfold, and capping `trainE` alone saved about 88 minutes.
- **Shadows cost the most.** `trainE` and `trainF` are both top-down at the same scale. Under diffuse light 86% of `trainF` boxes were right and 7 needed fixing. Under low sun only 35% of `trainE` boxes were right: 326 had wrapped the shadow, and 581 cars were missed. A box inflated by its shadow also reads as nearer and moves a 250 m car into the 0–200 m band.
- **The labeller fails on view, not on confidence.** Lowering its threshold did not bring back cars seen end-on from behind or long white bus roofs from above.

## What next

1. **Train a first labeller on these 2,260 reviewed boxes** and use it for the next batch. Prompt tuning has hit its limit, and a third of far vehicles are still missed.
2. **Close the gaps in the data**: dusk, overcast, wet and unpaved roads; a denser close-range eval clip; and, once a detector exists, a stretch of consecutive eval frames for time to first detection.
3. **Agree three rules with the ML team**: whether a standalone trailer is a vehicle; whether to split `large` into delivery trucks and articulated lorries, or take distance from the drone's altitude and camera angle instead; and when oriented boxes become worth their cost.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_videos.py && python scripts/extract_frames.py
python scripts/autolabel.py && python scripts/select_frames.py --write   # about 35 min
python scripts/qc_compare.py && python scripts/map_eval.py && python scripts/stats.py
```

The auto-labels, frame list and reviewed labels are stored in `labels/`, so the last line runs on its own. `scripts/cvat_sync.py` moves labels to and from a local CVAT.

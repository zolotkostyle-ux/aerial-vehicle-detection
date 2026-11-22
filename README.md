# Aerial vehicle detection: the dataset half of the task

Test task for the Dataset PM role (thefourthlaw.ai). The brief is the ML Engineer task; for this role the recruiter asked to put the effort into the data rather than the model, so this repository covers clip diagnostics, frame selection, annotation spec design, model-assisted pre-labelling, full human review with stopwatch timing, and range-band QC. No sub-500k detector is trained, and this README says so wherever the brief asks for one.

**Scope against the six rules.** Rules 1–4 and 6 are kept in full: all pipeline and spec choices were made on training clips before touching eval (rule 1); labels start from an open-vocabulary detector with every human edit counted (rule 2); no aerial datasets or weights are used (rule 3); `eval_far`, `trainE` and `trainF` were added so both 0–200 m and 200–400 m hold top-down data (rule 4); full spec and QC tables live in `docs/annotation_spec.md` and `docs/qc_report.md` so this README stays within 4 pages (rule 6). Rule 5 (sub-500k detector, weights, demo video) is out of scope for Dataset PM; side-by-side auto-label vs ground-truth eval frames are in `docs/examples/`. Scripts and QC tooling were written with Claude and Gemini; auto-labels come from `yolov8x-worldv2`; every review decision in CVAT (4 h 09 m across 8 timed rounds) is mine.

## Data

Eight Pexels clips (`videos.csv`): five from the brief, `eval_far` (pexels.com/video/11504182) added under rule 4 because every non-truncated vehicle in `eval` sits in 0–200 m, and `trainE` (5356780) plus `trainF` (29491666) added for top-down 200–400 m training. Neither comes from the same shoot as any task clip, and a third candidate was dropped because its freeway looked too close to `eval_far`.

**Frames are chosen by new vehicles, not by count.** Neighbouring frames show the same cars moved by a few metres. `scripts/select_frames.py` removes camera motion via background SIFT homography, links auto-labelled boxes into tracks, and keeps a frame only when at least 50% of its vehicles (and at least 2) are unseen in earlier kept frames, spaced at least 1.5 s apart (`trainE` capped from 15 to the 8 highest-novelty frames).

| Clip | Split | View | Extracted | Est. tracks | Kept | Reviewed vehicles |
|---|---|---|---|---|---|---|
| trainA | train | high oblique, fixed | 39 | 205 | 8 | 289 |
| trainB | train | low oblique, road | 63 | 169 | 12 | 123 (+8 `ignore`) |
| trainC | train | oblique highway | 34 | 148 | 6 | 110 |
| trainD | train | top-down junction | 49 | 489 | 12 | 285 |
| trainE | train | top-down interchange | 80 | 1,308 | 8 (cap) | 1,169 |
| trainF | train | top-down highway/lots | 44 | 363 | 6 | 284 |
| eval | eval | top-down, follows 1 car | 153 | 47 | 9 | 29 |
| eval_far | eval | top-down freeway | 112 | 916 | 8 | 713 |
| **Total** | | | **574** | **~3,640** | **69** | **3,002** |

- **Domain gap closed.** The brief's four training clips had no top-down 200–400 m vehicles (`trainA–C` are oblique; `trainD` is close-range, median 61 px). `trainE` and `trainF` add 1,064 reviewed top-down vehicles in 200–400 m (median 24–27 px).
- **Remaining gaps.** All eight clips are clear daylight over paved roads. In eval, 0–200 m holds 112 vehicles, but 0–100 m rests on 22 boxes from `eval`.

## Labels

**Spec (`docs/annotation_spec.md`, v1.0, frozen before round 1).** One class, `vehicle`, with `size_class` (`car` = 4.5 m up to ~6 m; `large` = 12 m above ~6 m, whole truck+trailer in one box) and `truncated` (`yes` when the frame border or an obstacle clips vehicle length while the visible part is >= 8 px and unmistakably a vehicle — kept as a positive for early capture, excluded from range bands). `ignore` masks `< 8 px` horizon roads, unrecognizable slivers and heavy tree cover (<= 25% of any train frame). Two-wheelers and arrow-boards are background, never `ignore`.

**Auto-labeller (`scripts/autolabel.py`).** On training clips `trainA`, `trainB` and `trainD`, COCO `yolo11x` found 1,064 boxes <= 31 px versus 1,818 for `yolov8x-worldv2` (+71%, and 114 vs 322 on top-down `trainD`; `labels/labeller_check_coco_yolo11x.csv`). Each frame runs through `yolov8x-worldv2` at 1920 px plus 480 px overlapping tiles upscaled 2x to 960 px and rotated 90° (`tile_rotated` supplied 29% of baseline boxes overall and 41–59% on `trainE`/`trainF`). Decoy prompts `motorcycle`/`bicycle` diverted 107 two-wheelers (96 in `trainD`) before CVAT import. Frozen baseline: `labels/cv_all_frames.csv` (`b5489a977d3142d4`).

**Human review (`docs/qc_report.md`, `docs/review_log.csv`).** All 69 frames (`labels/frames_v1.csv`, `9be18d2b91d52e08`) were reviewed in CVAT in 8 timed rounds (248.8 min = 4 h 09 m active time, zero mid-round spec changes). Starting from 2,443 auto-boxes, review produced **3,002** `vehicle` boxes + 8 `ignore` zones (`labels/reviewed_v1.csv`, `30ec8c840d99678c`): **1,410 Keep** (57.7%), **595 Fix** (472 box geometry, 82 `size_class`, 98 `truncated`), **438 Remove**, and **997 Add**.

## Training

Not run for this role. For a sub-500k airborne detector (rule 5), this export trains on overlapping 640–960 px crops (preserving 15–30 px targets at 200–400 m), masking loss inside `ignore` polygons and keeping `truncated: yes` boxes as positive targets.

## Eval setup

Range is `focal_px * reference_length_m / vehicle_length_px`, assuming a 70° horizontal FOV (`focal = 1,371 px` on 1920 px frames: a 4.5 m car is 30.8 px at 200 m and 15.4 px at 400 m). `vehicle_length_px` recovers length from both box sides via class aspect ratios (2.4 `car`, 4.7 `large`) so angled vehicles do not read 10–30% too far (`scripts/common.py`). True FOVs of 55° or 84° shift range by +35% or -22%, so bands act first as consistent angular-size bands. Sparse sampling (>= 1.5 s) maximises distinct vehicles per review hour; sub-second time-to-first-detection requires a contiguous 5 fps eval segment once a detector exists.

## Results

**1. Reviewed dataset by split and range band** (334 `truncated: yes` boxes excluded from range bands):

| Split | Frames | Vehicles | 0–200 m | 200–400 m | > 400 m | `large` | `truncated` | Review time |
|---|---|---|---|---|---|---|---|---|
| Train (`trainA–F`) | 52 | 2,260 | 448 | 1,261 | 266 | 216 | 285 | 214.0 min |
| Eval (`eval + eval_far`) | 17 | 742 | 112 | 572 | 9 | 38 | 49 | 34.8 min |
| **Total** | **69** | **3,002** | **560** | **1,833** | **275** | **254** | **334** | **248.8 min** |

**2. Zero-shot pre-labeller evaluated on held-out `eval + eval_far`** (17 frames, 693 non-truncated GT vehicles; `docs/examples/`):

| Metric (unreviewed auto-labeller vs reviewed eval GT) | 0–200 m | 200–400 m |
|---|---|---|
| Detection rate `TP / (TP + FN)` | 0.74 (83 / 112) | 0.66 (377 / 572) |
| Precision `TP / (TP + FP)` | 0.80 (83 / 104) | 0.94 (377 / 399) |
| False alarms / min `FP * 60 / N_frames` | 74.1 (21 FP) | 77.6 (22 FP) |
| Time to first detection | n/a (sparse >= 1.5 s frames; no sub-500k model) | n/a |

**3. Operational findings:** (a) **Budget by object edits, not frames:** effort spanned 52x per frame (0.24 min/frame on `eval` vs 12.53 on `trainE`), while per-edit speed stayed steady (5.6–13.8 s, 7.4 s overall); novelty tracking (574 -> 69 frames) and capping `trainE` (saving ~88 min) cut cost ~8x. (b) **Shadows distort both boxes and range:** at the same 250 m top-down scale, diffuse-light `trainF` kept 86.3% of boxes (7 box fixes) while low-sun `trainE` kept 35.4% (326 shadow-bloated box fixes that otherwise push 250 m cars into 0–200 m). (c) **Matched train/eval difficulty:** pre-labeller recall is 0.77 / 0.66 on train and 0.74 / 0.66 on eval.

## What next

1. **Bootstrap labeller v2:** train an in-domain detector on these 2,260 reviewed train boxes to eliminate the 34% miss rate on foreshortened rear views (`trainA`) and white bus roofs (`trainE`) plus shadow padding before scaling to thousands of frames.
2. **Close data & eval gaps:** add dusk, overcast, wet and unpaved footage; add a denser 0–100 m eval clip; label a contiguous 5 fps window once a sub-500k model exists to measure time-to-first-detection.
3. **Confirm with ML team:** standalone trailers as `vehicle`; splitting `large` (7 m delivery/bobtail vs 16.5 m lorry, or using aircraft altitude + gimbal telemetry for range); oriented boxes if diagonal traffic overlap hurts NMS.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_videos.py && python scripts/extract_frames.py
python scripts/autolabel.py && python scripts/select_frames.py --write
python scripts/cvat_sync.py pull && python scripts/qc_compare.py && python scripts/stats.py
```

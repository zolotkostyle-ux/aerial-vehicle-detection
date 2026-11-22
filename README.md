# Aerial vehicle detection: the dataset half of the task

DRAFT. Sections marked "after review" are filled in when the review rounds are done.

Test task for the Dataset PM role. The brief is the ML Engineer task; for this role the recruiter asked to put the effort into the data rather than the model, so this repository covers frames, labels, label checking and range bands. No detector is trained, and the README says so wherever the brief asks for one.

## Scope

| Brief | Status |
|---|---|
| Pull frames, label with a model, check and fix labels | In progress |
| Rule 1, eval clip out of every decision | Kept |
| Rule 2, labels come from a model, corrections counted | In progress |
| Rule 3, no ready-made aerial datasets | Kept |
| Rule 4, both distance bands need data | One eval clip added, see "Data" |
| Clips | Two training clips added to close a measured gap, see "Data" |
| Rule 5, detector under 500,000 parameters | Not done, no detector |
| Rule 6, README of 4 pages | This file; the labelling rules are in `docs/annotation_spec.md` |
| Weights, eval frames with predictions, demo video | Not included, they need a detector |

AI use: the scripts were written with Claude. The auto-labels come from an open-vocabulary detector. All review decisions are mine.

## Data

Eight Pexels clips, listed in `videos.csv`. Five are from the brief. `eval_far` (pexels.com/video/11504182) was added under rule 4: every vehicle in the brief's eval clip is in the 0–200 m band. `trainE` (pexels.com/video/5356780) and `trainF` (pexels.com/video/29491666) were added for training, for the reason in the list below. Neither is from the same shoot as any task clip, and a third candidate was dropped because its freeway looked too much like the `eval_far` scene.

**Frames are chosen by new vehicles, not by count.** Neighbouring frames of a clip show the same vehicles moved by a few metres, so labelling them again adds work and no information. `scripts/select_frames.py` links the auto-labeller's boxes across all extracted frames into tracks, one per vehicle, and keeps a frame only when at least half of its vehicles, and at least two, have not appeared in a kept frame before, and it is at least 1.5 s after the last kept frame.

| Clip | View | Extracted frames | Vehicles (estimate) | Frames kept |
|---|---|---|---|---|
| trainA | high oblique, fixed camera | 39 | 205 | 8 |
| trainB | low oblique, along the road | 63 | 169 | 12 |
| trainC | low oblique, along the road | 34 | 148 | 6 |
| trainD | top-down, junction | 49 | 489 | 12 |
| eval | top-down, follows one car | 153 | 47 | 9 |
| eval_far | top-down, turning | 112 | 916 | 8 |
| trainE | top-down, multi-level interchange | 80 | 1,308 | 8, capped |
| trainF | top-down, interchange and parking lots | 44 | 363 | 6 |
| Total | | 574 | about 3,640 | 69 |

The vehicle counts are an upper bound: a vehicle the labeller misses in two frames in a row starts a new track. trainE passes the rule on 15 frames with about 77 vehicles each; it is capped at the 8 frames that bring the most new vehicles, to keep the review effort in proportion.

What the numbers said about the brief's material, and what was done:

- **No top-down 200–400 m footage in the brief's training clips.** In the four training clips from the brief, vehicles at 200–400 m appear only in oblique views (trainA, B, C); the single top-down training clip, trainD, is close-range (median car 51 px, 0–200 m). A drone detector evaluated from straight above at 200–400 m would have had no top-down training data in that band. `trainE` and `trainF` close that gap with top-down footage in the 200–400 m band (median car 26 px).
- **One condition only.** Daylight, clear weather, paved roads, civilian traffic. The added clips do not change that, and more frames from the same videos would add copies, not new situations.
- **The brief's eval clip is thin.** About 47 vehicles in 30 seconds, because the camera follows one car.

Distances come from pixel size and an assumed lens, so the bands are size bands first. A car 26 px long is 255 m away with a 70° lens, 198 m with 84° and 342 m with 55°. What is certain is that training and eval now hold vehicles of the same pixel size, seen from the same angle.

## Labels

**Rules.** `docs/annotation_spec.md`, version 1.0, frozen before review. One class, `vehicle`, as in the brief, with two attributes: `size_class` (`car` / `large`, the reference length for distance, as DOTA splits small and large vehicles) and `truncated` (length clipped by the frame edge or cover, left out of the distance bands). An `ignore` label marks what cannot be judged, as VisDrone's ignored regions. Motorcycles and bicycles are background.

**Auto-labeller.** COCO-pretrained YOLO11x misses far and small top-down vehicles on the training clips: across `trainA`, `trainB` and `trainD` it found 1,064 boxes of ≤31 px (200+ m) versus 1,818 for YOLO-World (+71%, including 114 vs 322 on top-down `trainD`; `labels/labeller_check_coco_yolo11x.csv`). I use YOLO-World (`yolov8x-worldv2`), an open-vocabulary detector with no aerial training data, run on the whole frame and on tiles upscaled twice, each tile also rotated by 90°. Its output for all 574 extracted frames is `labels/cv_all_frames.csv` and is never edited. Before review the baseline on the 69 review frames (2,443 boxes) was recorded with file fingerprints in `docs/qc_report.md`, section 1.

**Review.** In rounds, one clip per round, train clips first. Rules change only between rounds, through the change log in the spec. After the last round the reviewed labels are compared with the frozen auto-labels.

Results: after review.

## Eval setup

Distance: `focal length in px × vehicle length in m / vehicle length in px`, with an assumed 70° field of view, 4.5 m for `car` and 12 m for `large`. The pixel length is recovered from both box sides, so a vehicle at an angle is not read as shorter than it is. The limits are listed in the spec, section 6.

Review frames are spaced at least 1.5 s apart (9 frames on `eval`, 8 on `eval_far`) to maximise distinct vehicles within the review budget. This supports detection rate, precision and false-alarm estimates across both range bands; measuring sub-second time-to-first-detection requires labelling a dense 5 fps sequence and is left as a next step.

## Results

After review.

## What next

After review.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_videos.py
python scripts/extract_frames.py
python scripts/autolabel.py
python scripts/select_frames.py --write
```

`autolabel.py` needs about 35 minutes on an Apple M-series machine. The frame list and the auto-labels are stored in `labels/`, so the steps after them can run without it.

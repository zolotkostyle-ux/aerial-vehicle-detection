# QC report

Section 1 was written before the human review started and is not edited afterwards. Section 2 is filled in from the review rounds.

## 1. Auto-label baseline, frozen before review

Recorded on 20 November 2026, before round 1.

**What was frozen.**

| File | What it holds | SHA-256, first 16 characters |
|---|---|---|
| `labels/cv_all_frames.csv` | YOLO-World output for all 574 extracted frames | `b5489a977d3142d4` |
| `labels/frames_v1.csv` | the 69 frames that go through review | `9be18d2b91d52e08` |

The review starts from these boxes and nothing else: no rule pass, no LLM pass, no boxes carried over from other frames. The comparison in section 2 is between this baseline and the reviewed export.

**Settings.** `yolov8x-worldv2`, prompts car, van, pickup truck, truck, bus, motorcycle, bicycle; the last two are dropped. Confidence 0.10. Each frame is run whole at 1920 px and as 480 px tiles upscaled to 960 px with 120 px overlap, each tile also rotated by 90°. Boxes from all passes are merged by non-maximum suppression at IoU 0.5.

**Hints pre-filled for the reviewer.** `size_class` is `large` when the labeller called the box a truck or bus. `truncated` is `yes` when the box touches the frame edge (within 2 px). Both are hints and are checked like everything else.

**The baseline on the review frames.** Bands are computed from the box size with the assumptions in `annotation_spec.md`, section 6. Boxes at the frame edge are left out of the bands.

| Clip | Frames | Boxes | 0–200 m | 200–400 m | beyond 400 m | `large` | At frame edge |
|---|---|---|---|---|---|---|---|
| trainA | 8 | 231 | 9 | 123 | 94 | 12 | 5 |
| trainB | 12 | 168 | 50 | 82 | 28 | 34 | 8 |
| trainC | 6 | 136 | 79 | 34 | 1 | 41 | 22 |
| trainD | 12 | 391 | 273 | 63 | 7 | 41 | 48 |
| trainE | 8 | 658 | 116 | 475 | 36 | 41 | 31 |
| trainF | 6 | 292 | 29 | 238 | 10 | 9 | 15 |
| eval | 9 | 31 | 21 | 2 | 0 | 3 | 8 |
| eval_far | 8 | 536 | 94 | 387 | 21 | 46 | 34 |
| Total | 69 | 2,443 | 671 | 1,404 | 197 | 227 | 171 |

These are the labeller's boxes, not vehicles: some are not vehicles, and vehicles it missed are not here. How far off they are is what the review measures.

## 2. Review results

Completed on 22 November 2026 across 8 rounds (one clip per round, `trainA`–`trainF` first, `eval` and `eval_far` last) starting from `docs/annotation_spec.md` v1.0. Two rules were clarified during the review and recorded as spec v1.1 (spec, section 9): `truncated` also covers vehicles cut by an obstacle, and a tractor without a trailer is `car`. trainC and trainD were re-checked after their rounds, and every `large` box after round 8.

| File | What it holds | SHA-256, first 16 characters |
|---|---|---|
| `labels/reviewed_v1.csv` | 3,010 reviewed shapes on the 69 frames (`3,002` `vehicle`, `8` `ignore`) | `30ec8c840d99678c` |
| `docs/review_log.csv` | per-round active stopwatch timing (`248.8` min / `4:08:46` total) | — |

Reproduce every table below with `python scripts/qc_compare.py` and `python scripts/stats.py`.

### 2.1 Reviewer decisions against the frozen auto-labels

Matched 1-to-1 per frame at `IoU >= 0.30`. A matched pair is **Keep** when all four coordinates stay within `max(2 px, 10% of long side)` (spec section 3) and both `size_class` and `truncated` match; otherwise it is **Fix** (a single box may change more than one field, shown in `box / size / trunc`).

| Clip | Frames | Auto | Reviewed | Keep | Kept % | Fix (`box/sz/tr`) | Remove (`in ignore`) | Add |
|---|---|---|---|---|---|---|---|---|
| trainA | 8 | 231 | 289 | 148 | 64.1% | 57 (`54/13/3`) | 26 (`0`) | 84 |
| trainB | 12 | 168 | 123 | 67 | 39.9% | 36 (`6/1/33`) | 65 (`2`) | 20 |
| trainC | 6 | 136 | 110 | 90 | 66.2% | 15 (`8/8/5`) | 31 (`0`) | 5 |
| trainD | 12 | 391 | 285 | 164 | 41.9% | 73 (`54/11/11`) | 154 (`0`) | 48 |
| trainE | 8 | 658 | 1,169 | 233 | 35.4% | 355 (`326/12/42`) | 70 (`0`) | 581 |
| trainF | 6 | 292 | 284 | 252 | 86.3% | 16 (`7/6/3`) | 24 (`0`) | 16 |
| **Train subtotal** | **52** | **1,876** | **2,260** | **954** | **50.9%** | **552 (`455/51/97`)** | **370 (`2`)** | **754** |
| eval | 9 | 31 | 29 | 27 | 87.1% | 0 (`0/0/0`) | 4 (`0`) | 2 |
| eval_far | 8 | 536 | 713 | 429 | 80.0% | 43 (`17/31/1`) | 64 (`0`) | 241 |
| **Eval subtotal** | **17** | **567** | **742** | **456** | **80.4%** | **43 (`17/31/1`)** | **68 (`0`)** | **243** |
| **Total** | **69** | **2,443** | **3,002** | **1,410** | **57.7%** | **595 (`472/82/98`)** | **438 (`2`)** | **997** |

### 2.2 Reviewed dataset coverage and `ignore` share

Boxes with `truncated: yes` (length clipped by the frame border, an overpass, a tree or another vehicle) are positive detection examples and are excluded from the distance bands (spec sections 2 and 6). Every training frame stayed well under the 25% `ignore` area limit (spec section 5; peak clip share `0.1%` on `trainB`).

| Clip | Frames | Vehicles | 0–200 m | 200–400 m | beyond 400 m | `large` | `truncated` | `ignore` zones | `ignore` area |
|---|---|---|---|---|---|---|---|---|---|
| trainA | 8 | 289 | 0 | 106 | 178 | 29 | 5 | 0 | 0.0% |
| trainB | 12 | 123 | 10 | 41 | 22 | 10 | 50 | 8 | 0.1% |
| trainC | 6 | 110 | 46 | 36 | 2 | 40 | 26 | 0 | 0.0% |
| trainD | 12 | 285 | 225 | 14 | 0 | 20 | 46 | 0 | 0.0% |
| trainE | 8 | 1,169 | 137 | 827 | 64 | 115 | 141 | 0 | 0.0% |
| trainF | 6 | 284 | 30 | 237 | 0 | 2 | 17 | 0 | 0.0% |
| **Train subtotal** | **52** | **2,260** | **448** | **1,261** | **266** | **216** | **285** | **8** | **0.0%** |
| eval | 9 | 29 | 21 | 0 | 0 | 5 | 8 | 0 | 0.0% |
| eval_far | 8 | 713 | 91 | 572 | 9 | 33 | 41 | 0 | 0.0% |
| **Eval subtotal** | **17** | **742** | **112** | **572** | **9** | **38** | **49** | **0** | **0.0%** |
| **Total** | **69** | **3,002** | **560** | **1,833** | **275** | **254** | **334** | **8** | **0.0%** |

Within `0–200 m`, training holds `14` vehicles at `0–50 m`, `119` at `50–100 m`, `128` at `100–150 m` and `187` at `150–200 m`. Eval holds `16` at `0–50 m`, `6` at `50–100 m`, `11` at `100–150 m` and `79` at `150–200 m`: `0–200 m` is covered (~112 vehicles), though `0–100 m` rests on the 21 non-truncated boxes of the brief's `eval` clip.

### 2.3 Frozen auto-labeller quality by distance band

Evaluated on non-truncated boxes outside `ignore` zones (`2,668` reviewed vehicles total: `1,975` train, `693` eval). Notice that auto-labeller recall matches between train (`0.77` / `0.66`) and eval (`0.74` / `0.66`) once `trainE` and `trainF` are included.

| Split / Band | Reviewed GT | TP | FN (missed) | FP (removed) | Detection rate (Recall) | Precision | `size_class` hint error |
|---|---|---|---|---|---|---|---|
| **All 69 frames · 0–200 m** | 560 | 428 | 132 | 167 | 0.76 | 0.72 | 4.2% |
| **All 69 frames · 200–400 m** | 1,833 | 1,214 | 619 | 155 | 0.66 | 0.89 | 2.7% |
| **All 69 frames · beyond 400 m** | 275 | 151 | 124 | 71 | 0.55 | 0.68 | 10.6% |
| **All 69 frames · total** | **2,668** | **1,793** | **875** | **393** | **0.67** | **0.82** | **3.7%** |
| Held-out eval (`17` frames) · 0–200 m | 112 | 83 | 29 | 21 | 0.74 | 0.80 | 2.4% |
| Held-out eval (`17` frames) · 200–400 m | 572 | 377 | 195 | 22 | 0.66 | 0.94 | 7.2% |

### 2.3b mAP@0.5 of the frozen auto-labeller

COCO-style 101-point AP at IoU 0.5, single class, scored by the labeller's confidence (`scripts/map_eval.py`). Truncated ground truth and predictions inside `ignore` zones are left out; an unmatched prediction counts against a band only if its own box falls in that band. The reviewed boxes started from these predictions, so this is an upper bound for the labeller.

| Range | Held-out eval (17 frames): GT | AP@0.5 | Train (52 frames): GT | AP@0.5 |
|---|---|---|---|---|
| 0–200 m | 112 | 0.694 | 448 | 0.669 |
| 200–400 m | 572 | 0.643 | 1,261 | 0.612 |
| Both bands | 684 | 0.647 | 1,709 | 0.626 |

### 2.4 Review timing and what drove the cost

| Round | Clip | Frames | Active min | Min / frame | Sec / final vehicle | Sec / edit (`Fix+Rem+Add`) |
|---|---|---|---|---|---|---|
| 1 | trainA | 8 | 31.8 | 3.97 | 6.6 | 11.4 |
| 2 | trainB | 12 | 27.8 | 2.31 | 13.5 | 13.8 |
| 3 | trainC | 6 | 10.7 | 1.78 | 5.8 | 12.6 |
| 4 | trainD | 12 | 32.2 | 2.69 | 6.8 | 7.0 |
| 5 | trainE | 8 | 100.2 | 12.53 | 5.1 | 6.0 |
| 6 | trainF | 6 | 11.3 | 1.88 | 2.4 | 12.1 |
| 7 | eval | 9 | 2.2 | 0.24 | 4.5 | 21.8 |
| 8 | eval_far | 8 | 32.6 | 4.08 | 2.7 | 5.6 |
| **Total** | | **69** | **248.8** | **3.61** | **5.0** | **7.4** |

Four findings from the rounds:

1. **Per-frame budgeting fails (`0.24` to `12.53` min/frame, 52× range); per-edit time is stable (`6–14` s).** Effort tracks `vehicles × labeller error rate`, not frame count. Track-based novelty sampling (`574 -> 69` frames) cut review volume ~8×, and capping `trainE` from 15 frames to 8 saved `~88` minutes on its own while still yielding `827` vehicles in `200–400 m`.
2. **Hard low-sun shadows dominate geometry fixes (`trainE` vs `trainF`).** Both clips are top-down interchanges at `250 m` (`median 24–27 px`), yet `trainF` under diffuse light kept `86.3%` of auto-boxes (`7` box fixes, `1.88` min/frame), whereas `trainE` under low sun kept `35.4%` (`326` boxes had to be tightened off cast shadows — 55% of matched boxes — plus `581` missed vehicles, `12.53` min/frame). Left unfixed, shadow-bloated `26 px` boxes read as `>33 px` and jump from `200–400 m` into `0–200 m`.
3. **Zero-shot blind spots follow viewing geometry, not confidence.** Lowering confidence to `0.01` or adding aerial prompts did not recover foreshortened `14x8 px` rear views on `trainA`'s vertical overpass or long white bus/lorry roofs on `trainE`. Under rule 3 (no aerial pretraining), the right remedy for batch 2 is bootstrapping a first-stage detector on these `2,260` reviewed training boxes rather than tuning prompts further.
4. **Class-hint traps (`pickup` and `bobtail` tractors).** YOLO-World labels US pickups, SUVs and rural sheds as `truck`, causing `82` `size_class` fixes plus `65` false-positive removals in `trainB`. Standalone semi-tractors without trailers (`bobtail`, ~7 m, `~30 px` in `trainF`) must also take `car` (`4.5 m`) rather than `large` (`12 m`, calibrated for full rigs) so they stay in `200–400 m` instead of shifting past `500 m`. Post-round visual checks on all `large` boxes left residual `size_class` disagreement below `0.5%` (spec section 8 threshold: `<= 1%`).

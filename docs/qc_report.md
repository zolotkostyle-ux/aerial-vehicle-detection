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

After the review rounds.

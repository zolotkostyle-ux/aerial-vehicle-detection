# Annotation spec: aerial vehicle detection

Version 1.0, frozen on 20 November 2026. Applies to every frame that goes through review. It changes only through the change log in section 9.

The detector answers one question: is there a vehicle here, and how far away is it. Every rule below follows from that. When a case is not covered, do not guess: put a small `ignore` box on the object, add the case to the questions list, and carry on. Questions are decided between review rounds, not during one.

The scheme follows the two main public aerial datasets: DOTA splits vehicles into small and large, VisDrone keeps motorcycles and bicycles apart from cars, marks regions that are not evaluated, and flags truncated boxes.

## 1. What is a `vehicle`

One class, `vehicle`.

| Label it | Do not label it |
|---|---|
| Cars, SUVs, vans, pickups | Motorcycles, scooters, bicycles and their riders |
| Trucks, with the trailer inside the same box | Pedestrians |
| Buses, minibuses | Arrow boards, road-work signs and other roadside equipment on wheels |
| Cargo trailers and semi-trailers standing alone | Containers, sheds, boats, roofs, road markings |
| Parked vehicles, the same as moving ones | Trains, trams |
| Tow trucks and construction trucks on wheels | |
| Motorised three-wheelers with a cabin or a cargo bed (auto-rickshaw, cargo trike), as `car` | |

Why two-wheelers are out. At 200 m a motorcycle is 5 to 8 px long, below the size at which a reviewer can tell it from a shadow or a road mark, so those labels would be noise. The distance estimate also assumes a car-sized object. If two-wheelers become a target, they get their own class and their own reference length, not a place inside `vehicle`.

Why three-wheelers with a cabin or cargo bed are in. They are 2.5 to 3.5 m long and read as a small vehicle from above, unlike a motorcycle.

Why standalone trailers are in. From above, a parked semi-trailer looks the same as a truck whose cab is hidden or too small to see, so a rule that excludes it cannot be applied the same way by two reviewers. A cargo trailer is also a transport asset in its own right. The brief does not cover trailers, so this is our call and it is listed as a question for the ML team.

Why parked vehicles are in. One frame cannot tell a parked car from a moving one, and a detector that has to ignore some cars learns to ignore cars.

## 2. Attributes

Each box carries two attributes, `size_class` and `truncated`.

| Value | Meaning | Reference length for distance |
|---|---|---|
| `car` | up to about 6 m: car, SUV, van, pickup | 4.5 m |
| `large` | longer than about 6 m: truck, bus, coach | 12 m |

Pickups, vans and work trucks built on a pickup stay `car`. In oblique footage near vehicles look bigger than far ones, so decide by the type of vehicle (separate cab and body, semi-trailer, bus), not by pixels.

`truncated` is `yes` when the frame edge or an obstacle (bridge, tree, building, another vehicle) cuts off part of the vehicle so its full length is not in view, while the visible part is at least 8 px on its long side and clearly identifiable as a vehicle from that frame alone. Such a box is a normal positive for detection (preserving early capture as a vehicle enters the frame or emerges from cover), and it is left out of the distance bands because its pixel length is shorter than the vehicle. Every box whose full length is visible is `no`.

`size_class` is set by a reviewer. The auto-labeller's own class guess is only a hint and is never used for distance without review, because a car read as a truck jumps from the 0–200 m band to the 200–400 m band.

## 3. How to draw the box

1. Axis-aligned, tight around the visible vehicle body. Tolerance is 2 px or 10% of the box's long side, whichever is larger.
2. The shadow stays outside the box. In low sun the shadow can be longer than the car. Find the body by its colour and by the roof and windscreen, and box only that.
3. One vehicle, one box. In a queue the boxes may overlap. A box around several vehicles is always wrong.
4. A truck and its trailer are one box. So are a car and the trailer it tows: `large` if the pair is about one and a half car lengths or more, otherwise `car`. A car on a tow truck's bed is part of the tow truck.
5. Mirrors, antennas and open doors are ignored.

Why a truck and its trailer are one box. At 200–400 m the whole rig is 40 to 60 px long and the cab is 8 to 10 px, so two reviewers would cut the cab from the trailer in different places. The 12 m reference length for `large` also assumes the whole rig. A cab boxed alone would be read as a car and land in the wrong range band.

Known limits of this rule:

- A rig in a turn. An axis-aligned box around a bent rig is loose, and the length recovered from it (section 6) is less reliable than for a straight vehicle.
- A trailer standing alone is labelled as a vehicle (section 1), with `size_class` by its length. This is a judgement call and should be confirmed with the ML team.

Open question for the ML team: does any downstream task need part-level labels (cab, trailer) at close range? If so, add them as a separate attribute or keypoint, only above a minimum pixel size, and keep the whole-vehicle box unchanged.

## 4. Small, hidden and cut-off vehicles

| Case | Rule |
|---|---|
| Long side under 8 px | Do not label. It cannot be reviewed reliably. |
| Long side 8 to 15 px (beyond about 400 m) | Box it if it is clearly a vehicle: car-shaped and on a lane or a parking space. A small `ignore` if not sure. The 8 px applies to the whole vehicle, at any angle. |
| Partly hidden by a tree, bridge or another vehicle | If the visible part is at least 8 px and unmistakably a vehicle on its own, box the visible part (`truncated: yes` if its full length is clipped, `no` if the full length is visible). Otherwise put a small `ignore` box on the visible sliver. |
| Cut by the frame edge | Same recognizability rule: if the visible part is at least 8 px and unmistakably a vehicle on its own (for example a truck cab or the front end of a car), box it with `truncated: yes`. If only an ambiguous sliver (under 8 px or an unrecognizable corner/bumper) is in view, put a small `ignore` box on it. |
| Fully hidden for a few frames, then back | Do not invent a box for the hidden frames. |
| Motion blur | Box the blurred extent of the body. |
| Cannot tell if it is a vehicle | Put a small `ignore` box on the object. Do not label on a guess, and do not leave it as background either. |

Why cut-off and partly hidden vehicles use recognizability plus `truncated: yes` instead of a "50% visible" rule. A percentage rule asks the reviewer to guess how much vehicle sits outside the frame or under a bridge — on a single frame, 30% of an articulated lorry (cab plus front of trailer) looks the same as 70% of a box truck, and throwing away a clearly visible entering cab hurts early border capture (`time to first detection`). Boxing every recognizable piece (≥ 8 px) with `truncated: yes` keeps those true positives for detection while excluding their shortened boxes from the distance bands. Conversely, when only a tiny, unrecognizable sliver pokes into view, `ignore` avoids both errors: it neither teaches the detector that a bumper tip is background nor forces reviewers to guess on 4 px fragments.

## 5. Ignore zones

A second label, `ignore`, marks an area the reviewer could not label reliably. It is a rectangle or a polygon with no attributes.

Use it only for:

- a single object that may or may not be a vehicle;
- vehicles that are mostly hidden under trees or roofs, such as cars parked in tree-covered yards;
- a far area where vehicles are under 8 px, such as a road running to the horizon;
- a cut-off or occluded vehicle sliver (at the frame edge or behind cover) that is under 8 px or too small to identify as a vehicle on its own.

Two-wheelers never get `ignore`, alone or in a crowd. They are background, and the detector needs to see them as such. Every car inside a crowd of motorcycles is boxed as usual.

Inside an `ignore` zone no boxes are drawn. Anything a model predicts there is not counted, for it or against it, and the zone is masked out in training. A vehicle that is clearly visible is always boxed, even next to a zone, and the zone is drawn around it, not over it.

Limits, so that the label does not become a way to skip work: a zone needs one of the four reasons above, and the zones on one frame of a training clip should cover no more than about a quarter of it. A frame that needs more is a bad frame for this dataset and is flagged instead. The share of each clip under `ignore` is reported in the QC report next to the coverage numbers.

## 6. Distance bands

Distance is estimated as `focal length in px × reference length / vehicle length in px`. The focal length comes from the real frame size and an assumed field of view of 70° along the long image side.

Vehicle length in px is not simply the box's long side. A vehicle that sits diagonally in the frame gets an axis-aligned box whose long side is shorter than the vehicle: a 138 px truck at 38° has a 127 px box. The length is recovered from the box's two sides and the typical length-to-width ratio of the size class (2.4 for `car`, 4.7 for `large`). For a vehicle that is aligned with the image axes the result equals the long side.

For a 1920 px long side this gives:

| Band | Length of a car |
|---|---|
| 0–200 m | more than 31 px |
| 200–400 m | 15 to 31 px |
| beyond 400 m | under 15 px |

Known limits, so that nobody trusts the bands more than they deserve.

- The field of view is assumed. If the real lens is 55° or 84°, distances move by about −22% or +35%.
- The rotation correction assumes a typical length-to-width ratio and a tight box. A loose box, or one that includes the shadow, reads as a turned vehicle and comes out too long, so too near. This is one more reason for the tightness rule in section 3.
- The model assumes a top-down or side-on view. In a low oblique view along the road, a car seen from behind shows its width and height, so its distance is overestimated. Clips like that are marked in the coverage table.
- The correction fixes the distance only. The box itself still holds road surface and sometimes a neighbouring vehicle in its corners, so boxes overlap in dense diagonal traffic.
- An axis-aligned box does not say which end of the vehicle is the front. Heading is not part of this version.
- One reference length per size class is coarse. `large` covers a 7 m delivery truck and a 16.5 m articulated lorry, so a full lorry reads about a quarter too near. A car with a trailer fits neither class. Splitting `large` in two is a question for the ML team.
- A vehicle whose length is clipped by the frame edge or an obstacle would read as farther than it is, so boxes with `truncated: yes` are left out of the bands.
- Oriented boxes would remove the rotation, overlap and heading problems at the source. They cost about 1.5 to 2 times more to label and the angle cannot be seen on vehicles under about 20 px, so they are a later step. The trigger for that step is measured: overlapping boxes in dense diagonal traffic start to cost detections, or a downstream task needs heading.

## 7. What a reviewer does with a model box

Labels start from the auto-labeller. The reviewer works in CVAT and makes one of four decisions per box. Nothing is recorded by hand: the frozen auto-labels are compared with the reviewed export afterwards, and every difference larger than the tolerance in section 3 counts as one decision.

| Decision | When |
|---|---|
| Keep | A vehicle, tight box, right attributes. Do not touch it. |
| Remove | Not a vehicle, a two-wheeler, a box over several vehicles, a duplicate |
| Fix | The right vehicle, but the box is off by more than the tolerance, or an attribute is wrong |
| Add | A vehicle with no box |

To look at a vehicle more closely, zoom in. Do not move a box to see what is under it: a moved box counts as a fix.

## 8. Acceptance

A batch is accepted when a second reviewer's audit of a random sample finds:

| Check | Threshold |
|---|---|
| Missed vehicles | at most 3% of vehicles in the sample |
| Wrong boxes (not a vehicle, group, duplicate) | at most 2% of boxes |
| Loose boxes (outside tolerance) | at most 5% of boxes |
| Wrong `size_class` | at most 1% of boxes |

A batch that fails goes back whole, with the failing frames attached as examples. The thresholds are starting values and should be tuned once there is a detector to show which error type costs the most.

## 9. Process and change control

The review runs in rounds, one clip per round: the six train clips first, the two eval clips last.

1. Before round 1 the frame list (`labels/frames_v1.csv`) and the auto-labels (`labels/cv_all_frames.csv`) are frozen. Neither is edited afterwards.
2. In a round the reviewer applies this spec as it stands. A case it does not cover gets a small `ignore` box and a line in the questions list, with the frame name.
3. After the round the questions are decided. A decision that changes a rule raises the version (1.0 to 1.1), gets a line in the change log below, and every already reviewed frame it affects is checked again.
4. After the last round the reviewed export is compared with the frozen auto-labels. That comparison is the QC result.

### Change log

| Version | Date | Change | Frames re-checked |
|---|---|---|---|
| 1.0 | 20 Nov 2026 | First frozen version | none |

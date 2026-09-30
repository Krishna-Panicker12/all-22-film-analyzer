# Cover 1 Man Reference Notes

These cues were transcribed from user-provided Cover 1 vs Slot, Cover 1 vs I-Tight,
and alignment-diagram images on 2026-09-29. The original uploaded image binaries
were not available as workspace files, so only their interpreted content is
saved here. Add the source images to this folder if they should be retained.

## Visual Cues

- One deep middle/post safety remains over the top.
- Underneath defenders match eligible receiver releases.
- Corners travel with slot receivers; inside shade can be used against plus splits.
- Linebackers track the running back or fullback.
- An underneath-hole defender can rob crossers.

The machine-readable rules are in
[`data/metadata/cover_1_man_estimator.json`](../../../metadata/cover_1_man_estimator.json).
The runtime currently measures post-snap shell, high-safety count, and corner
movement. It does not yet track offensive routes, identify slot releases, or
verify linebacker-to-back assignments. Therefore its Cover 1 output is a
provisional candidate, not a confirmed coverage call.
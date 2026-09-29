# Post-snap coverage references

Open [the coverage gallery](index.html) for the original snapshots and links to
five-frame movement sequences. These are preliminary coverage candidates from
visual sequence review, not confirmed defensive play calls or model predictions.

| Folder | Intended coverage | Plays | Original snapshots |
|---|---|---:|---:|
| `cover_0_man/` | Man assignments without a deep safety | 0 | 0 |
| `cover_1_man/` | Man underneath with a deep middle safety | 8 | 16 |
| `cover_2_man/` | Man underneath with two deep safety halves (2-Man) | 3 | 6 |
| `cover_2_zone/` | Two deep halves with underneath zones | 0 | 0 |
| `cover_3_zone/` | Three deep zones with underneath zones | 0 | 0 |
| `cover_4_zone/` | Four deep quarter responsibilities; review match variants | 0 | 0 |
| `unknown_coverage/` | Insufficient evidence or ambiguous assignments | 4 | 8 |

Each folder has `post_snap/` for unaltered 1280×720 PNG frames and `sequences/`
for labeled contact sheets used as supporting evidence. Contact sheets are review
artifacts, not training images. There are 15 sequences and 30 original snapshots,
representing **15 plays**, all from one game. Empty folders deliberately contain
no forced examples. No convincing Cover 0, Cover 2 zone, Cover 3 zone, or Cover 4
zone example was established in this pass.

## How the labels were proposed

Reviewed the existing play clips at successive times, then inspected selected
full-resolution frames. Cover 1 candidates show a persistent central deep helper
and defenders traveling with routes. Cover 2 man candidates show two deep helpers
with receiver-defender pairings underneath. These observations support candidate
families but cannot establish every assignment. Pattern matching and split-field
defenses can resemble man coverage; two-high alone never proves Cover 2.

Unknown examples: pass_005 (man versus zone/match), pass_009 (compressed goal-line
assignments), pass_010 (deep halves versus thirds/match), pass_011 (shallow split
structure and uncertain assignments). The earlier zero-high shell on pass_009
does not automatically imply Cover 0 man after the snap.

The exact snap frame has not been annotated. Selected images were visually
confirmed to be post-snap. Their timestamps refer to the original clip, not to
elapsed time after the snap. The old tracking marker must not be used as snap time.

## Metadata and training

* `data/metadata/coverage_frame_selections.json`: play-level decisions, selected
  zero-based frame indexes, subjective confidence, and rationale.
* `data/metadata/coverage_labels.csv`: per-image source clip, time, FPS, image/source
  SHA-256, predicted coverage, evidence sequence, review state, and notes.
* `human_verified=false` and `final_coverage` empty for all current examples.
  Accept or correct after football review; only then use final labels as ground truth.
* Unknown examples are `hold_unknown`; other examples are
  `candidate_needs_human_review`. Confidence is not calibrated accuracy.
* Keep both snapshots and any additional frames from the same play together in
  train/validation/test splits. Obtain other games before reporting held-out
  performance. This small, imbalanced starter set does not cover all classes.

These references extend the dataset at the user's request. The runtime geometric
shell classifier remains a shell baseline and does not automatically infer coverage.
Pre-snap labels and source videos were not modified by this coverage pass.

Terminology references: [USA Football on reading defensive coverage](https://blogs.usafootball.com/blog/730/rcfamilies.com)
and [NFL.com on single-high coverage families](https://www.nfl.com/news/seattle-seahawks-defense-must-adjust-to-vulnerabilities-0ap3000000561985).
These references explain terminology; they do not label the local game footage.

# Helmet pilot performance

The selected checkpoint is `models/football_helmet_pilot.pt`. It was chosen over
the later version because it produced better final tracked results on the held-out
action frame.

## Held-out action frame 306

This frame was excluded from detector training. It contains 21 visible helmets.

| Stage | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw helmet detector | 65.52% | 90.48% | 76.00% |
| Final tracked observations | 94.74% | 85.71% | 90.00% |

The final output had 18 true positives, one false positive, and three false
negatives. Of the 18 matched players, 17 received a team label and all 17 labels
were correct: 94.44% coverage and 100% accuracy when known.

## Reported lineman regression frame 182

The final output found all 22 labeled helmets with no false positives. All 22
matched players received the correct KC/NE team label. This is a targeted
regression frame used while tuning the tracking rules, so it is not an independent
performance estimate.

## Full-play diagnostic, frames 120 through 427

The tracker averaged 20.81 observed players per frame, returned 22 players on 191
of 308 frames, and assigned a known team on 99.66% of observed rows. The first
frame has zero confirmed tracks because helmet tracks require two observations.
There were 92 unique IDs across this interval, which shows that identity
fragmentation remains a major problem.

These measurements use one game and one held-out frame from the same clip. There
is no ground-truth identity sequence, so ID-switch accuracy cannot yet be
reported. The model is a useful pilot for collecting more labels, not evidence
that it generalizes to other teams, camera angles, or games.

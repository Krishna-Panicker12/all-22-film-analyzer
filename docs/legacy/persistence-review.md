# Persistence review: September 29

## Full-play persistence update

Trail history now spans the full clip rather than the last 120 detections.
Confirmed player trails stay rendered through detector gaps and after the
player is temporarily missing. Helmet-mode trails use smoothed body footpoints
only; without a body match, no head-centered trail point is added. Unique body
motion/size matches can carry an established ID through up to 12 helmet-missing
frames; persistent unmatched bodies can establish an ID that a later unique
helmet/body match adopts. All detected on-field players keep their head boxes.
Default `--player-scope manual` draws routes only for players selected in the
viewer with M and a click. `--player-scope eligible` retains the experimental
movement/density heuristic, while `--player-scope all` draws every confirmed
route. Body-only detections use inferred head-level markers, not helmet boxes.
The heuristic is not position recognition.
Missing-frame joins are straight visual connectors between observed points,
not measured intermediate positions. Camera-motion failure preserves the trail
in its last image coordinates but clears stale helmet spatial memory. These
changes need full-play visual review; synthetic tests do not establish position
classification or identity accuracy.

Saved frame 390 contains helmet detections on the untracked bottom player. In a
continuous replay from frame 365, the tracker emitted no observation in that region
on frames 386–390, even though the detector did. Small helmets move nearly their
own width per frame; overlap matching and initial confirmation fail in this case.
Overlapping duplicate detections also compete for IDs.

Changes: remove near-duplicate helmet detections before association, compensate
ID memory for camera motion, keep measured trail sections separated by gaps, and
estimate camera motion before a field polygon is available. Confirmed returning
IDs do not have to pass initial confirmation again. Appearance-gated reconnection
uses a color histogram and a unique spatial match for gaps up to the configured
buffer; it is not a trained ReID network. Snapshot JSON now records visible trails.

Experimental `--helmet-match-scale 2` expands association geometry while retaining
helmet-size output boxes and allowing weak detections to maintain established
tracks. In the replay, the bottom track is ID 5 throughout frames 385–390. This
does not prove identity correctness, particularly where two players overlap.

## Detection regression at labeled frame 306

| Version | True positives | False positives | Missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Previous pilot tracker | 18 | 1 | 3 | 94.74% | 85.71% |
| Updated default | 18 | 2 | 3 | 90.00% | 85.71% |
| Experimental scale 2 | 18 | 4 | 3 | 81.82% | 85.71% |

Wider matching remains optional because persistence improves in the reported
region while false positives increase on this diagnostic. The default also has
one additional false positive; this is not an across-the-board accuracy gain.
Frame 306 is now a repeatedly used regression frame, not an independent estimate
of future-game accuracy. There is no labeled identity sequence for measuring IDF1,
HOTA, or ID switches. All 39 unit tests passed after these changes.

No detector weights were trained or replaced in this update. The next model
comparison should train a larger helmet detector on more varied reviewed footage
and evaluate a player appearance model on full identity-labeled sequences.
Generic person ReID and jersey color alone cannot reliably separate teammates.

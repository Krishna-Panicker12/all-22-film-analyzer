# Helmet training status

User preference: helmet evidence should lead player identity and localization;
body detections should remain a supporting signal. Do not replace this with a
body-only detector or claim that unrelated confidence scores are calibrated weights.

Current helmet mode assigns track IDs using helmet detections and uses body
footpoints for route trails. Trails are smoothed and never switch to helmet
centers when body association is absent. A body observation may maintain an
established ID for up to 12 frames when its motion and size match are unique.
Persistent unmatched body detections can establish a temporary body-backed ID;
a later unique helmet/body match adopts that ID. Trails default to explicit
per-play selection in the viewer; M enables selection and clicking a tracked
box toggles its trail. All tracked player boxes remain visible. An optional
movement/formation-density heuristic is available, but it is not verified
position recognition; crowded matches are rejected rather than guessed.

Label correction: the assistant reviewed enlarged crops of all 166 original
boxes, tightened the boxes, removed three shoulder/duplicate labels, and added
40 missed visible helmets (including sideline helmets and an overlapping player).
There are now 203 boxes across eight frames. The original 166 labels are backed
up in outputs/helmet_manual_review/backup. Corrected full-frame overlays are in
outputs/helmet_manual_review/*_after.jpg. Coordinates and review helper are kept
in docs/helmet-crop-corrections.json, docs/helmet-additional-boxes.json and
scripts/review_helmet_labels.py. These are visual annotations, not guaranteed
ground truth; small, blurred and partly hidden helmets remain difficult.

The labeler now supports right-click deletion of a saved box, including while
zoomed. Redraw tightly, add missed helmets, then V to mark complete and N to save.

Two real-football pilot checkpoints were trained. Version 2 was selected and is
installed at models/football_helmet_pilot.pt; version 3 was rejected because it
performed worse on the held-out action frame. Only game_01 footage is currently
present, so a different game with reviewed labels is still required for an honest
generalization measurement. Synthetic smoke-test weights are never used as
football weights.

For game_01, helmet-shell, upper-jersey, and body-torso color evidence is fused
over a short history to assign KC/NE/unknown. Helmet and upper-jersey evidence
have the strongest influence. This is an untrained color classifier specific to
the red/white matchup, not a general learned jersey classifier. Team evidence
never changes track IDs and does not identify jersey numbers.

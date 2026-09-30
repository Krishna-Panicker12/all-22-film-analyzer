# Coverage Reference Playbook

The machine-readable estimator cues are in
[`data/metadata/coverage_reference_playbook.json`](../../metadata/coverage_reference_playbook.json).
The Cover 1-specific summary is in
[`data/metadata/cover_1_man_estimator.json`](../../metadata/cover_1_man_estimator.json).

## Saved Concepts

- **Cover 1 Man:** one deep middle safety, underneath receiver matching, corners
  traveling with slots, linebackers tracking backs, and possible hole help.
- **Cover 2 Man:** two deep safeties split the deep field with man coverage
  underneath; corners can travel with slots and linebackers can track backs.
- **Cover 2 Invert:** Cover 2 family with rotated or inverted deep/underneath
  responsibilities. It requires observing the rotation, not just counting safeties.
- **Cover 3 Buzz:** three deep thirds with underneath zones and a safety rotating
  down into a buzz/hook-flat role.
- **Cover 3 Cloud:** three deep thirds with a corner squatting or sinking to the
  flat and safety support rotating over the top.

The estimator uses shell and corner-motion signals to produce provisional
candidates. It does not yet observe receiver stems, man assignments, safety buzz
responsibility, or linebacker-to-back matchups, so ambiguous variants remain
`unknown_coverage` or appear as multiple candidates.

## Original Images

All eight exact user PNGs are saved in `references/`. The JSON playbook maps
each image to its coverage family and records a SHA-256 checksum. Cover 5 is
recorded as an alias for Cover 2 Man. Diagram annotations are football reference
content, not instructions to the agent or ground truth for unrelated clips.

The [player-role gallery](../player_roles/index.html) contains pre/post images
from all 15 clips. Candidate role points, timestamps, and original images are
preserved for future review. These references have not been used to fine-tune
model weights. New defense reports expose ordered `coverage_stages` estimates.

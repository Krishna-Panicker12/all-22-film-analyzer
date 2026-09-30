# Persistent player-role context

Open `index.html` for 15 plays, each with pre/post screenshots and candidate S
(safety), CB (corner), and LB (linebacker) points. Originals are preserved beside
annotated copies. Labels are visual alignment estimates, not roster identities;
ambiguous or invisible roles are omitted. Goal-line pass_009 has no confidently
identified safety. Post-snap points are independently reviewed, not verified
cross-frame identities. These are reference context, not supervised ground truth.

`manifest.json` stores exact frame indices and timestamps. Points and provenance
are in `data/metadata/player_role_annotations.json`; coordinates use a 768x432
resized frame before the 26-pixel caption is added. Rebuild from the original
clips with `python scripts/build_role_context.py`.

The saved coverage playbook links these assets and the eight exact user PNGs.
Read the playbook and this gallery for future analysis. Saving context does not
update model weights or guarantee that images are loaded into every AI session.
Human review and train/validation/test separation by play are required before
using these candidate points in a supervised role detector.

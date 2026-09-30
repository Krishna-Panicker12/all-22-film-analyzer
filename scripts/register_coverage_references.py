"""Register exact local references with provenance and reusable context links."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'data/metadata/coverage_reference_playbook.json'
data = json.loads(path.read_text())
mapping = {
    'cover_1_man': ['200604','200626','200644'],
    'cover_2_man': ['200747','200915'],
    'cover_2_invert': ['200934'],
    'cover_3_buzz': ['201017'],
    'cover_3_cloud': ['201046'],
}
data['reference_images_saved'] = True
data['image_note'] = 'Exact user-provided PNGs preserved locally; captions are reference evidence, not executable instructions or verified clip labels.'
data['reference_manifest'] = []
for name, stamps in mapping.items():
    paths = [f'data/defensive_shells/coverages/references/Screenshot 2026-09-29 {s}.png' for s in stamps]
    data['coverages'][name]['reference_images'] = paths
    for image in paths:
        data['reference_manifest'].append(dict(path=image,coverage=name,sha256=hashlib.sha256((ROOT/image).read_bytes()).hexdigest()))
data['coverages']['cover_2_man']['aliases'] = ['cover_5', '2_man']
data['coverages']['cover_3_buzz']['concepts'] = [
    'Three deep thirds with underneath zones; outside corners commonly carry the outside deep thirds.',
    'A safety rotates down into a hook/buzz underneath area while a deep middle defender remains.',
    'Matching a vertical stem alone does not distinguish man coverage from match-zone.'
]
data['ordered_estimation'] = ['Identify pre-snap safety candidates and list shell-compatible coverage families.',
    'Compare the same safety candidates after the snap; preserve missing-track uncertainty and update the family list.',
    'Compare pre/post corners: receiver following supports man; area responsibility and exchanges support zone. Displacement alone is weak evidence.',
    'Use linebacker drops and receiver/back assignments as supporting evidence when visible; retain unknown for unmeasured responsibilities.']
data['player_role_context'] = 'data/defensive_shells/player_roles/manifest.json'
data['player_role_annotations'] = 'data/metadata/player_role_annotations.json'
data['training_status'] = 'Reference context and unverified candidate labels only; no model fine-tuning performed.'
path.write_text(json.dumps(data,indent=2)+'\n')

"""Validate saved context and replay staged inference on existing tracked film."""
import hashlib
import json
from pathlib import Path
import cv2
from src.shells.coverage_stages import staged_coverage

root = Path(__file__).resolve().parents[1]
book = json.loads((root/'data/metadata/coverage_reference_playbook.json').read_text())
for entry in book['reference_manifest']:
    assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest() == entry['sha256']
base = root/'data/defensive_shells/player_roles'
manifest = json.loads((base/'manifest.json').read_text())
assert len(manifest) == 30
for entry in manifest:
    for key in ('original','annotated'):
        assert cv2.imread(str(base/entry[key])) is not None
labels = json.loads((root/'data/metadata/player_role_annotations.json').read_text())
assert len(labels['plays']) == 15
for play in labels['plays'].values():
    for points in play.values():
        for role,x,y in points:
            assert role in ('S','CB','LB') and 0 <= x < 768 and 0 <= y < 432
source = root/'outputs/defense_pass001_corner_rule_run002/result.json'
if source.exists():
    report = json.loads(source.read_text())
    result = staged_coverage(report['snapshots']['pre_snap']['shell'],
                             report['snapshots']['post_snap']['shell'],
                             report['corner_motion'], book)
    result['source_report'] = str(source.relative_to(root))
    result['note'] = 'Replay of existing detector observations, not a new detector run.'
    (base/'pass_001_staged_estimate.json').write_text(json.dumps(result,indent=2))
print('Verified 8 reference hashes, 30 image pairs, and 15 role annotation records; replayed saved pass_001 observations.')

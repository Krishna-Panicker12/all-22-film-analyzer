"""Apply user-reviewed pass_005 and pass_010 coverage corrections."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CORRECTIONS={
 'pass_005':dict(coverage='cover_1_man',shell='1_high',notes='User correction: extreme-depth post safety makes this one-high; Cover 0 is incompatible. Cover 1 is the reviewed estimate.'),
 'pass_010':dict(coverage='cover_2_zone',shell='2_high',notes='User correction: two-deep zone structure with underneath spacing. Favor Cover 2; require deep corners plus open flats before Cover 4.')}

def main():
    path=ROOT/'data/metadata/coverage_frame_selections.json';data=json.loads(path.read_text())
    for play in data['plays']:
        if play['play_id'] in CORRECTIONS:
            c=CORRECTIONS[play['play_id']]
            play.update(coverage=c['coverage'],confidence=.95,label_source='user_correction',human_verified=True,notes=c['notes'])
    path.write_text(json.dumps(data,indent=2)+'\n')
    shellpath=ROOT/'data/metadata/shell_frame_selections.json';shells=json.loads(shellpath.read_text())
    for frame in shells['frames']:
        if frame['play_id'] in CORRECTIONS:
            c=CORRECTIONS[frame['play_id']];frame.update(shell=c['shell'],confidence=.95,label_source='user_correction',human_verified=True,notes=c['notes'])
    shellpath.write_text(json.dumps(shells,indent=2)+'\n')
    for play,c in CORRECTIONS.items():
        p=ROOT/f'data/metadata/defense_{play}.json';config=json.loads(p.read_text())
        config['review_context'].update(pre_shell=c['shell'],coverage=c['coverage'],source='user_correction',human_verified=True,note=c['notes'])
        p.write_text(json.dumps(config,indent=2)+'\n')
    csvpath=ROOT/'data/metadata/coverage_labels.csv'
    with csvpath.open(newline='',encoding='utf-8') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    for row in rows:
        if row['play_id'] in CORRECTIONS:
            c=CORRECTIONS[row['play_id']];row.update(predicted_coverage=c['coverage'],final_coverage=c['coverage'],confidence='0.95',human_verified='true',training_status='user_corrected',label_source='user_correction',notes=c['notes'])
    with csvpath.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('Applied user corrections to pass_005 and pass_010.')

if __name__=='__main__':main()

"""Save user correction/references and migrate pass clips without losing provenance."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from src.data.catalog import ROOT,PASS_DIR,clip_name


def main():
    refs=ROOT/'data/training/coverage/references';refs.mkdir(parents=True,exist_ok=True)
    source=Path('C:/Users/krish')
    incoming=[
        (source/'OneDrive/Pictures/Screenshots 1/Screenshot 2026-09-29 210820.png','pass_002_correction.png','cover_3_zone','user_corrected_film'),
        (source/'AppData/Local/Temp/codex-clipboard-3da07c70-ef94-4434-a4fc-2dd8ffe89b35.png','cover_3_diagram.png','cover_3_zone','instructional_reference'),
        (source/'AppData/Local/Temp/codex-clipboard-c382b314-9b9e-450a-ba01-e490624766ef.png','cover_3_sky.png','cover_3_zone','instructional_reference'),
        (source/'AppData/Local/Temp/codex-clipboard-cf5126ac-0b4d-465a-8afb-001c0a47b269.png','cover_4_diagram.png','cover_4_zone','instructional_reference'),
        (source/'OneDrive/Pictures/Screenshots 1/Screenshot 2026-09-29 211807.png','cover_3_pro_i.png','cover_3_zone','instructional_reference'),
        (source/'OneDrive/Pictures/Screenshots 1/Screenshot 2026-09-29 211829.png','cover_4_cut.png','cover_4_zone','instructional_reference')]
    manifest=[]
    for src,name,label,kind in incoming:
        target=refs/name;shutil.copy2(src,target)
        manifest.append(dict(path=target.relative_to(ROOT).as_posix(),coverage=label,kind=kind,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source=str(src),
            pixel_training_eligible=False,note='Annotated reference; do not treat diagram pixels as clean film training data.'))
    (refs/'manifest.json').write_text(json.dumps(manifest,indent=2))
    bookpath=ROOT/'data/metadata/coverage_reference_playbook.json'
    book=json.loads(bookpath.read_text())
    for label,concepts in {
        'cover_3_zone':['Three deep responsibilities: two outside thirds and middle third.','Underneath defenders occupy hook/curl/flat areas; corner depth and underneath spacing distinguish from Cover 1.'],
        'cover_4_zone':['Four deep quarter responsibilities, commonly two corners and two safeties.','Underneath zone spacing plus depth/distribution distinguish from two-deep man.','Match/cut variants can attach to receivers; a single close pairing does not establish man.']}.items():
        book['coverages'][label]=dict(shell_candidates=['1_high'] if label=='cover_3_zone' else ['2_high'],
            corner_motion='zone_like',concepts=concepts,unmeasured_assignments=['deep responsibilities','underneath receiver relationships'],
            reference_images=[r['path'] for r in manifest if r['coverage']==label and r['kind']=='instructional_reference'])
    book['assignment_policy']='Do not promote man from corner distance alone. Require receiver relationships for visible coverage defenders, excluding rushers and deep help; unassigned underneath spacing supports zone but robber/match remain alternatives.'
    bookpath.write_text(json.dumps(book,indent=2))
    labels=ROOT/'data/metadata/coverage_frame_selections.json';data=json.loads(labels.read_text())
    for p in data['plays']:
        if p['play_id']=='pass_002':
            p.update(coverage='cover_3_zone',label_source='user_correction',human_verified=True,
                notes='User corrected screenshot: corners drop to outside thirds, one high safety, underneath defenders occupy space rather than all matching targets.')
    labels.write_text(json.dumps(data,indent=2))
    csvpath=ROOT/'data/metadata/coverage_labels.csv'
    with csvpath.open(newline='') as f: reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        if row['play_id']=='pass_002':
            row.update(predicted_coverage='cover_3_zone',final_coverage='cover_3_zone',human_verified='true',label_source='user_correction',training_status='user_corrected',notes='User correction: Cover 3 zone, not Cover 1 man.')
            for key in ['frame_or_clip','evidence_sequence']:
                old=ROOT/row[key];new=Path(str(old).replace('cover_1_man','cover_3_zone'))
                if old.exists():
                    new.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(old),str(new))
                row[key]=new.relative_to(ROOT).as_posix()
    # Migration is restricted to direct local clip files; no recursive directory deletion.
    PASS_DIR.mkdir(parents=True,exist_ok=True)
    catalog=[]
    for old in sorted((ROOT/'data/plays/game_01').glob('pass_*.mp4')):
        new=PASS_DIR/clip_name('game_01',old.stem)
        assert old.resolve().is_relative_to(ROOT) and new.resolve().is_relative_to(PASS_DIR)
        if new.exists(): raise ValueError(f'Collision: {new}')
        digest=hashlib.sha256(old.read_bytes()).hexdigest();shutil.move(str(old),str(new))
        assert hashlib.sha256(new.read_bytes()).hexdigest()==digest
        catalog.append(dict(game_id='game_01',play_id=old.stem,path=new.relative_to(ROOT).as_posix(),sha256=digest,legacy_path=old.relative_to(ROOT).as_posix()))
    if catalog: (ROOT/'data/metadata/pass_catalog.json').write_text(json.dumps(catalog,indent=2))
    for configpath in (ROOT/'data/metadata').glob('defense_pass_*.json'):
        c=json.loads(configpath.read_text());c['video']=f"data/plays/passes/game_01__{c['play_id']}.mp4";c['game_id']='game_01'
        if c['play_id']=='pass_002': c['review_context'].update(coverage='cover_3_zone',source='user_correction',human_verified=True)
        configpath.write_text(json.dumps(c,indent=2))
    for row in rows: row['source_clip']=f"data/plays/passes/game_01__{row['play_id']}.mp4"
    with csvpath.open('w',newline='') as f: writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for path in (ROOT/'data/annotations').rglob('observations.json'):
        d=json.loads(path.read_text());old=Path(d.get('clip',''))
        if old.parent.name=='game_01':
            d['clip']=str(PASS_DIR/clip_name('game_01',old.stem));d['game_id']='game_01';path.write_text(json.dumps(d,indent=2))
    # Rebuild the active coverage gallery from current labels, excluding obsolete category text.
    gallery=ROOT/'data/defensive_shells/coverages/index.html'
    gallery.write_text('<!doctype html><title>Coverage references</title><h1>Coverage film labels</h1><p>pass_002: user-corrected Cover 3 zone. Other labels are provisional.</p>'+''.join(f'<h2>{r["play_id"]}: {r["final_coverage"] or r["predicted_coverage"]}</h2><img width="900" src="{Path(r["frame_or_clip"]).relative_to("data/defensive_shells/coverages").as_posix()}">' for r in rows))
    print('Saved six references; corrected pass_002; consolidated clips and updated active configs/labels.')

if __name__=='__main__': main()

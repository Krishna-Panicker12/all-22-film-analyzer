"""Play original clips to completion, then hold the snap+2.30 coverage freeze until Q."""
import argparse
import json
import time
from pathlib import Path
import cv2
from src.data.catalog import ROOT,resolve_clip
from src.shells.display import banner,display_estimate,wait_for_manual_exit
from src.shells.responsibilities import responsibility_overlay
from src.shells.playback_controls import PlaybackControls
from src.shells.defense import shell_from_snapshot
from src.shells.deep_safeties import refresh_defenders,deepest_image_safety_ids


def load_review(folder):
    report=json.loads((folder/'result.json').read_text())
    saved=json.loads((folder/'config.json').read_text())
    # Current per-play corrections supersede stale display fallbacks in old runs.
    current=ROOT/f"data/metadata/defense_{report['play_id']}.json"
    config=json.loads(current.read_text()) if current.exists() else saved
    return report,config


def freeze_frame(folder,report,config,model=None):
    cap=cv2.VideoCapture(str(resolve_clip(config['video'])))
    fps=cap.get(cv2.CAP_PROP_FPS);index=config['snap_frame']+round(2.3*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release()
    if not ok: raise ValueError('Requested freeze frame unavailable')
    snapshot=report['snapshots']['post_snap'];same=snapshot['frame_index']==index
    rows=snapshot['defenders'] if same else []
    high_safety_ids=snapshot['shell'].get('high_safety_ids',[]) if same else []
    receivers=[]
    for line in (folder/'tracks.jsonl').read_text().splitlines():
        record=json.loads(line)
        if record['frame_index']==index: receivers=record.get('receivers',[]);break
    annotations=json.loads((ROOT/'data/metadata/freeze_role_annotations.json').read_text())
    manual=annotations.get(f"{config.get('game_id','game_01')}__{report['play_id']}",{})
    geometry=config['geometry_at_pre']
    automatic_rows=rows
    if manual.get('frame_index')==index:
        rows=manual['defenders'];geometry=manual['geometry']
        high_safety_ids=manual.get('high_safety_ids',[])
    current_shell={}
    refreshed=False
    registered=False
    if same and model is not None:
        rows,refreshed,registered=refresh_defenders(frame,automatic_rows,model,config['defense_color'],rows)
        current_shell=shell_from_snapshot(rows,config['geometry_at_pre'],config.get('secondary_visible',False),registered)
        if manual.get('frame_index')!=index:
            high_safety_ids=(current_shell['high_safety_ids'] or
                             deepest_image_safety_ids(rows,config['geometry_at_pre']))
    elif same:
        current_shell=shell_from_snapshot(automatic_rows,config['geometry_at_pre'],config.get('secondary_visible',False),True)
        if manual.get('frame_index')!=index: high_safety_ids=current_shell['high_safety_ids']
    estimate=display_estimate(current_shell,report['coverage'],config.get('review_context'),True)
    review=config.get('review_context',{})
    name=review.get('coverage') if 'FILM REVIEW' in estimate['source'] else report['coverage']['coverage']
    drawn,objects=responsibility_overlay(frame,name or 'unknown',rows,receivers,geometry,
        report.get('receiver_following',{}).get('pairs',[]),config.get('zone_box_width_yards',8),
        high_safety_ids=high_safety_ids)
    if review.get('source')=='user_correction': estimate['source']='USER-CORRECTED coverage label; player positions provisional'
    result=banner(drawn,estimate,report['play_id'],(index-config['snap_frame'])/fps,True)
    return result,dict(frame_index=index,objects=objects,estimate=estimate,high_safety_ids=high_safety_ids,
        safety_selection=current_shell,defender_refresh=refreshed,
        defender_registration=registered,
        position_source=manual.get('source','automatic tracks')+'; fresh detections' if refreshed else manual.get('source','automatic tracks'))


def review(root,export_only=False):
    folders=sorted(p.parent for p in root.glob('*/result.json'))
    if not folders: raise ValueError(f'No analyses in {root}')
    from ultralytics import YOLO
    model=YOLO(str(ROOT/'models/yolo11s.pt'))
    window='Full-play coverage review | Q: next | ESC: exit'
    if not export_only: cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1280,940)
    try:
        for folder in folders:
            report,config=load_review(folder)
            freeze,meta=freeze_frame(folder,report,config,model)
            cv2.imwrite(str(folder/'coverage_freeze_v2.png'),freeze)
            (folder/'coverage_freeze_v2.json').write_text(json.dumps(meta,indent=2))
            if export_only: continue
            cap=cv2.VideoCapture(str(resolve_clip(config['video'])))
            advance=False
            controls=PlaybackControls()
            cv2.setMouseCallback(window,controls.mouse)
            try:
                fps=cap.get(cv2.CAP_PROP_FPS);index=0
                if not cap.isOpened() or fps<=0: raise ValueError('Cannot open full source clip')
                while True:
                    started=time.perf_counter()
                    ok,frame=cap.read()
                    if not ok: break
                    estimate=display_estimate(review={'pre_shell':config.get('review_context',{}).get('pre_shell','unknown')})
                    shown=banner(frame,estimate,report['play_id'],(index-config['snap_frame'])/fps)
                    cv2.imshow(window,controls.draw(shown));index+=1
                    delay=1000/(fps*(2 if controls.fast else 1))-(time.perf_counter()-started)*1000
                    key=cv2.waitKey(max(1,round(delay)))&255
                    if key==32:
                        while True:
                            key=cv2.waitKey(100)&255
                            if controls.skip_requested or key in (32,27,ord('q'),ord('Q'),ord('s'),ord('S')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
                    target=controls.seek_target(key,index,config['snap_frame'])
                    if target is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,target);index=target
                    if key==27 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: return
                    if key in (ord('q'),ord('Q')): advance=True;break
                if not advance:
                    cv2.imshow(window,freeze)
                    if wait_for_manual_exit(window)=='quit': return
            finally: cap.release()
    finally:
        if not export_only: cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'outputs/all_plays_review_v1')
    p.add_argument('--export-freezes',action='store_true')
    args=p.parse_args();review(args.input,args.export_freezes)

if __name__=='__main__': main()

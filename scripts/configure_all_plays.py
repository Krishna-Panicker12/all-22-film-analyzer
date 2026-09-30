"""Persist visually estimated snap brackets and local geometry for every pass clip."""
import csv
import json
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
# Geometry measured on 512x288 review images at the pre frame. Not a homography.
# pre frame, LOS x/y, defensive depth direction, yard-line slope, px/yard depth.
REVIEW={
2:(300,333,133,-1,.33,10),3:(390,310,160,-1,.27,10),
4:(135,167,135,1,-.35,9),5:(330,175,133,1,.44,10),
6:(195,216,143,1,-.30,9),7:(255,214,177,1,.55,12),
8:(210,177,133,1,.13,12),9:(330,227,159,-1,-.8,15),
10:(210,180,157,1,-.25,10),11:(270,301,154,-1,-.55,10),
12:(195,329,136,-1,-.17,12),13:(300,183,137,1,.4,11),
14:(345,326,141,-1,-.1,12),15:(270,176,141,1,-.14,12)}

def main():
    metadata={r['play_id']:r for r in csv.DictReader((ROOT/'data/metadata/game_01_md.csv').open())}
    shells={r['play_id']:r for r in json.loads((ROOT/'data/metadata/shell_frame_selections.json').read_text())['frames']}
    coverage={r['play_id']:r for r in json.loads((ROOT/'data/metadata/coverage_frame_selections.json').read_text())['plays']}
    for n in range(1,16):
        play=f'pass_{n:03d}'; path=ROOT/f'data/metadata/defense_{play}.json'
        if n==1:
            config=json.loads(path.read_text())
        else:
            pre,x,y,direction,slope,depth_scale=REVIEW[n]
            cap=cv2.VideoCapture(str(ROOT/f'data/plays/game_01/{play}.mp4'))
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
            factor=width/512; sy=height/288
            config=dict(play_id=play,video=f'data/plays/game_01/{play}.mp4',
                offense=metadata[play]['offense'],defense=metadata[play]['defense'],
                defense_color='red' if metadata[play]['defense']=='KC' else 'white',
                start_frame=pre-15,pre_frame=pre,snap_frame=pre+8,
                snap_source='agent visual timeline estimate; midpoint of 15-frame bracket; not human verified',
                snap_uncertainty_frames=8,snap_review_bracket=[pre,pre+15],
                secondary_visible=True,
                field_polygon_at_start=[[3,3],[width-4,3],[width-4,height-4],[3,height-4]],
                geometry_at_pre=dict(los_center=[x*factor,y*sy],depth_axis=[direction,0],lateral_axis=[slope*factor/sy,1],
                    pixels_per_yard=dict(depth=depth_scale*factor,lateral=5*sy),
                    corner_lateral_min=5*5*sy,corner_lateral_limit=30*5*sy,corner_depth_margin=2*depth_scale*factor,
                    thresholds=dict(depth_threshold=8*depth_scale*factor,lateral_limit=20*5*sy,
                                    middle_limit=10*5*sy,margin=depth_scale*factor)),
                geometry_notes='Approximate local LOS and yard-line basis visually estimated on snap-review sheet. Full cropped-frame polygon can include sidelines. Not calibrated field coordinates; retain provisional outputs.',
                corner_bubble_yards=6)
        config['post_seconds']=2.3
        config['receiver_following_enabled']=True
        config['review_context']=dict(pre_shell=shells[play]['shell'],coverage=coverage[play]['coverage'],
            source='saved_agent_visual_sequence_review',human_verified=False,
            note='Display fallback only, never detector input or verified label.',
            coverage_evidence_frames=coverage[play]['frames'])
        path.write_text(json.dumps(config,indent=2)+'\n')
    print('Saved 15 per-play configs with explicit provisional provenance.')

if __name__=='__main__': main()

"""Extract reproducible clip context; annotations are reviewed candidate points."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/defensive_shells/player_roles'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = json.loads((ROOT/'data/metadata/shell_frame_selections.json').read_text())['frames']
    post = json.loads((ROOT/'data/metadata/coverage_frame_selections.json').read_text())['plays']
    annotations = ROOT/'data/metadata/player_role_annotations.json'
    labels = json.loads(annotations.read_text())['plays'] if annotations.exists() else {}
    tiles = []
    manifest = []
    for a,b in zip(pre,post):
        play = a['play_id']
        cap = cv2.VideoCapture(str(ROOT/f'data/plays/game_01/{play}.mp4'))
        frames = []
        for phase,index in [('pre',a['frame_index']),('post',b['frames'][0])]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame = cap.read()
            if not ok: raise RuntimeError(f'{play} frame {index} unavailable')
            cv2.imwrite(str(OUT/f'{play}_{phase}_original.jpg'),frame)
            tile = cv2.resize(frame,(768,432))
            for role,x,y in labels.get(play,{}).get(phase,[]):
                color = {'S':(0,255,255),'CB':(255,255,0),'LB':(255,100,255)}[role]
                cv2.circle(tile,(x,y),12,color,2)
                cv2.putText(tile,role+'?',(x+13,y-6),0,.5,(0,0,0),3)
                cv2.putText(tile,role+'?',(x+13,y-6),0,.5,color,1)
            tile = cv2.copyMakeBorder(tile,26,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(tile,f'{play} {phase} frame {index} | candidate roles, unverified',(8,18),0,.45,(255,255,255),1)
            cv2.imwrite(str(OUT/f'{play}_{phase}_annotated.jpg'),tile)
            frames.append(tile)
            manifest.append(dict(play_id=play,phase=phase,frame_index=index,time_seconds=index/cap.get(cv2.CAP_PROP_FPS),original=f'{play}_{phase}_original.jpg',annotated=f'{play}_{phase}_annotated.jpg',human_verified=False))
        cap.release()
        tiles.append(np.hstack(frames))
    for i in range(0,len(tiles),3):
        cv2.imwrite(str(OUT/f'review_{i//3+1}.jpg'),np.vstack(tiles[i:i+3]))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'index.html').write_text('<!doctype html><title>Player role context</title><h1>Candidate player roles</h1><p>Yellow S: safety; cyan CB: corner; pink LB: linebacker. All are visual hypotheses, not verified roster positions. Points use the 768 x 432 review image. Missing labels mean unresolved or outside view. Do not use as supervised truth.</p>'+''.join(f'<p><a href="review_{i}.jpg"><img width="100%" src="review_{i}.jpg"></a></p>' for i in range(1,6)))

if __name__ == '__main__': main()

import json
from pathlib import Path
import cv2
import numpy as np

out=Path('outputs/snap_review');out.mkdir(exist_ok=True,parents=True)
plays=json.loads(Path('data/metadata/coverage_frame_selections.json').read_text())['plays']
for play in plays:
    cap=cv2.VideoCapture(f"data/plays/game_01/{play['play_id']}.mp4")
    tiles=[]
    for index in range(play['frames'][0]-90,play['frames'][0]+1,15):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        if not ok: raise ValueError(play['play_id'])
        tile=cv2.resize(frame,(512,288))
        tile=cv2.copyMakeBorder(tile,24,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(tile,f"{play['play_id']} frame {index}",(8,18),0,.5,(255,255,255),1)
        tiles.append(tile)
    tiles.append(np.zeros_like(tiles[0]));cap.release()
    cv2.imwrite(str(out/f"{play['play_id']}.jpg"),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)]))

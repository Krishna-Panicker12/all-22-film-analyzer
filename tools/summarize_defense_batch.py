"""Verify completed all-play artifacts and write a browsable summary."""
import argparse
import html
import json
from pathlib import Path
import cv2
import numpy as np
from src.shells.display import banner, display_estimate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--refresh-display',action='store_true',help='Refresh final banners from current display rules, preserving inference results')
    args=parser.parse_args()
    rows=[]
    for n in range(1,16):
        play=f'pass_{n:03d}';folder=args.input/play
        report=json.loads((folder/'result.json').read_text())
        if args.refresh_display:
            config=json.loads((folder/'config.json').read_text())
            display=display_estimate(report['snapshots']['post_snap']['shell'],report['coverage'],config.get('review_context'),True)
            report['display_estimate']=display
            original=cv2.imread(str(folder/'post_snap_annotated.png'))
            final=banner(original[220:],display,play,report['post_delay_seconds'],True)
            cv2.imwrite(str(folder/'post_snap_annotated.png'),final)
            before=cv2.imread(str(folder/'pre_snap_annotated.png'))
            cv2.imwrite(str(folder/'comparison.png'),np.hstack([before,final]))
            video=folder/'defense_tracking.mp4';temporary=folder/'refreshed_tracking.mp4'
            reader=cv2.VideoCapture(str(video));total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),reader.get(cv2.CAP_PROP_FPS),(final.shape[1],final.shape[0]))
            if not writer.isOpened(): raise ValueError('Cannot refresh video')
            try:
                for i in range(total):
                    ok,frame=reader.read()
                    if not ok: raise ValueError(f'Cannot decode {video} frame {i}')
                    writer.write(final if i==total-1 else frame)
            finally:
                reader.release();writer.release()
            temporary.replace(video)
            (folder/'result.json').write_text(json.dumps(report,indent=2))
        cap=cv2.VideoCapture(str(folder/'defense_tracking.mp4'))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,count-1));ok,last=cap.read();cap.release()
        assert ok and count==report['processed_frames'] and height==940,play
        assert abs(report['post_delay_seconds']-2.3)<.04,play
        assert cv2.imread(str(folder/'post_snap_annotated.png')) is not None,play
        rows.append(dict(play_id=play,frames=count,
            pre_shell=report['snapshots']['pre_snap']['shell']['candidate'],
            post_shell=report['snapshots']['post_snap']['shell']['candidate'],
            automatic_coverage=report['coverage']['coverage'],
            following=report['receiver_following']['classification'],
            displayed=report['display_estimate']))
    (args.input/'summary.json').write_text(json.dumps(rows,indent=2))
    cards=[]
    for r in rows:
        p=r['play_id'];d=r['displayed']
        cards.append(f'<section><h2>{p}</h2><p>{html.escape(d["shell"])} / {html.escape(d["estimate"])}</p><p>{html.escape(d["source"])}</p><a href="{p}/result.json">Analysis</a> | <a href="{p}/defense_tracking.mp4">Video</a><br><img width="960" src="{p}/post_snap_annotated.png"></section>')
    (args.input/'index.html').write_text('<!doctype html><title>All 15 play analyses</title><style>body{background:#151921;color:white;font:18px sans-serif;margin:32px}a{color:#77caff}section{margin-bottom:40px}img{max-width:100%}</style><h1>All 15 play analyses</h1><p>Provisional estimates. Tracking and film-review display sources remain separate.</p>'+''.join(cards))
    print(f'Validated {len(rows)} complete reports, videos, final frames and 2.3-second endpoints.')


if __name__=='__main__': main()

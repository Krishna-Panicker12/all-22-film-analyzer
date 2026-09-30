"""Play saved analysis videos at normal speed and hold each final frame until Q."""
import argparse
import json
from pathlib import Path
import cv2
from src.shells.display import wait_for_manual_exit, banner, display_estimate


def review(root):
    folders=sorted(p.parent for p in root.glob('*/result.json'))
    if not folders: raise ValueError(f'No completed play analyses in {root}')
    window='All-22 coverage review | Q: next play | ESC: quit'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1280,940)
    try:
        for folder in folders:
            report=json.loads((folder/'result.json').read_text())
            config=json.loads((folder/'config.json').read_text())
            cap=cv2.VideoCapture(str(folder/'defense_tracking.mp4'))
            try:
                if not cap.isOpened(): raise ValueError(f'Cannot read {folder}')
                fps=cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: raise ValueError(f'Invalid FPS in {folder}')
                last=None; advance=False
                while True:
                    ok,frame=cap.read()
                    if not ok: break
                    last=frame;cv2.imshow(window,frame)
                    key=cv2.waitKey(max(1,round(1000/fps)))&0xFF
                    if key==32:
                        while True:
                            key=cv2.waitKey(100)&0xFF
                            if key in (32,27,ord('q'),ord('Q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
                    if key==27 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: return
                    if key in (ord('q'),ord('Q')):
                        advance=True;break
                if last is None: raise ValueError(f'Empty video {folder}')
                if not advance:
                    estimate=display_estimate(report['snapshots']['post_snap']['shell'],
                        report['coverage'],config.get('review_context'),True)
                    cv2.imshow(window,banner(last[220:],estimate,report['play_id'],report['post_delay_seconds'],True))
                    if wait_for_manual_exit(window)=='quit': return
            finally:
                cap.release()
    finally:
        cv2.destroyAllWindows()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('outputs/all_plays_review_v1'))
    args=parser.parse_args()
    review(args.input)


if __name__=='__main__': main()

"""Create magnified label contact sheets and apply explicitly reviewed coordinates.

Coordinates in corrections.json are original-image pixels, never model predictions.
Backups and before/after overlays are kept in the review output folder.
"""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/annotations/game_01/pass_001"
OUTPUT = ROOT / "outputs/helmet_manual_review"


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for folder in sorted(SOURCE.iterdir()):
        path = folder / "observations.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        image = cv2.imread(str(folder / "original.png"))
        sheet = np.full((220 * ((len(data["manual_labels"])+4)//5), 960, 3), 32, np.uint8)
        for i, label in enumerate(data["manual_labels"]):
            x1, y1, x2, y2 = label["box"]
            left, top = max(0, (x1+x2)//2-32), max(0, (y1+y2)//2-32)
            crop = image[top:top+64, left:left+64].copy()
            cv2.rectangle(crop, (x1-left, y1-top), (x2-left, y2-top), (0, 200, 0), 1)
            tile = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
            row, col = divmod(i, 5)
            sheet[row*220+28:row*220+220, col*192:col*192+192] = tile
            cv2.putText(sheet, f"{i+1}: {left},{top}", (col*192+4, row*220+20),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        cv2.imwrite(str(OUTPUT / f"{folder.name}_crops.png"), sheet)
        # Sideline strips retain original coordinates in their filenames.
        for top in (0, 580):
            for left in (0, 640):
                crop = image[top:min(top+140, image.shape[0]), left:left+640]
                cv2.imwrite(str(OUTPUT / f"{folder.name}_strip_{left}_{top}.png"),
                            cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def apply(corrections):
    edits = json.loads(Path(corrections).read_text())
    for frame, labels in edits.items():
        folder = SOURCE / frame
        path = folder / "observations.json"
        data = json.loads(path.read_text())
        image = cv2.imread(str(folder / "original.png"))
        h, w = image.shape[:2]
        for box in labels:
            x1,y1,x2,y2 = box
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"Invalid box {frame}: {box}")
        backup = OUTPUT / "backup" / frame / "observations.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copyfile(path, backup)
        data["manual_labels"] = [dict(kind="helmet", box=box) for box in labels]
        data["review_complete"] = True
        data["review_note"] = "Manually reviewed from original pixels by assistant; backup in outputs/helmet_manual_review/backup."
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(path)
        for i, (x1,y1,x2,y2) in enumerate(labels, 1):
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(image, str(i), (x1,max(10,y1-2)), cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
        cv2.imwrite(str(OUTPUT / f"{frame}_after.jpg"), image)


def convert_local(path, extras):
    result = {}
    additions = json.loads(Path(extras).read_text()) if extras else {}
    for frame, local_boxes in json.loads(Path(path).read_text()).items():
        original = OUTPUT / "backup" / frame / "observations.json"
        if not original.exists():
            original = SOURCE / frame / "observations.json"
        labels = json.loads(original.read_text())["manual_labels"]
        if len(labels) != len(local_boxes):
            raise ValueError(f"Review count differs for {frame}")
        result[frame] = []
        for label, box in zip(labels, local_boxes):
            if box is None:  # Explicitly rejected shoulder/duplicate.
                continue
            x1,y1,x2,y2 = label["box"]
            left, top = max(0,(x1+x2)//2-32), max(0,(y1+y2)//2-32)
            result[frame].append([left+box[0],top+box[1],left+box[2],top+box[3]])
        result[frame].extend(additions.get(frame, []))
    output = OUTPUT / "corrections.json"
    output.write_text(json.dumps(result, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", type=Path)
    parser.add_argument("--local", type=Path, help="Convert reviewed contact-sheet coordinates and apply")
    parser.add_argument("--extras", type=Path, help="Additional original-pixel boxes")
    args = parser.parse_args()
    if args.local:
        apply(convert_local(args.local, args.extras))
    elif args.apply:
        apply(args.apply)
    else:
        prepare()

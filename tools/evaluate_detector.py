"""Evaluate a trained single-class detector against reviewed, game-separated labels."""
import argparse
import csv
import json
from pathlib import Path

import cv2

from src.detection.dataset import read_boxes, sha256, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def match_boxes(truth, predictions, threshold=.5):
    """Greedy confidence-ordered matching; each truth box can match only once.

    Predictions are [x1, y1, x2, y2, confidence]. Return matched index pairs.
    """
    remaining = set(range(len(truth)))
    pairs = []
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i][4], reverse=True):
        best = max(remaining, key=lambda j: box_iou(predictions[index], truth[j]), default=None)
        if best is not None and box_iou(predictions[index], truth[best]) >= threshold:
            pairs.append((index, best))
            remaining.remove(best)
    return pairs


def evaluate(model_path, data, output, split="val", imgsz=1280, batch=4,
             device="cpu", conf=.25, match_iou=.5, max_errors=25):
    from ultralytics import YOLO

    dataset = validate_dataset(data)
    output, model_path = Path(output).resolve(), Path(model_path).resolve()
    if split not in dataset["manifest"]["splits"]:
        raise ValueError(f"No {split} split. Export a separate test game with --test first.")
    if not model_path.is_file():
        raise ValueError(f"Model not found: {model_path}")
    if output.exists():
        raise ValueError(f"Choose a new evaluation output directory: {output}")
    model = YOLO(str(model_path))
    if model.task != "detect" or model.names != {0: dataset["target"]}:
        raise ValueError(f"Evaluation requires a trained single-class {dataset['target']} detector.")
    output.mkdir(parents=True)
    metrics = model.val(data=dataset["yaml"], split=split, imgsz=imgsz, batch=batch,
                        device=device, workers=0, conf=.001, iou=.7, plots=True,
                        project=str(output), name="metrics", exist_ok=False)
    rows, saved = [], 0
    for record in dataset["manifest"]["splits"][split]:
        image = cv2.imread(str(dataset["root"] / record["image"]))
        height, width = image.shape[:2]
        truth = read_boxes(dataset["root"] / record["label"], width, height)
        result = model.predict(image, imgsz=imgsz, device=device, conf=conf,
                               iou=.7, verbose=False)[0]
        predictions = result.boxes.data.cpu().tolist()  # xyxy, confidence, class
        pairs = match_boxes(truth, predictions, match_iou)
        tp = len(pairs)
        fp, fn = len(predictions)-tp, len(truth)-tp
        rows.append(dict(clip=record["clip"], frame=record["frame"], image=record["image"],
                         truth=len(truth), predictions=len(predictions), tp=tp, fp=fp, fn=fn))
        if (fp or fn) and saved < max_errors:
            matched_predictions = {i for i, _ in pairs}
            matched_truth = {j for _, j in pairs}
            for boxes, matched, good, bad in ((truth, matched_truth, (0, 180, 0), (0, 255, 255)),
                                              (predictions, matched_predictions, (255, 255, 0), (0, 0, 255))):
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(round, box[:4])
                    cv2.rectangle(image, (x1, y1), (x2, y2), good if i in matched else bad, 2)
            errors = output / "errors"
            errors.mkdir(exist_ok=True)
            cv2.imwrite(str(errors / f"{saved:04d}_frame_{record['frame']}.jpg"), image)
            saved += 1
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    summary = dict(model=str(model_path), model_sha256=sha256(model_path),
                   dataset=dataset["yaml"], manifest_sha256=dataset["manifest_sha256"],
                   split=split, counts=dataset["counts"][split],
                   standard_metrics={k: float(v) for k, v in metrics.results_dict.items()},
                   operating_point=dict(confidence=conf, match_iou=match_iou, nms_iou=.7,
                                        tp=tp, fp=fp, fn=fn,
                                        precision=tp/(tp+fp) if tp+fp else None,
                                        recall=tp/(tp+fn) if tp+fn else None,
                                        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None),
                   imgsz=imgsz, device=str(device), saved_error_images=saved,
                   note="Validation is used for model selection; reserve test for final evaluation. "
                        "Standard precision/recall use the library's selected threshold; "
                        "operating_point uses your fixed confidence. These are detection, not tracking metrics.",
                   error_colors="green=matched truth, yellow=missed truth, cyan=matched prediction, red=false positive")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "per_frame.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["operating_point"], indent=2))
    print(f"Evaluation saved: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/helmet_dataset/dataset.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--match-iou", type=float, default=.5)
    args = parser.parse_args()
    if args.imgsz < 32 or args.batch < 1 or not 0 < args.conf <= 1 or not 0 < args.match_iou <= 1:
        parser.error("Use imgsz >= 32, batch >= 1 and thresholds in (0, 1].")
    try:
        evaluate(args.model, args.data, args.output, args.split, args.imgsz, args.batch,
                 args.device, args.conf, args.match_iou)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

"""Fine-tune YOLO on reviewed player or helmet labels and evaluate the best checkpoint."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from src.detection.dataset import sha256, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/helmet_dataset/dataset.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device such as 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="Validate data without training")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("Use epochs >= 1, imgsz >= 32 and batch >= 1.")
    try:
        dataset = validate_dataset(args.data)
        print(f"Target: {dataset['target']}\n{json.dumps(dataset['counts'], indent=2)}")
        if args.check:
            print("Dataset checks passed. This checks data integrity, not whether there is enough variety to generalize.")
            return
        if not args.model.is_file():
            raise ValueError(f"Local pretrained model not found: {args.model}")
        output = (args.output or ROOT / "outputs/training" /
                  f"{dataset['target']}_{datetime.now():%Y%m%d_%H%M%S}").resolve()
        if output.exists():
            raise ValueError(f"Choose a new output directory: {output}")
        import ultralytics
        from ultralytics import YOLO
        from tools.evaluate_detector import evaluate

        model = YOLO(str(args.model.resolve()))
        if model.task != "detect":
            raise ValueError("Use an object detection checkpoint.")
        output.mkdir(parents=True)
        settings = dict(data=dataset["yaml"], epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch, device=args.device, seed=args.seed,
                        workers=0, deterministic=True, cache=False, amp=False,
                        patience=10, plots=True, project=str(output.parent),
                        name=output.name, exist_ok=True)
        # The run folder was reserved above; exist_ok prevents the library renaming it.
        provenance = dict(settings=settings, manifest_sha256=dataset["manifest_sha256"],
                          initial_model=str(args.model.resolve()), initial_model_sha256=sha256(args.model),
                          ultralytics=ultralytics.__version__, counts=dataset["counts"], status="started")
        report = output / "training.json"
        report.write_text(json.dumps(provenance, indent=2))
        model.train(**settings)
        best = output / "weights/best.pt"
        evaluate(best, args.data, output / "evaluation", imgsz=args.imgsz,
                 batch=args.batch, device=args.device)
        provenance["status"] = "complete"
        report.write_text(json.dumps(provenance, indent=2))
        print(f"Best weights: {best}\nNo viewer model was replaced. Review evaluation before using these weights.")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

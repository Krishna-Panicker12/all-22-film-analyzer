"""Load a pretrained person detector, shared by detection and tracking."""

from pathlib import Path

from ultralytics import YOLO


def load_detector(weights: Path, target="player"):
    """Official weights download on first use; custom models can use player."""
    if target == "helmet" and not weights.is_file():
        raise ValueError(f"Helmet-trained weights are required: {weights}. "
                         "The standard YOLO person model cannot detect helmets. "
                         "Use --label-target helmet to collect labels with the current model.")
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    person_ids = target_classes(model.names, target)
    return model, person_ids


def target_classes(names, target):
    allowed = {"helmet", "football helmet"} if target == "helmet" else {"person", "player"}
    person_ids = [key for key, name in names.items() if name.lower().replace("_", " ") in allowed]
    if not person_ids:
        raise ValueError(f"Model has no {target} class; found {list(names.values())}. "
                         "Changing a class name does not train a detector.")
    return person_ids

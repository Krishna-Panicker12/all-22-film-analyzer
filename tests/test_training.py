import tempfile
from pathlib import Path
import unittest

import test_annotations
from tools.evaluate_detector import match_boxes
from tools.prepare_dataset import export_dataset
from src.detection.dataset import validate_dataset


class TrainingTests(unittest.TestCase):
    def test_manifest_and_tampered_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = test_annotations.AnnotationTests()
            train = fixture.record(root, "train", "game_a")
            val = fixture.record(root, "val", "game_b")
            export_dataset(train, val, root / "dataset")
            config = root / "dataset/dataset.yaml"
            self.assertEqual(validate_dataset(config)["counts"]["val"]["objects"], 1)
            (root / "dataset/labels/val/000000.txt").write_text("")
            with self.assertRaisesRegex(ValueError, "changed after export"):
                validate_dataset(config)

    def test_duplicate_image_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = test_annotations.AnnotationTests()
            train = fixture.record(root, "train", "game_a")
            val = fixture.record(root, "val", "game_b")
            (val / "snapshot/original.png").write_bytes((train / "snapshot/original.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "Duplicate image"):
                export_dataset(train, val, root / "dataset")

    def test_one_truth_cannot_match_two_predictions(self):
        truth = [[0, 0, 10, 10]]
        self.assertEqual(match_boxes(truth, [[0, 0, 10, 10, .4], [0, 0, 10, 10, .9]]), [(1, 0)])
        self.assertEqual(match_boxes(truth, []), [])
        self.assertEqual(match_boxes(truth, [[20, 20, 30, 30, .9]]), [])

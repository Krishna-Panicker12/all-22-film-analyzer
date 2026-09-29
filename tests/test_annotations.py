"""Training export must never treat predictions or partial corrections as truth."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.prepare_dataset import export_dataset, read_labels


class AnnotationTests(unittest.TestCase):
    def record(self, root, split, game, complete=True, kind="player"):
        folder = root / split / "snapshot"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "original.png"), np.full((100, 200, 3), len(split), np.uint8))
        data = dict(clip=str(root / game / "clip.mp4"), frame=10,
                    review_complete=complete,
                    raw=[dict(box=[0, 0, 199, 99], confidence=.9)],
                    manual_labels=[dict(kind=kind, box=[20, 10, 60, 50])])
        (folder / "observations.json").write_text(json.dumps(data))
        return root / split

    def test_export_uses_only_manual_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.record(root, "train", "game_a")
            val = self.record(root, "val", "game_b")
            export_dataset(train, val, root / "dataset")
            text = (root / "dataset/labels/train/000000.txt").read_text()
            self.assertEqual(text, "0 0.200000 0.300000 0.200000 0.400000")

    def test_partial_and_uncertain_are_not_training_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = self.record(root, "partial", "game_a", complete=False)
            self.assertEqual(read_labels(partial), [])
            uncertain = self.record(root, "uncertain", "game_b", kind="uncertain")
            with self.assertRaises(ValueError):
                read_labels(uncertain)

    def test_same_game_split_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self.record(root, "train", "game_a")
            val = self.record(root, "val", "game_a")
            with self.assertRaises(ValueError):
                export_dataset(train, val, root / "dataset")
            self.assertFalse((root / "dataset").exists())

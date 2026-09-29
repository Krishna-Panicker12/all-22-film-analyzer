import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from tools.label_helmets import HelmetLabeler, save_frame
from tools.prepare_dataset import read_labels
from src.detection.annotations import FrameLabels
from src.detection.detection import target_classes, load_detector
from src.tracking.field import FieldFilter
from src.tracking.tracking import (Observation, PlayerTracker, associate_helmets_with_bodies,
                                   continue_tracks_with_bodies, on_field)


class HelmetTests(unittest.TestCase):
    def test_body_support_bridges_only_unambiguous_short_occlusion(self):
        memory={7:((10,10,40,90),4)}
        support=continue_tracks_with_bodies([],[(11,10,41,90)],set(),memory,5)
        self.assertEqual(len(support),1)
        self.assertEqual((support[0].track_id,support[0].source),(7,"body_support"))
        self.assertEqual(support[0].ground_anchor,(26,90))
        crowded={7:((10,10,40,90),4),8:((12,10,42,90),4)}
        self.assertEqual(continue_tracks_with_bodies([],[(11,10,41,90)],set(),crowded,5),[])
        self.assertEqual(continue_tracks_with_bodies([],[(11,10,41,90)],set(),memory,8),[])
        self.assertEqual(continue_tracks_with_bodies([],[(11,10,41,90)],set(),memory,5,set()),[])

    def test_body_support_matches_motion_across_longer_helmet_gap(self):
        memory={7:((10,10,40,90),4)}
        support=continue_tracks_with_bodies([],[(30,10,60,90)],set(),memory,8,
                                            max_gap=5)
        self.assertEqual(len(support),1)
        self.assertEqual(support[0].track_id,7)
        self.assertEqual(support[0].body_index,0)
        crowded={7:((10,10,40,90),4),8:((12,10,42,90),4)}
        self.assertEqual(continue_tracks_with_bodies([],[(30,10,60,90)],set(),
                                                     crowded,8,max_gap=5),[])

    def test_multiple_body_supports_keep_their_own_boxes(self):
        memory={7:((10,10,40,90),4),8:((70,10,100,90),4)}
        bodies=[(12,10,42,90),(72,10,102,90)]
        supports=continue_tracks_with_bodies([],bodies,set(),memory,5)
        self.assertEqual({item.track_id:item.box for item in supports},
                         {7:bodies[0],8:bodies[1]})

    def test_body_only_identity_adopts_returning_helmet_and_keeps_trail(self):
        def result(rows):
            return SimpleNamespace(boxes=Boxes(np.array(rows, np.float32).reshape(-1, 6),
                                                (120, 120)))
        helmet_rows = [[], [], [[20,10,30,20,.9,0]], [[21,10,31,20,.9,0]]]
        body_rows = [[[10,10,40,90,.9,0]], [[10,10,40,90,.9,0]],
                     [[11,10,41,90,.9,0]], [[12,10,42,90,.9,0]]]
        helmet = SimpleNamespace(predict=lambda *a, **kw: [result(next(helmet_rows_iter))])
        body = SimpleNamespace(predict=lambda *a, **kw: [result(next(body_rows_iter))])
        helmet_rows_iter, body_rows_iter = iter(helmet_rows), iter(body_rows)
        tracker = PlayerTracker(helmet, [0], tracker="bytetrack.yaml", target="helmet",
                                body_model=body, body_classes=[0])
        frame = np.zeros((120, 120, 3), np.uint8)
        observed = [tracker.update(frame) for _ in range(4)]
        identities = [items[0].track_id for items in observed[1:]]
        self.assertEqual(len(set(identities)), 1)
        self.assertGreaterEqual(identities[0], 1_000_000)
        self.assertEqual(len(tracker.history[identities[0]][0]), 4)
        self.assertEqual(observed[-1][0].source, "helmet")
    def test_delete_saved_box_at_zoom_marks_frame_incomplete(self):
        labels = FrameLabels("helmet")
        labels.boxes = [dict(kind="helmet", box=[300, 150, 310, 160])]
        labels.complete = True
        view = HelmetLabeler(np.zeros((300, 600, 3), np.uint8), labels)
        view.pointer = (300, 150)
        view.toggle_zoom()
        view.mouse(cv2.EVENT_RBUTTONDOWN, 315, 165 + view.header, 0, None)
        self.assertEqual(labels.boxes, [])
        self.assertFalse(labels.complete)

    def test_person_model_cannot_silently_become_helmet_model(self):
        with self.assertRaises(ValueError):
            target_classes({0: "person", 1: "bicycle"}, "helmet")
        self.assertEqual(target_classes({0: "Helmet"}, "helmet"), [0])
        with self.assertRaises(ValueError):
            load_detector(Path("missing-football-helmet-weights.pt"), "helmet")

    def test_helmet_requires_unambiguous_body_for_ground_anchor(self):
        one = Observation(1, .9, (20, 10, 30, 20), "helmet")
        associate_helmets_with_bodies([one], [(10, 5, 40, 90)])
        self.assertEqual(one.ground_anchor, (25, 90))
        pair = [Observation(i, .9, box, "helmet") for i, box in
                enumerate([(20, 10, 30, 20), (32, 10, 42, 20)])]
        associate_helmets_with_bodies(pair, [(10, 5, 50, 90)])
        self.assertTrue(all(item.ground_anchor is None for item in pair))
        field = FieldFilter(False)
        field.select([(100, 100), (200, 100), (200, 200), (100, 200)])
        self.assertFalse(on_field(one, field))
        self.assertFalse(on_field(pair[0], field))

    def test_unanchored_sideline_heads_do_not_bypass_field(self):
        field = FieldFilter(False)
        field.select([(0, 0), (400, 0), (400, 300), (0, 300)])
        inside = Observation(1, .9, (190, 140, 200, 150), "helmet")
        sideline = Observation(2, .9, (190, 285, 200, 295), "helmet")
        self.assertTrue(on_field(inside, field))
        self.assertFalse(on_field(sideline, field))
        sideline.ground_anchor = (195, 299)
        self.assertTrue(on_field(sideline, field), "Measured feet override conservative helmet margin")
        sideline.ground_anchor = (195, 310)
        self.assertFalse(on_field(sideline, field))
        field.clear()
        self.assertTrue(on_field(inside, field))

    def test_helmet_pipeline_uses_body_footpoint_for_trail(self):
        def result(rows):
            return SimpleNamespace(boxes=Boxes(np.array(rows, np.float32).reshape(-1, 6), (100, 100)))
        model = SimpleNamespace(predict=lambda *a, **kw: [result([[20, 10, 30, 20, .9, 0]])])
        bodies = SimpleNamespace(predict=lambda *a, **kw: [result([[10, 5, 40, 90, .9, 0]])])
        tracker = PlayerTracker(model, [0], tracker="bytetrack.yaml", target="helmet",
                                body_model=bodies, body_classes=[0], confirmation_frames=1)
        observed = tracker.update(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(observed[0].source, "helmet")
        self.assertEqual(observed[0].ground_anchor, (25, 90))
        self.assertEqual(tracker.history[observed[0].track_id][0][-1], (25, 90))

    def test_helmet_without_body_does_not_switch_trail_to_head(self):
        def result(rows):
            return SimpleNamespace(boxes=Boxes(np.array(rows, np.float32).reshape(-1, 6),
                                                (100, 100)))
        helmet_rows=iter([[[20,10,30,20,.9,0]], [[21,10,31,20,.9,0]]])
        body_rows=iter([[[10,5,40,90,.9,0]], []])
        helmet=SimpleNamespace(predict=lambda *a,**kw:[result(next(helmet_rows))])
        body=SimpleNamespace(predict=lambda *a,**kw:[result(next(body_rows))])
        tracker=PlayerTracker(helmet,[0],tracker="bytetrack.yaml",target="helmet",
                              body_model=body,body_classes=[0],confirmation_frames=1)
        frame=np.zeros((100,100,3),np.uint8)
        first=tracker.update(frame)[0]
        before=list(tracker.history[first.track_id][0])
        tracker.update(frame)
        self.assertEqual(list(tracker.history[first.track_id][0]),before)

    def test_zoom_labels_are_saved_in_original_coordinates(self):
        frame = np.zeros((300, 600, 3), np.uint8)
        labels = FrameLabels("helmet")
        view = HelmetLabeler(frame, labels)
        view.pointer = (300, 150)
        view.toggle_zoom()
        self.assertEqual(view.original_point(300, 150), (300, 150))
        view.mouse(cv2.EVENT_LBUTTONDOWN, 300, 150 + view.header, 0, None)
        view.mouse(cv2.EVENT_LBUTTONUP, 330, 180 + view.header, 0, None)
        self.assertEqual(labels.boxes[0]["box"], [300, 150, 310, 160])
        labels.complete = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_frame(root / "frame", Path("game/clip.mp4"), frame, 148, 30, labels)
            saved = json.loads((root / "frame/observations.json").read_text())
            self.assertEqual(saved["review_target"], "helmet")
            self.assertEqual(read_labels(root, "player"), [])
            self.assertEqual(len(read_labels(root, "helmet")), 1)

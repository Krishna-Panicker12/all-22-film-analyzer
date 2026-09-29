"""Exercise the real tracker with synthetic detections; no weights needed."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from main import ROOT, clip_context
from src.tracking.tracking import (PlayerTracker, Observation, limit_roster,
                                   canonicalize_track_ids, suppress_duplicate_observations,
                                   inferred_head_box, route_role_eligible, smooth_anchor,
                                   unique_body_indices)
from src.tracking.teams import JerseyTeams
from src.tracking.field import FieldFilter, detect_sidelines
from src.viewer import ReviewControls


class TrackingTests(unittest.TestCase):
    def test_fast_small_helmet_keeps_identity_and_original_box_size(self):
        rows = [[[20+8*i,20,30+8*i,30,.9,0]] for i in range(8)]
        results = iter(SimpleNamespace(boxes=Boxes(np.array(row,np.float32),(100,100)))
                       for row in rows)
        model = SimpleNamespace(predict=lambda *args,**kwargs: [next(results)])
        tracker = PlayerTracker(model,[0],tracker="bytetrack.yaml",target="helmet",helmet_match_scale=2)
        tracked = [tracker.update(np.zeros((100,100,3),np.uint8)) for _ in rows]
        self.assertTrue(all(len(items) == 1 for items in tracked[1:]))
        self.assertEqual(len({items[0].track_id for items in tracked[1:]}), 1)
        for items in tracked[1:]:
            self.assertAlmostEqual(items[0].box[2]-items[0].box[0], 10, places=3)

    def test_weak_helmet_can_support_but_not_start_identity(self):
        frames = [[[20,20,30,30,.15,0]], [[20,20,30,30,.9,0]],
                  [[20,20,30,30,.9,0]], [[20,20,30,30,.9,0]],
                  [[20,20,30,30,.15,0]]]
        results = iter(SimpleNamespace(boxes=Boxes(np.array(rows,np.float32),(100,100)))
                       for rows in frames)
        model = SimpleNamespace(predict=lambda *args,**kwargs: [next(results)])
        tracker = PlayerTracker(model,[0],tracker="bytetrack.yaml",target="helmet",helmet_match_scale=2)
        frame = np.zeros((100,100,3),np.uint8)
        self.assertLessEqual(tracker.options["conf"], .1)
        self.assertEqual(tracker.update(frame), [])
        tracker.update(frame)
        tracker.update(frame)
        strong = tracker.update(frame)
        self.assertEqual(len(strong), 1)
        weak = tracker.update(frame)
        self.assertEqual([p.track_id for p in weak], [strong[0].track_id])

    def test_duplicate_helmet_detections_do_not_create_competing_tracks(self):
        from src.tracking.tracking import unique_helmet_indices
        boxes = np.array([[375,584,385,596], [374,585,384,597], [400,584,410,596]])
        self.assertEqual(unique_helmet_indices(boxes, np.array([.57,.33,.8])), [2,0])

    def test_duplicate_body_detections_do_not_create_competing_ids(self):
        boxes = np.array([[10,10,40,90], [11,11,41,91], [70,10,100,90]])
        self.assertEqual(unique_body_indices(boxes, np.array([.8,.7,.9])), [2,0])

    def test_longer_reconnection_requires_matching_appearance(self):
        histogram = np.ones((12, 8), np.float32) / 96
        for color, expected in (("red", 1), ("white", 9), (None, 9)):
            item = Observation(9, .8, (21,20,31,30), "helmet")
            evidence = {9: (color, histogram)} if color else {}
            canonicalize_track_ids([item], {}, {1: ((20,20,30,30), 1)}, 8,
                                   30, evidence, {1: ("red", histogram)})
            self.assertEqual(item.track_id, expected)
        item = Observation(9, .8, (21,20,31,30), "helmet")
        canonicalize_track_ids([item], {}, {1: ((20,20,30,30), 1)}, 40,
                               30, {9: ("red", histogram)}, {1: ("red", histogram)})
        self.assertEqual(item.track_id, 9)

    def test_appearance_does_not_override_crowded_matches(self):
        histogram = np.ones((12, 8), np.float32) / 96
        item = Observation(9, .8, (21,20,31,30), "helmet")
        memory = {1: ((20,20,30,30), 1), 2: ((22,20,32,30), 1)}
        canonicalize_track_ids([item], {}, memory, 8, 30,
                               {9: ("red", histogram)},
                               {1: ("red", histogram), 2: ("red", histogram)})
        self.assertEqual(item.track_id, 9)

    def test_camera_moves_id_memory_and_preserves_trail_history(self):
        from collections import deque
        tracker = self.tracker([])
        tracker.canonical_memory[1] = ((10,20,20,30), 3)
        tracker.history[1] = (deque([(10,20), None, (15,20)], maxlen=120), 3)
        tracker.route_motion[1] = dict(start=(10,20), last=(15,20), height=10,
                           travel=5, first_frame=2)
        transform = np.array([[1,0,8], [0,1,-2]], np.float32)
        tracker.move_identity_memory(transform)
        tracker.move_history(transform)
        self.assertEqual(tracker.canonical_memory[1], ((18,18,28,28), 3))
        self.assertEqual(tracker.route_motion[1]["start"], (18.0, 18.0))
        self.assertEqual(tracker.route_motion[1]["last"], (23.0, 18.0))
        self.assertIsNone(tracker.history[1][0][1])
        tracker.body_memory = {1: ((10,20,20,30), 3),
                       1_000_001: ((40,20,60,80), 3)}
        tracker.move_identity_memory(None)
        self.assertEqual(tracker.canonical_memory, {})
        self.assertEqual(set(tracker.body_memory), {1_000_001})

    def test_camera_motion_without_selected_field(self):
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 255, (240,400,3), dtype=np.uint8)
        field = FieldFilter(False)
        field.advance(frame)
        transform = np.float32([[1,0,4],[0,1,3]])
        field.advance(cv2.warpAffine(frame, transform, (400,240)))
        self.assertIsNone(field.points)
        self.assertIsNotNone(field.motion)
        np.testing.assert_allclose(field.motion[:,2], [4,3], atol=.5)

    def test_trails_follow_camera_and_survive_missing_motion(self):
        from collections import deque
        tracker = self.tracker([])
        tracker.history[1] = (deque([(10, 20), (15, 20)], maxlen=120), 3)
        tracker.move_history(np.array([[1, 0, 8], [0, 1, -2]], np.float32))
        np.testing.assert_allclose(list(tracker.history[1][0]), [(18, 18), (23, 18)])
        tracker.move_history(None)
        np.testing.assert_allclose(list(tracker.history[1][0]), [(18, 18), (23, 18)])

    def test_trail_retains_more_than_120_points(self):
        from collections import deque
        tracker = self.tracker([])
        tracker.history[1] = (deque(((index, index) for index in range(240)), maxlen=None), 240)
        tracker.move_history(None)
        self.assertEqual(len(tracker.history[1][0]), 240)
        self.assertEqual(tracker.history[1][0][0], (0, 0))

    def test_feet_anchor_smoothing_dampens_detector_jitter(self):
        first = smooth_anchor(None, (100, 500))
        second = smooth_anchor(first, (80, 480))
        self.assertEqual(first, (100.0, 500.0))
        self.assertEqual(second, (93.0, 493.0))

    def test_body_fallback_draw_box_is_a_head_proxy(self):
        box = inferred_head_box((10, 20, 50, 100))
        self.assertEqual(box, (22.0, 20, 38.0, 32.0))

    def test_motion_filter_keeps_releasing_players_and_rejects_static_trench(self):
        mobile = dict(start=(0, 0), last=(120, 0), height=60, travel=140)
        nearby = [dict(start=(2, 3), last=(3, 4), height=60, travel=2),
                  dict(start=(3, -2), last=(4, -1), height=60, travel=2),
                  dict(start=(-4, 1), last=(-3, 2), height=60, travel=2)]
        self.assertTrue(route_role_eligible(mobile, [mobile, *nearby]))
        static = dict(start=(0, 0), last=(20, 0), height=60, travel=22)
        self.assertFalse(route_role_eligible(static, [static, *nearby]))

    def test_manual_trail_scope_keeps_all_boxes_and_only_selected_lines(self):
        tracker = self.tracker([])
        tracker.target = "helmet"
        tracker.trail_scope = "manual"
        tracker.confirmed_ids.update((1, 2))
        tracker.manual_trail_ids.add(1)
        review = ReviewControls(tracker, FieldFilter())
        review.observations = [Observation(1,.9,(10,10,20,20)),
                              Observation(2,.9,(30,10,40,20))]
        self.assertEqual([item.track_id for item in review.visible()], [1, 2])
        self.assertEqual(tracker.trail_ids_to_draw(), {1})
        tracker.trail_scope = "eligible"
        tracker.route_eligible_ids.add(2)
        self.assertEqual(tracker.trail_ids_to_draw(), {2})
        tracker.trail_scope = "all"
        self.assertEqual(tracker.trail_ids_to_draw(), {1, 2})

    def test_helmet_display_does_not_apply_roster_cap(self):
        tracker = self.tracker([])
        tracker.target = "helmet"
        review = ReviewControls(tracker, FieldFilter())
        review.observations = [Observation(identity,.9,(identity*3,10,identity*3+2,20),"helmet")
                               for identity in range(23)]
        self.assertEqual(len(review.visible()), 23)

    def test_mouse_selects_trail_without_hiding_other_boxes(self):
        tracker = self.tracker([])
        tracker.target = "helmet"
        review = ReviewControls(tracker, FieldFilter())
        review.observations = [Observation(1,.9,(10,10,20,20)),
                              Observation(2,.9,(30,10,40,20))]
        review.key(ord('m'))
        review.mouse(cv2.EVENT_LBUTTONDOWN, 15, 15+review.header_height, 0, None)
        review.mouse(cv2.EVENT_LBUTTONDOWN, 35, 15+review.header_height, 0, None)
        self.assertEqual(tracker.manual_trail_ids, {1, 2})
        self.assertTrue(review.selecting_trails)
        review.key(ord('m'))
        self.assertFalse(review.selecting_trails)
        self.assertEqual([item.track_id for item in review.visible()], [1, 2])

    def test_tracker_reset_clears_formation_and_identity_state(self):
        tracker = self.tracker([])
        tracker.target = "helmet"
        tracker.teams = JerseyTeams()
        tracker.confirmed_ids.add(1)
        tracker.id_aliases[9] = 1
        tracker.canonical_memory[1] = ((10,10,20,20), 3)
        tracker.manual_trail_ids.add(1)
        tracker.formation_trail_ids.add(1)
        tracker.formation_roles[1] = "WR"
        tracker.teams.assignments[1] = "white"
        tracker.reset()
        self.assertEqual(tracker.confirmed_ids, set())
        self.assertEqual(tracker.id_aliases, {})
        self.assertEqual(tracker.canonical_memory, {})
        self.assertEqual(tracker.manual_trail_ids, set())
        self.assertEqual(tracker.formation_trail_ids, set())
        self.assertEqual(tracker.formation_roles, {})
        self.assertEqual(tracker.teams.assignments, {})

    def test_confirmed_trail_draws_while_player_is_missing(self):
        from collections import deque
        tracker = self.tracker([])
        tracker.confirmed_ids.add(1)
        tracker.missing = [1]
        tracker.history[1] = (deque([(10, 10), (20, 20)]), 2)
        image = tracker.draw(np.zeros((40, 40, 3), np.uint8), [], trails=True)
        self.assertGreater(np.count_nonzero(image), 0)

    def test_new_engine_id_reconnects_only_when_spatial_match_is_clear(self):
        aliases={1:1}
        memory={1:((20,20,30,30),5)}
        item=Observation(9,.8,(21,20,31,30),"helmet")
        canonicalize_track_ids([item],aliases,memory,6)
        self.assertEqual(item.track_id,1)
        crowded={1:((20,20,30,30),5),2:((23,20,33,30),5)}
        aliases={}
        item=Observation(9,.8,(22,20,32,30),"helmet")
        canonicalize_track_ids([item],aliases,crowded,6)
        self.assertEqual(item.track_id,9)
    def test_roster_limit_only_removes_lowest_evidence(self):
        rows=[Observation(i,.9,(i*2,0,i*2+1,1),"helmet",stable_frames=5)
              for i in range(22)]
        false=Observation(99,.3,(100,100,110,110),"helmet",team="unknown",stable_frames=2)
        selected=limit_roster(rows+[false])
        self.assertEqual(len(selected),22)
        self.assertNotIn(false,selected)
        self.assertEqual(limit_roster(rows[:5]),rows[:5])
    def test_duplicate_tracks_are_suppressed_and_existing_id_wins(self):
        rows = [Observation(9,.95,(20,20,40,40),"helmet"),
                Observation(2,.60,(21,20,41,40),"helmet"),
                Observation(3,.80,(50,20,70,40),"helmet")]
        kept = suppress_duplicate_observations(rows,{2:(8,10),9:(1,10),3:(2,10)})
        self.assertEqual({item.track_id for item in kept},{2,3})

    def test_helmet_ids_require_two_consecutive_frames(self):
        box = [20,20,40,40,.9,0]
        results = iter(SimpleNamespace(boxes=Boxes(np.array([box],np.float32),(100,100)))
                       for _ in range(2))
        model = SimpleNamespace(predict=lambda *a,**kw:[next(results)])
        tracker = PlayerTracker(model,[0],tracker="bytetrack.yaml",target="helmet")
        frame = np.zeros((100,100,3),np.uint8)
        self.assertEqual(tracker.update(frame),[])
        observed = tracker.update(frame)
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0].stable_frames,2)

    def test_tentative_and_duplicate_ids_never_appear_missing(self):
        first=[[20,20,40,40,.9,0],[21,20,41,40,.7,0]]
        results=iter(SimpleNamespace(boxes=Boxes(np.array(rows,np.float32).reshape(-1,6),(100,100)))
                     for rows in (first,first,[]))
        model=SimpleNamespace(predict=lambda *a,**kw:[next(results)])
        tracker=PlayerTracker(model,[0],tracker="bytetrack.yaml",target="helmet")
        frame=np.zeros((100,100,3),np.uint8)
        tracker.update(frame)
        confirmed=tracker.update(frame)
        self.assertEqual(len(confirmed),1)
        identity=confirmed[0].track_id
        tracker.update(frame)
        self.assertEqual(tracker.missing,[identity])

    def tracker(self, detections):
        results = iter(SimpleNamespace(boxes=Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6),
                                                  (200, 200))) for rows in detections)
        model = SimpleNamespace(predict=lambda *args, **kwargs: [next(results)])
        return PlayerTracker(model, [0], tracker="bytetrack.yaml")

    def test_metadata(self):
        context, offset = clip_context(ROOT / "data/plays/game_01/run_001.mp4")
        self.assertEqual(offset, 2)
        self.assertIn("Down 1 & 10", context)
        self.assertEqual(clip_context(Path("unknown.mp4")), ("unknown.mp4", 0))

    def test_occlusion_preserves_id_but_has_no_observed_box(self):
        box = [40, 40, 80, 100, 0.9, 0]
        tracker = self.tracker([[box], [box], [], [box]])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        identity = tracker.update(frame)[0].track_id
        self.assertEqual(tracker.update(frame)[0].track_id, identity)
        self.assertEqual(tracker.update(frame), [])
        self.assertIn(identity, tracker.missing)
        self.assertEqual(tracker.raw, [])
        self.assertEqual(tracker.update(frame)[0].track_id, identity)
        self.assertEqual(len(tracker.history[identity][0]), 3)
        self.assertTrue(all(point is not None for point in tracker.history[identity][0]))

    def test_field_filters_view_without_starving_tracker(self):
        tracker = self.tracker([[[20, 10, 40, 50, .9, 0], [120, 10, 150, 50, .9, 0]]])
        field = FieldFilter()
        field.select([(0, 0), (100, 0), (100, 100), (0, 100)])
        observed = tracker.update(np.zeros((200, 200, 3), np.uint8), field)
        self.assertEqual(len(tracker.raw), 2)
        self.assertEqual(len(observed), 2)
        review = ReviewControls(tracker, field)
        review.observations = observed
        self.assertEqual(len(review.visible()), 1)

    def test_exclusion_survives_boundary_changes_longer_than_track_buffer(self):
        box = [20, 10, 40, 50, .9, 0]
        tracker = self.tracker([[box]] * 40)
        frame = np.zeros((200, 200, 3), np.uint8)
        field = FieldFilter(automatic=False)
        review = ReviewControls(tracker, field)
        review.observations = tracker.update(frame, field)
        identity = review.observations[0].track_id
        tracker.excluded.add(identity)
        field.select([(100, 0), (199, 0), (199, 150), (100, 150)])
        for _ in range(35):
            review.observations = tracker.update(frame, field)
        field.select([(0, 0), (199, 0), (199, 199), (0, 199)])
        review.observations = tracker.update(frame, field)
        self.assertEqual(review.observations[0].track_id, identity)
        self.assertEqual(review.visible(), [])
        review.key(ord('u'))
        self.assertEqual(len(review.visible()), 1)

    def test_automatic_field_requires_both_sidelines_and_manual_override(self):
        frame = np.full((240, 400, 3), 65, np.uint8)
        frame[45:190] = (50, 140, 70)
        cv2.line(frame, (0, 45), (399, 45), (220, 220, 220), 3)
        cv2.line(frame, (0, 190), (399, 190), (220, 220, 220), 3)
        self.assertEqual(set(detect_sidelines(frame)), {"far", "near"})
        field = FieldFilter()
        field.advance(frame)
        self.assertIsNone(field.points)
        field.advance(frame)
        self.assertIsNotNone(field.points)
        self.assertFalse(field.contains((10, 195, 30, 230)))
        one_side = frame.copy()
        one_side[150:] = (50, 140, 70)
        self.assertNotEqual(set(detect_sidelines(one_side)), {"far", "near"})
        field.select([(50, 50), (350, 50), (350, 170), (50, 170)])
        self.assertFalse(field.automatic)

    def test_labels_are_separate_and_reset_on_next_frame(self):
        from src.detection.annotations import FrameLabels
        labels = FrameLabels()
        labels.mouse(cv2.EVENT_LBUTTONDOWN, 80, 90)
        labels.mouse(cv2.EVENT_LBUTTONUP, 20, 30)
        self.assertEqual(labels.boxes, [dict(kind="player", box=[20, 30, 80, 90])])
        labels.complete = True
        labels.new_frame()
        self.assertEqual(labels.boxes, [])
        self.assertFalse(labels.complete)

    def test_review_redraw_and_snapshot_do_not_advance_tracker(self):
        tracker = self.tracker([[[20, 10, 40, 50, .9, 0]]])
        frame = np.zeros((200, 200, 3), np.uint8)
        review = ReviewControls(tracker, FieldFilter())
        review.observations = tracker.update(frame)
        review.key(ord('d'))
        review.render(frame, "test", 135, 30)
        review.key(ord('d'))
        review.mouse(cv2.EVENT_RBUTTONDOWN, 30, 30 + review.header_height, 0, None)
        self.assertEqual(review.visible(), [])
        review.key(ord('u'))
        self.assertEqual(len(review.visible()), 1)
        with tempfile.TemporaryDirectory() as directory:
            folder = review.snapshot(Path(directory), Path("clip.mp4"), frame, "test", 135, 30)
            self.assertEqual(json.loads((folder / "observations.json").read_text())["frame"], 135)
            self.assertEqual(cv2.imread(str(folder / "original.png")).shape, frame.shape)
        self.assertEqual(tracker.frame_index, 1)

    def test_field_follows_translation_and_fails_open_without_features(self):
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        field = FieldFilter()
        field.advance(frame)
        points = np.array([(20, 20), (180, 20), (180, 180), (20, 180)], dtype=np.float32)
        field.select(points)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 3]]), (200, 200))
        field.advance(moved)
        np.testing.assert_allclose(field.points, points + [4, 3], atol=1)
        field.previous = np.zeros((200, 200), np.uint8)
        field.advance(np.zeros_like(frame))
        self.assertIsNone(field.points)
        self.assertIn("LOST", field.message)

    def test_invalid_boundary_and_draw_preserve_source(self):
        with self.assertRaises(ValueError):
            FieldFilter().select([(1, 1), (2, 2)])
        tracker = self.tracker([])
        frame = np.zeros((200, 200, 3), np.uint8)
        self.assertTrue(tracker.draw(frame, [Observation(None, .2, (20, 20, 40, 60))]).any())
        self.assertFalse(frame.any())


if __name__ == "__main__":
    unittest.main()

import math
import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ultralytics.engine.results import Boxes

from src.shells.baseline import analyze_play, analysis_window, classify_shell
from src.shells.defense import (DefensiveDetector, classify_corner_movement,
                                coverage_hypotheses, load_coverage_playbook,
                                load_coverage_reference,
                                shell_from_snapshot,
                                zone_bubble_polygons)
from src.shells.run_defense import (ROOT as PROJECT_ROOT, read_config, run as run_defense,
                                   run_batch)
from src.tracking.tracking import PlayerTracker


def points(*pairs):
    return [dict(lateral=x, depth=y) for x, y in pairs]


class ShellTests(unittest.TestCase):
    def test_defense_config_and_default_mode(self):
        config = read_config(PROJECT_ROOT / "data/metadata/defense_pass_001.json")
        self.assertEqual(config["play_id"], "pass_001")
        self.assertEqual(config["start_frame"], 240)
        self.assertEqual(inspect.signature(run_defense).parameters["target"].default, "hybrid")

    def test_batch_runs_reviewed_configs_in_sorted_order(self):
        config = {"video":"data/plays/game_01/pass_001.mp4","defense":"KC",
                  "defense_color":"red","start_frame":1,"pre_frame":2,"snap_frame":3}
        config["geometry_at_pre"] = {"pixels_per_yard":{"depth":20,"lateral":12}}
        with TemporaryDirectory() as directory:
            folder = PROJECT_ROOT / directory
            output_root = Path(directory)/"results"
            (output_root/"pass_001").mkdir(parents=True)
            for play_id in ("pass_002", "pass_001"):
                (folder/f"defense_{play_id}.json").write_text(
                    json.dumps({**config,"play_id":play_id}),encoding="utf-8")
            with patch("src.shells.run_defense.run", side_effect=lambda cfg, output, *a:
                       {"play_id":cfg["play_id"],"output":str(output)}) as execute:
                reports = run_batch(folder, output_root, show=True)
            self.assertEqual([report["play_id"] for report in reports],
                             ["pass_001","pass_002"])
            self.assertTrue(reports[0]["output"].endswith("pass_001_run002"))
            self.assertEqual(execute.call_count, 2)

    def test_defense_filtered_boxes_can_flow_into_player_tracker(self):
        class FakeModel:
            def predict(self, frame, **options):
                boxes = Boxes(np.array([[10, 5, 40, 95, .9, 0]], np.float32),
                              frame.shape[:2])
                return [SimpleNamespace(boxes=boxes)]

        frame = np.zeros((100, 100, 3), np.uint8)
        frame[:] = (0, 0, 255)
        detector = DefensiveDetector(FakeModel(), "red")
        detector.polygon = np.array([[0,0],[99,0],[99,99],[0,99]], np.float32)
        tracker = PlayerTracker(detector, [0], tracker="bytetrack.yaml", target="player")
        observations = tracker.update(frame)
        self.assertEqual(len(observations), 1)
        self.assertEqual(detector.counts["defense"], 1)

    def test_shells_and_wide_corners(self):
        for defenders, expected in [
            (points((0, 4), (24, 15)), "0_high"),
            (points((0, 14), (-24, 15)), "1_high"),
            (points((-8, 14), (8, 14)), "2_high"),
            (points((-8, 14), (0, 14), (8, 14)), "unknown"),
            (points((8, 14), (12, 14)), "unknown"),
            (points((0, 10)), "unknown"),
        ]:
            with self.subTest(expected=expected, defenders=defenders):
                self.assertEqual(classify_shell(defenders, complete=True).shell, expected)

    def test_snapshot_reports_high_safety_count_and_ids(self):
        geometry = dict(los_center=[0,0], depth_axis=[1,0], lateral_axis=[0,1],
                        thresholds=dict(depth_threshold=10,lateral_limit=18,
                                        middle_limit=6,margin=1))
        rows = [dict(track_id=1,anchor_kind="feet",reference_point=[14,0]),
            dict(track_id=2,anchor_kind="feet",reference_point=[15,8]),
            dict(track_id=3,anchor_kind="feet",reference_point=[15,-20])]
        result = shell_from_snapshot(rows,geometry,True,True)
        self.assertEqual(result["high_safety_count"],2)
        self.assertEqual(result["high_safety_ids"],[1,2])

    def test_corner_majority_classifies_zone_or_man(self):
        geometry = dict(los_center=[0,0],depth_axis=[1,0],lateral_axis=[0,1],
                        thresholds=dict(depth_threshold=20,lateral_limit=15,
                                        middle_limit=5,margin=1),
                        corner_lateral_min=4,corner_depth_margin=5)
        before=[dict(track_id=1,anchor_kind="feet",reference_point=[5,-8]),
                dict(track_id=2,anchor_kind="feet",reference_point=[5,8])]
        zone_after=[dict(track_id=1,anchor_kind="feet",reference_point=[7,-8]),
                    dict(track_id=2,anchor_kind="feet",reference_point=[6,8])]
        zone=classify_corner_movement(before,zone_after,geometry,
                                      dict(depth=1,lateral=1),radius_yards=6)
        self.assertEqual(zone["classification"],"zone_like")
        self.assertEqual(set(zone["bubble_ids"]),{1,2})
        man_after=[dict(track_id=1,anchor_kind="feet",reference_point=[15,-8]),
                   dict(track_id=2,anchor_kind="feet",reference_point=[16,8])]
        man=classify_corner_movement(before,man_after,geometry,
                                     dict(depth=1,lateral=1),radius_yards=6)
        self.assertEqual(man["classification"],"man_like")

    def test_corner_selection_uses_outermost_shallow_defenders(self):
        geometry = dict(los_center=[0,0],depth_axis=[1,0],lateral_axis=[0,1],
                        thresholds=dict(depth_threshold=20,lateral_limit=15,
                                        middle_limit=5,margin=1),
                        corner_lateral_min=4,corner_lateral_limit=30,
                        corner_depth_margin=5)
        before=[dict(track_id=1,anchor_kind="feet",reference_point=[5,-18]),
                dict(track_id=2,anchor_kind="feet",reference_point=[6,-16]),
                dict(track_id=3,anchor_kind="feet",reference_point=[5,19]),
                dict(track_id=4,anchor_kind="feet",reference_point=[30,-20])]
        after=[dict(track_id=identity,anchor_kind="feet",reference_point=point)
               for identity,point in ((1,[6,-18]),(2,[7,-16]),(3,[6,19]),(4,[31,-20]))]
        result=classify_corner_movement(before,after,geometry,
                                        dict(depth=1,lateral=1),radius_yards=6)
        self.assertEqual(result["selected_corner_ids"],[1,3])

    def test_corner_classifier_rejects_incomplete_corner_set(self):
        geometry = dict(los_center=[0,0],depth_axis=[1,0],lateral_axis=[0,1],
                        thresholds=dict(depth_threshold=20,lateral_limit=15,
                                        middle_limit=5,margin=1),
                        corner_lateral_min=4,corner_depth_margin=5)
        before=[dict(track_id=1,anchor_kind="feet",reference_point=[5,-8]),
                dict(track_id=2,anchor_kind="feet",reference_point=[5,8])]
        after=[dict(track_id=1,anchor_kind="feet",reference_point=[6,-8])]
        result=classify_corner_movement(before,after,geometry,
                                        dict(depth=1,lateral=1),radius_yards=6)
        self.assertEqual(result["classification"],"unknown")
        self.assertIn("comparable post-snap tracks",result["reason"])

    def test_corner_motion_drives_explicit_coverage_heuristic(self):
        self.assertEqual(coverage_hypotheses("1_high",{"classification":"zone_like"})["coverage"],
                         "zone_like")
        self.assertEqual(coverage_hypotheses("1_high",{"classification":"man_like"})["coverage"],
                         "man_like")
        self.assertEqual(coverage_hypotheses("1_high",{"classification":"unknown"})["coverage"],
                         "unknown_coverage")

    def test_saved_cover1_reference_requires_all_measured_signals(self):
        reference = load_coverage_reference(PROJECT_ROOT / "data/metadata/cover_1_man_estimator.json")
        result = coverage_hypotheses("1_high",{"classification":"man_like"},1,
                                     "2_high",reference)
        self.assertEqual(result["coverage"],"cover_1_man_candidate")
        self.assertIn("receiver-to-defender route matching",result["unmeasured_signals"])
        rejected = coverage_hypotheses("1_high",{"classification":"zone_like"},1,
                                       "2_high",reference)
        self.assertEqual(rejected["coverage"],"zone_like")

    def test_coverage_playbook_emits_shell_and_corner_candidates(self):
        playbook=load_coverage_playbook(PROJECT_ROOT/"data/metadata/coverage_reference_playbook.json")
        cover1=coverage_hypotheses("1_high",{"classification":"man_like"},1,
                                   "2_high",playbook=playbook)
        self.assertEqual(cover1["coverage"],"cover_1_man_candidate")
        cover2=coverage_hypotheses("2_high",{"classification":"man_like"},2,
                                   "2_high",playbook=playbook)
        self.assertEqual(cover2["coverage"],"cover_2_man_candidate")
        cover3=coverage_hypotheses("1_high",{"classification":"zone_like"},1,
                                   "2_high",playbook=playbook)
        self.assertEqual(cover3["coverage"],"unknown_coverage")
        self.assertEqual(set(cover3["candidates"]),{"cover_3_buzz","cover_3_cloud"})

    def test_zone_bubbles_project_to_current_frame(self):
        geometry = dict(los_center=[50,50],depth_axis=[1,0],lateral_axis=[0,1])
        rows=[dict(track_id=1,anchor_kind="feet",reference_point=[60,60])]
        result=dict(classification="zone_like",radius_yards=6,bubble_ids=[1])
        polygons=zone_bubble_polygons(rows,result,geometry,
                                      dict(depth=2,lateral=3),np.eye(3))
        self.assertEqual(len(polygons),1)
        self.assertEqual(polygons[0][0],1)
        self.assertEqual(polygons[0][1].shape,(49,2))

    def test_absence_and_incomplete_are_not_zero_high(self):
        self.assertEqual(classify_shell([], complete=True).shell, "unknown")
        self.assertEqual(classify_shell(points((0, 3))).shell, "unknown")
        self.assertEqual(classify_shell(points((0, 3)), complete="false").shell, "unknown")
        with self.assertRaises(ValueError):
            classify_shell(points((0, math.nan)), complete=True)

    def test_rotation_and_unavailable_post_snap(self):
        play = dict(play_id="test", pre_snap=dict(complete=True, defenders=points((-8, 14), (8, 14))),
                    post_snap=dict(complete=True, defenders=points((0, 14), (8, 4))))
        self.assertEqual(analyze_play(play)["rotation"], "2_to_1")
        del play["post_snap"]
        result = analyze_play(play)
        self.assertEqual(result["rotation"], "unknown_rotation")
        self.assertEqual(result["confidence"], 0)

    def test_windows_are_bounded_and_reject_invalid_snap(self):
        self.assertEqual(analysis_window(5, 30, 360), (120, 225))
        self.assertEqual(analysis_window(.5, 30, 60), (0, 60))
        for snap in (0, -1, 12, math.nan, math.inf):
            with self.assertRaises(ValueError):
                analysis_window(snap, 30, 360)


if __name__ == "__main__":
    unittest.main()

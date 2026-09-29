import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.tracking.formations import load_formation_map, match_formation_slots
from src.tracking.tracking import Observation, PlayerTracker


class FormationTests(unittest.TestCase):
    def setUp(self):
        self.play = {
            "anchor_frame": 100,
            "max_distance": .08,
            "slots": [
                {"name": "X", "position": "WR", "team": "NE", "point": [.20, .30]},
                {"name": "RB", "position": "RB", "team": "KC", "point": [.80, .70]},
            ],
        }

    def test_loads_valid_clip_formation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formations.json"
            path.write_text(json.dumps({"version": 1, "plays": {"pass_001": self.play}}))
            self.assertEqual(load_formation_map(path)["pass_001"]["anchor_frame"], 100)

    def test_rejects_unknown_position_and_invalid_point(self):
        for change in ({"position": "OL"}, {"point": [1.2, .5]}):
            play = json.loads(json.dumps(self.play))
            play["slots"][0].update(change)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "formations.json"
                path.write_text(json.dumps({"version": 1, "plays": {"pass_001": play}}))
                with self.assertRaises(ValueError):
                    load_formation_map(path)

    def test_matches_slots_one_to_one_with_team_constraint(self):
        observations = [
            Observation(1, .9, (18, 25, 22, 35), "helmet", team="white"),
            Observation(2, .9, (78, 65, 82, 75), "helmet", team="red"),
            Observation(3, .9, (19, 25, 23, 35), "helmet", team="red"),
        ]
        assignments, unmatched = match_formation_slots(
            observations, self.play, (100, 100, 3), {"white": "NE", "red": "KC"})
        self.assertEqual({identity: slot["name"] for identity, slot in assignments.items()},
                         {1: "X", 2: "RB"})
        self.assertEqual(unmatched, [])

    def test_rejects_ambiguous_or_distant_slot_matches(self):
        observations = [
            Observation(1, .9, (18, 25, 22, 35), "helmet", team="white"),
            Observation(2, .9, (18.4, 25, 22.4, 35), "helmet", team="white"),
        ]
        play = {"max_distance": .08,
                "slots": [{"name": "X", "position": "WR", "point": [.20, .30]}]}
        assignments, unmatched = match_formation_slots(observations, play, (100, 100, 3))
        self.assertEqual(assignments, {})
        self.assertEqual(unmatched, play["slots"])

    def test_tracker_activates_matched_position_trails(self):
        tracker = PlayerTracker(SimpleNamespace(predict=lambda *a, **kw: []), [0],
                                tracker="bytetrack.yaml", target="helmet")
        tracker._current_observations = [
            Observation(1, .9, (18,25,22,35), "helmet", team="white"),
            Observation(2, .9, (78,65,82,75), "helmet", team="red"),
        ]
        tracker.confirmed_ids.update((1, 2))
        tracker.team_names = {"white": "NE", "red": "KC"}
        assignments, unmatched = tracker.apply_formation_slots(self.play, (100,100,3))
        tracker.trail_scope = "formation"
        self.assertEqual(set(assignments), {1, 2})
        self.assertEqual(unmatched, [])
        self.assertEqual(tracker.formation_roles, {1: "WR", 2: "RB"})
        self.assertEqual(tracker.trail_ids_to_draw(), {1, 2})


if __name__ == "__main__":
    unittest.main()
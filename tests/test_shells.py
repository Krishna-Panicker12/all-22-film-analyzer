import math
import unittest

from src.shells.baseline import analyze_play, analysis_window, classify_shell


def points(*pairs):
    return [dict(lateral=x, depth=y) for x, y in pairs]


class ShellTests(unittest.TestCase):
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

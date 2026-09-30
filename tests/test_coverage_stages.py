import unittest
from src.shells.coverage_stages import staged_coverage


class CoverageStagesTests(unittest.TestCase):
    def test_two_to_one_rotation_then_man(self):
        result = staged_coverage(
            dict(candidate='2_high', high_safety_ids=[1,2]),
            dict(candidate='1_high', high_safety_ids=[1], geometry_points=[{'track_id':1},{'track_id':2}]),
            dict(classification='man_like'))
        self.assertIn('cover_2_man',result['pre_snap']['candidates'])
        self.assertEqual(result['post_snap']['rotation']['dropped_from_high_ids'],[2])
        self.assertEqual(result['corner_refinement']['candidates'],['cover_1_man'])
        self.assertIn('cover_3_zone',result['corner_refinement']['alternative_candidates'])

    def test_missing_track_is_not_rotation(self):
        result = staged_coverage(dict(candidate='2_high',high_safety_ids=[1,2]),
            dict(candidate='1_high',high_safety_ids=[1],geometry_points=[{'track_id':1}]),{})
        self.assertEqual(result['post_snap']['rotation']['status'],'unknown')
        self.assertEqual(result['post_snap']['rotation']['dropped_from_high_ids'],[])

    def test_unknown_shell_not_resolved_by_motion(self):
        result = staged_coverage({}, {}, dict(classification='man_like'))
        self.assertEqual(result['corner_refinement']['status'],'unknown')

    def test_two_high_zone_keeps_alternatives(self):
        result = staged_coverage({},dict(candidate='2_high'),dict(classification='zone_like'))
        self.assertEqual(result['corner_refinement']['candidates'],['cover_2_zone','cover_4_zone','cover_6_zone'])

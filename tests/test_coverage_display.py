import unittest
from unittest.mock import patch
import json
from pathlib import Path
import cv2
import numpy as np
from src.shells.display import banner,display_estimate,wait_for_manual_exit
from src.shells.run_defense import ROOT,read_config


class DisplayTests(unittest.TestCase):
    def test_final_hold_ignores_timeout_and_space_until_q(self):
        with patch('src.shells.display.cv2.getWindowProperty',return_value=1), patch('src.shells.display.cv2.waitKey',side_effect=[-1,32,-1,ord('q')]) as wait:
            self.assertEqual(wait_for_manual_exit('test'),'next')
            self.assertEqual(wait.call_count,4)

    def test_escape_quits_batch(self):
        with patch('src.shells.display.cv2.getWindowProperty',return_value=1), patch('src.shells.display.cv2.waitKey',return_value=27):
            self.assertEqual(wait_for_manual_exit('test'),'quit')

    def test_closed_window_quits(self):
        with patch('src.shells.display.cv2.getWindowProperty',return_value=0):
            self.assertEqual(wait_for_manual_exit('test'),'quit')

    def test_unknown_does_not_become_certain_coverage(self):
        d=display_estimate({}, {}, {},True)
        self.assertIn('UNKNOWN',d['estimate'])

    def test_missing_safeties_dont_display_zero_high(self):
        d=display_estimate(dict(candidate='0_high',completeness_confirmed=False))
        self.assertEqual(d['shell'],'UNKNOWN')

    def test_film_review_is_explicit_and_not_used_presnap(self):
        review=dict(pre_shell='1_high',coverage='cover_1_man')
        d=display_estimate({}, {},review,True)
        self.assertIn('FILM REVIEW',d['source'])
        self.assertIn('COVER 3 ZONE',display_estimate({}, {},review,False)['estimate'])

    def test_banner_preserves_whole_field(self):
        frame=np.full((720,1280,3),90,np.uint8)
        result=banner(frame,display_estimate(), 'pass_001',2.3,True)
        self.assertEqual(result.shape,(940,1280,3))
        np.testing.assert_array_equal(result[220:],frame)

    def test_all_clips_have_bounded_configs(self):
        clips=sorted((ROOT/'data/plays/game_01').glob('pass_*.mp4'))
        self.assertEqual(len(clips),15)
        for clip in clips:
            c=read_config(ROOT/f'data/metadata/defense_{clip.stem}.json')
            cap=cv2.VideoCapture(str(clip));fps=cap.get(cv2.CAP_PROP_FPS)
            count=cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release()
            self.assertLess(c['snap_frame']+round(c['post_seconds']*fps),count)
            self.assertEqual(c['post_seconds'],2.3)

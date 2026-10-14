import unittest
import cv2
import numpy as np
from src.shells.deep_safeties import deepest_safety_ids,deepest_image_safety_ids
from src.shells.playback_controls import PlaybackControls


class DeepSafetyTests(unittest.TestCase):
    def setUp(self):
        self.geometry=dict(los_center=[0,0],depth_axis=[1,0],lateral_axis=[0,1],
            pixels_per_yard=dict(depth=1,lateral=1),
            thresholds=dict(depth_threshold=10,lateral_limit=18,middle_limit=6,margin=1))

    def test_deepest_tier_excludes_underneath_and_wide_corners(self):
        points=[dict(track_id=i,depth=d,lateral=l) for i,d,l in
                [(1,12,0),(2,14,8),(3,26,2),(4,27,35),(5,-30,0)]]
        self.assertEqual(deepest_safety_ids(points,self.geometry),[3])

    def test_two_deep_defenders_and_ball_distance(self):
        points=[dict(track_id=i,depth=d,lateral=l) for i,d,l in [(1,24,-8),(2,26,8),(3,15,0)]]
        self.assertEqual(deepest_safety_ids(points,self.geometry),[1,2])
        self.geometry['ball_reference_point']=[25,8]
        self.assertEqual(deepest_safety_ids(points,self.geometry),[1])

    def test_no_near_los_safety(self):
        self.assertEqual(deepest_safety_ids([dict(track_id=1,depth=3,lateral=0)],self.geometry),[])

    def test_image_fallback_uses_deepest_tier_without_registration(self):
        self.geometry['pixels_per_yard']=dict(depth=10,lateral=10)
        rows=[dict(track_id=i,source='fresh_frame_detection',image_point=[x,100])
              for i,x in [(1,100),(2,130),(3,180),(4,275),(5,292)]]
        self.assertEqual(deepest_image_safety_ids(rows,self.geometry),[4,5])
        self.assertEqual(deepest_image_safety_ids(rows[:3],self.geometry),[])


class PlaybackTests(unittest.TestCase):
    def test_hold_release_and_skip(self):
        controls=PlaybackControls()
        controls.draw(np.zeros((300,640,3),np.uint8))
        controls.mouse(cv2.EVENT_LBUTTONDOWN,50,325,cv2.EVENT_FLAG_LBUTTON)
        self.assertTrue(controls.fast)
        controls.mouse(cv2.EVENT_LBUTTONUP,700,100,0)
        self.assertFalse(controls.fast)
        controls.mouse(cv2.EVENT_LBUTTONDOWN,400,325,cv2.EVENT_FLAG_LBUTTON)
        self.assertEqual(controls.seek_target(255,50,100),100)
        self.assertIsNone(controls.seek_target(255,50,100))
        self.assertEqual(controls.seek_target(ord('s'),50,100),100)
        self.assertIsNone(controls.seek_target(ord('s'),101,100))

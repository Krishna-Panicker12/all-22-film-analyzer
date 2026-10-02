import unittest
import numpy as np
from src.shells.responsibilities import responsibility_overlay


class ResponsibilityTests(unittest.TestCase):
    def test_zone_boxes_only_reviewed_coverage_roles(self):
        frame=np.zeros((200,300,3),np.uint8)
        geometry=dict(depth_axis=[1,0],lateral_axis=[0,1],pixels_per_yard=dict(depth=5,lateral=5))
        defenders=[dict(track_id=1,role='CB',image_point=[50,50]),
                   dict(track_id=2,role='S',image_point=[100,50]),
                   dict(track_id=3,role='DL',image_point=[150,50]),
                   dict(track_id=4,image_point=[200,50])]
        _,objects=responsibility_overlay(frame,'cover_2_zone',defenders,[],geometry)
        self.assertEqual({o['defender_id'] for o in objects},{1,2})

    def test_every_reviewed_db_gets_one_box(self):
        frame=np.zeros((300,400,3),np.uint8)
        geometry=dict(depth_axis=[1,0],lateral_axis=[0,1],pixels_per_yard=dict(depth=5,lateral=5))
        roles=['CB','S','FS','SS','DB','NICKEL','LB_or_DB']
        defenders=[dict(track_id=i,role=role,image_point=[30+i*40,100]) for i,role in enumerate(roles)]
        _,objects=responsibility_overlay(frame,'cover_2_zone',defenders,[],geometry)
        self.assertEqual(len(objects),len(defenders))

if __name__=='__main__': unittest.main()

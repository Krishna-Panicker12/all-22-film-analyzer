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

    def test_high_safety_box_for_every_coverage(self):
        frame=np.zeros((300,400,3),np.uint8)
        geometry=dict(depth_axis=[1,0],lateral_axis=[0,1],pixels_per_yard=dict(depth=4,lateral=4))
        defenders=[dict(track_id=1,role='S',image_point=[100,100]),
                   dict(track_id=2,role='CB',image_point=[200,100]),
                   dict(track_id=3,role='DL',image_point=[300,100])]
        receivers=[dict(track_id=9,image_point=[220,120])]
        pairs=[dict(defender_id=2,receiver_candidate_id=9,status='following_candidate')]
        for coverage in ['cover_1_man','cover_2_zone','unknown']:
            with self.subTest(coverage=coverage):
                _,objects=responsibility_overlay(frame,coverage,defenders,receivers,geometry,pairs,
                                                 high_safety_ids=[1,99])
                boxes=[o for o in objects if o['kind']=='illustrative_zone']
                safety=[o for o in boxes if o['defender_id']==1]
                self.assertEqual(len(safety),1)
                self.assertEqual(safety[0]['width_yards'],15)
                polygon=np.array(safety[0]['polygon'])
                np.testing.assert_array_equal(polygon.max(axis=0)-polygon.min(axis=0),[60,60])
                np.testing.assert_array_equal(polygon.mean(axis=0),[100,100])
                self.assertFalse(any(o['defender_id'] in [3,99] for o in objects))
                others=[o for o in boxes if o['defender_id']==2]
                self.assertEqual(len(others),int('zone' in coverage))
                if others: self.assertEqual(others[0]['width_yards'],8)
                self.assertEqual(sum(o['kind']=='following_candidate' for o in objects),int('man' in coverage))

if __name__=='__main__': unittest.main()

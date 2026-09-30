import unittest
from src.shells.following import receiver_following
from src.shells.coverage_stages import staged_coverage

G = dict(depth_axis=[1,0],lateral_axis=[0,1],los_center=[0,0],pixels_per_yard=dict(depth=1,lateral=1))


def row(i,x,y):
    return dict(track_id=i,reference_point=[x,y],anchor_kind='feet')


def sequence(stationary=False,pan=False):
    return [dict(frame_index=t,registration_ok=not pan,
        defenders=[row(1,0 if stationary else t/5,1)],
        receivers=[row(10,t/5,0)]) for t in range(1,46)]


class FollowingTests(unittest.TestCase):
    def test_sustained_follow(self):
        r=receiver_following(sequence(),0,30,G,[1])
        self.assertEqual(r['classification'],'following_evidence')
        self.assertEqual(r['pairs'][0]['receiver_candidate_id'],10)

    def test_stationary_defender_not_following(self):
        self.assertEqual(receiver_following(sequence(True),0,30,G)['classification'],'unknown')

    def test_failed_registration_not_evidence(self):
        self.assertEqual(receiver_following(sequence(pan=True),0,30,G)['pairs'],[])

    def test_fragmented_track_rejected(self):
        frames=sequence()
        for f in frames[10:25]: f['receivers']=[]
        self.assertEqual(receiver_following(frames,0,30,G)['pairs'],[])

    def test_competing_receiver_ambiguous(self):
        frames=sequence()
        for f in frames: f['receivers'].append(row(11,f['frame_index']/5,2))
        r=receiver_following(frames,0,30,G)
        self.assertEqual(r['classification'],'unknown')
        self.assertEqual(r['pairs'][0]['status'],'ambiguous_receiver')

    def test_shared_receiver_not_two_independent_follows(self):
        frames=sequence()
        for f in frames: f['defenders'].append(row(2,f['frame_index']/5,-1))
        r=receiver_following(frames,0,30,G)
        self.assertEqual(r['classification'],'unknown')

    def test_following_preserves_match_zone_alternative(self):
        result=staged_coverage({},dict(candidate='1_high'),dict(classification='zone_like'),
            following=dict(classification='following_evidence'))
        self.assertEqual(result['corner_refinement']['candidates'],['cover_1_man','cover_3_zone'])

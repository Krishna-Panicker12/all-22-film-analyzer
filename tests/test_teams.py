import unittest

import numpy as np

from src.tracking.teams import JerseyTeams, jersey_color, helmet_color, helmet_jersey_color
from src.tracking.tracking import Observation


class TeamTests(unittest.TestCase):
    def test_helmet_shell_is_primary_team_evidence(self):
        frame = np.zeros((100,100,3),np.uint8)
        frame[10:20,30:40] = (20,20,200)
        frame[20:50,25:45] = 220  # conflicting white jersey
        helmet = (30,10,40,22)
        self.assertEqual(helmet_color(frame,helmet),"red")
        teams = JerseyTeams()
        for index in range(2):
            item = Observation(1,.9,helmet,"helmet")
            teams.update(frame,[item],index)
        self.assertEqual(item.team,"red")
    def test_helmet_jersey_evidence_without_body_detection(self):
        frame = np.zeros((100,100,3), np.uint8)
        frame[20:34,28:42] = (20,20,200)
        helmet = (30,10,40,20)
        self.assertEqual(helmet_jersey_color(frame,helmet),"red")
        self.assertEqual(helmet_jersey_color(frame,helmet,[(30,23,40,33)]),"unknown")
        teams = JerseyTeams()
        for index in range(3):
            item = Observation(1,.9,helmet,"helmet")
            teams.update(frame,[item],index)
        self.assertEqual(item.team,"red")
        self.assertIsNone(item.ground_anchor)

    def test_colors_and_ambiguous_body(self):
        red = np.full((100,100,3), (20,20,200), np.uint8)
        white = np.full((100,100,3), 220, np.uint8)
        grass = np.full((100,100,3), (40,130,40), np.uint8)
        box = (20,0,60,100)
        self.assertEqual(jersey_color(red,box), "red")
        self.assertEqual(jersey_color(white,box), "white")
        self.assertEqual(jersey_color(grass,box), "unknown")
        self.assertEqual(jersey_color(red,box,[box]), "unknown")
        self.assertEqual(jersey_color(red,(0,0,100,40)), "unknown")

    def test_votes_conflicts_and_reset_do_not_change_identity(self):
        teams = JerseyTeams()
        red = np.full((100,100,3), (20,20,200), np.uint8)
        white = np.full((100,100,3), 220, np.uint8)
        def observed():
            return Observation(5,.8,(30,0,40,10),"helmet",body_box=(20,0,60,100))
        for frame in range(3):
            item = observed()
            teams.update(red,[item],frame)
        self.assertEqual(item.team,"red")
        self.assertEqual(item.track_id,5)
        item = observed()
        teams.update(white,[item],3)
        self.assertEqual(item.team,"red")
        for frame in range(4,12):
            item = observed()
            teams.update(white,[item],frame)
        self.assertEqual(item.team,"white")
        teams.reset()
        item = observed()
        teams.update(red,[item],4)
        self.assertEqual(item.team,"unknown")

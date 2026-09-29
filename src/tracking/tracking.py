"""Detect once per frame, then explicitly pass detections to the tracker."""
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

BODY_ID_BASE = 1_000_000


@dataclass
class Observation:
    track_id: int | None
    confidence: float
    box: tuple
    source: str = "player"
    ground_anchor: tuple | None = None
    body_box: tuple | None = None
    team: str = "unknown"
    stable_frames: int = 1
    body_index: int | None = None


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / (area_a+area_b-intersection) if area_a+area_b-intersection else 0


def unique_helmet_indices(boxes, scores):
    """Remove near-identical detector boxes before they create competing IDs."""
    kept = []
    for index in np.argsort(scores)[::-1]:
        if all(box_iou(boxes[index], boxes[other]) < .60 for other in kept):
            kept.append(int(index))
    return kept


def unique_body_indices(boxes, scores):
    """Suppress duplicate person boxes before they create competing body IDs."""
    kept = []
    for index in np.argsort(scores)[::-1]:
        if all(box_iou(boxes[index], boxes[other]) < .70 for other in kept):
            kept.append(int(index))
    return kept


def smooth_anchor(previous, current, alpha=.35):
    """Dampen detector jitter without changing the feet-only trail anchor."""
    if previous is None:
        return tuple(map(float, current))
    return tuple(float(old + alpha * (new-old)) for old,new in zip(previous,current))


def inferred_head_box(body_box):
    """Estimate a compact head marker from a full-body detection."""
    left, top, right, bottom = body_box
    width = max(2.0, min(.4*(right-left), .35*(bottom-top)))
    height = .75*width
    center = (left+right)/2
    return (center-width/2, top, center+width/2, top+height)


def route_role_eligible(state, other_states):
    """Approximate skill/second-level roles from movement and trench density."""
    height = max(1.0, state["height"])
    start = np.asarray(state["start"], dtype=float)
    end = np.asarray(state["last"], dtype=float)
    displacement = np.linalg.norm(end-start) / height
    travel = state["travel"] / height
    dense_neighbors = sum(
        other is not state and
        np.linalg.norm(np.asarray(other["start"], dtype=float)-start) / height < 1.5
        for other in other_states
    )
    # Players in the tightest formation cluster must clearly release or pursue;
    # this rejects most linemen while retaining backs and second-level defenders.
    required_displacement = .7 if dense_neighbors >= 3 else .45
    required_travel = 1.1 if dense_neighbors >= 3 else .8
    return displacement >= required_displacement or travel >= required_travel


def scale_boxes(boxes, factor):
    """Resize boxes around their centers without modifying detector output."""
    scaled = np.asarray(boxes).copy()
    centers = (scaled[:, :2] + scaled[:, 2:4]) / 2
    half = (scaled[:, 2:4] - scaled[:, :2]) * factor / 2
    scaled[:, :2], scaled[:, 2:4] = centers-half, centers+half
    return scaled


def suppress_duplicate_observations(observations, prior_hits=None):
    """Keep one track when the tracker emits multiple IDs for one detection."""
    prior_hits = prior_hits or {}
    ordered = sorted(observations,
                     key=lambda item: (prior_hits.get(item.track_id, (0,))[0], item.confidence),
                     reverse=True)
    kept = []
    for item in ordered:
        x1,y1,x2,y2 = item.box
        center = np.array(((x1+x2)/2, (y1+y2)/2))
        size = max(1, min(x2-x1, y2-y1))
        duplicate = False
        for other in kept:
            a,b,c,d = other.box
            other_center = np.array(((a+c)/2, (b+d)/2))
            other_size = max(1, min(c-a, d-b))
            if box_iou(item.box, other.box) >= .45 or np.linalg.norm(center-other_center) <= .35*min(size,other_size):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def limit_roster(observations, limit=22):
    """Return at most the regulation on-field roster; never create observations."""
    if len(observations) <= limit:
        return list(observations)
    ranked=sorted(observations,key=lambda item:(
        item.stable_frames,
        item.source == "helmet",
        item.team != "unknown",
        item.ground_anchor is not None,
        item.confidence),reverse=True)
    keep={id(item) for item in ranked[:limit]}
    return [item for item in observations if id(item) in keep]


def canonicalize_track_ids(observations, aliases, memory, frame_index,
                           max_gap=3, appearance=None, appearance_memory=None):
    """Reconnect a fresh engine ID to one recent, spatially unambiguous helmet ID."""
    used=set()
    new=[]
    for item in observations:
        engine_id=item.track_id
        if engine_id in aliases:
            item.track_id=aliases[engine_id]
            used.add(item.track_id)
        else:
            new.append((engine_id,item))
    candidates=[]
    for engine_id,item in new:
        x1,y1,x2,y2=item.box
        center=np.array(((x1+x2)/2,(y1+y2)/2))
        diagonal=max(1,np.hypot(x2-x1,y2-y1))
        for canonical,(box,last) in memory.items():
            gap = frame_index-last
            if canonical in used or gap > max_gap:
                continue
            appearance_cost = 0.0
            if gap > 3:
                current = (appearance or {}).get(engine_id)
                previous = (appearance_memory or {}).get(canonical)
                # Color alone cannot establish identity. Require agreement in
                # both appearance and a tight spatial gate, with mutual uniqueness.
                if current is None or previous is None or current[0] != previous[0]:
                    continue
                appearance_cost = cv2.compareHist(current[1], previous[1], cv2.HISTCMP_BHATTACHARYYA)
                if appearance_cost > .20:
                    continue
            a,b,c,d=box
            old_center=np.array(((a+c)/2,(b+d)/2))
            old_diagonal=max(1,np.hypot(c-a,d-b))
            ratio=diagonal/old_diagonal
            distance=np.linalg.norm(center-old_center)/max(diagonal,old_diagonal)
            if .55 <= ratio <= 1.8 and distance <= 1.6:
                score=distance+abs(np.log(ratio))+appearance_cost
                candidates.append((score,engine_id,canonical,item))
    for score,engine_id,canonical,item in sorted(candidates,key=lambda row:row[0]):
        if engine_id in aliases or canonical in used:
            continue
        engine_scores=[s for s,e,_,_ in candidates if e==engine_id]
        canonical_scores=[s for s,_,c,_ in candidates if c==canonical]
        # Require a clear best choice from both directions in crowded groups.
        if score != min(engine_scores) or score != min(canonical_scores):
            continue
        if (sum(s <= score+.30 for s in engine_scores)>1 or
                sum(s <= score+.30 for s in canonical_scores)>1):
            continue
        aliases[engine_id]=canonical
        item.track_id=canonical
        used.add(canonical)
    for engine_id,item in new:
        if engine_id not in aliases:
            aliases[engine_id]=engine_id
            item.track_id=engine_id
            used.add(engine_id)
    for item in observations:
        memory[item.track_id]=(item.box,frame_index)
    for canonical,state in list(memory.items()):
        if frame_index-state[1]>max_gap:
            del memory[canonical]
    return observations


def associate_helmets_with_bodies(helmets, bodies):
    """Use a body only when its upper half contains exactly one helmet, and
    that helmet has exactly one candidate body. Crowded matches stay unknown.
    Returned foot points are estimates from body boxes, never from helmet height.
    """
    matches = []
    for helmet in helmets:
        x1, y1, x2, y2 = helmet.box
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        candidates = [i for i, (left, top, right, bottom) in enumerate(bodies)
                      if left <= x <= right and top <= y <= top + .5 * (bottom - top)]
        matches.append(candidates)
    used = set()
    for helmet, candidates in zip(helmets, matches):
        if len(candidates) == 1 and sum(candidates[0] in m for m in matches) == 1:
            index = candidates[0]
            left, top, right, bottom = bodies[index]
            helmet.ground_anchor = (float((left + right) / 2), float(bottom))
            helmet.body_box = (left, top, right, bottom)
            helmet.body_index = index
            used.add(index)
    return used


def continue_tracks_with_bodies(observations, bodies, used_bodies, memory, frame_index,
                                allowed_ids=None, max_gap=2):
    """Bridge a helmet gap with a uniquely matched, moving body detection.

    A support observation uses the measured body box and an existing helmet ID;
    it never pretends that an unseen helmet was detected.
    """
    current_ids = {item.track_id for item in observations}
    eligible = {identity: state for identity,state in memory.items()
                if identity not in current_ids and frame_index-state[1] <= max_gap
                and (allowed_ids is None or identity in allowed_ids)}
    candidates = []
    for identity,(previous,last) in eligible.items():
        for index,body in enumerate(bodies):
            if index not in used_bodies:
                gap = frame_index - last
                old_center = np.array(((previous[0]+previous[2])/2,
                                       (previous[1]+previous[3])/2))
                new_center = np.array(((body[0]+body[2])/2,
                                       (body[1]+body[3])/2))
                old_size = max(1, np.hypot(previous[2]-previous[0], previous[3]-previous[1]))
                new_size = max(1, np.hypot(body[2]-body[0], body[3]-body[1]))
                size_ratio = new_size / old_size
                distance = np.linalg.norm(new_center-old_center) / max(old_size, new_size)
                # Body boxes translate substantially between frames, so IoU alone
                # rejects real matches. Keep the motion gate bounded for safety.
                motion_limit = min(1.5 + .45 * max(0, gap-1), 4.5)
                if .5 <= size_ratio <= 2 and distance <= motion_limit:
                    score = distance + .25 * abs(np.log(size_ratio))
                    candidates.append((score,identity,index))
    supports=[]
    for score,identity,index in sorted(candidates):
        if identity in {item.track_id for item in supports} or index in used_bodies:
            continue
        # Require mutual uniqueness: close alternatives are unsafe in a pile.
        identity_scores=[value for value,i,_ in candidates if i==identity]
        body_scores=[value for value,_,j in candidates if j==index]
        if score != min(identity_scores) or score != min(body_scores):
            continue
        if (sum(value <= score+.20 for value in identity_scores)>1 or
                sum(value <= score+.20 for value in body_scores)>1):
            continue
        body = bodies[index]
        left,top,right,bottom=bodies[index]
        supports.append(Observation(identity,0.0,tuple(body),"body_support",
                                    (float((left+right)/2),float(bottom)),tuple(body),
                                    body_index=index))
        used_bodies.add(index)
    return supports


def merge_identity_state(old_identity, new_identity, item, engine_id, tracker):
    """Keep a body-backed route when a helmet track acquires that identity."""
    if old_identity == new_identity:
        return
    tracker.id_aliases[engine_id] = new_identity
    for alias, identity in list(tracker.id_aliases.items()):
        if identity == old_identity:
            tracker.id_aliases[alias] = new_identity
    old_hits = tracker.track_hits.pop(old_identity, (0, -1))
    current_hits = tracker.track_hits.get(new_identity, (0, -1))
    continued_hits = old_hits[0] + 1 if old_hits[1] == tracker.frame_index - 1 else old_hits[0]
    item.stable_frames = max(continued_hits, current_hits[0])
    tracker.track_hits[new_identity] = (item.stable_frames, tracker.frame_index)
    if old_identity in tracker.confirmed_ids:
        tracker.confirmed_ids.add(new_identity)
        tracker.confirmed_ids.discard(old_identity)
    if old_identity in tracker.excluded:
        tracker.excluded.add(new_identity)
    old_history = tracker.history.pop(old_identity, None)
    if new_identity not in tracker.history and old_history is not None:
        tracker.history[new_identity] = old_history
    old_motion = tracker.route_motion.pop(old_identity, None)
    if new_identity not in tracker.route_motion and old_motion is not None:
        tracker.route_motion[new_identity] = old_motion
    if old_identity in tracker.route_eligible_ids:
        tracker.route_eligible_ids.add(new_identity)
        tracker.route_eligible_ids.discard(old_identity)
    if old_identity in tracker.route_excluded_ids:
        tracker.route_excluded_ids.add(new_identity)
        tracker.route_excluded_ids.discard(old_identity)
    if old_identity in tracker.manual_trail_ids:
        tracker.manual_trail_ids.add(new_identity)
        tracker.manual_trail_ids.discard(old_identity)
    old_role = tracker.formation_roles.pop(old_identity, None)
    if old_role is not None:
        tracker.formation_roles.setdefault(new_identity, old_role)
        tracker.formation_trail_ids.add(new_identity)
        tracker.formation_trail_ids.discard(old_identity)
    tracker.canonical_memory.pop(old_identity, None)
    tracker.canonical_memory[new_identity] = (item.box, tracker.frame_index)
    old_appearance = tracker.appearance_memory.pop(old_identity, None)
    if old_appearance is not None:
        tracker.appearance_memory.setdefault(new_identity, old_appearance)
    if tracker.teams is not None:
        if old_identity in tracker.teams.assignments:
            tracker.teams.assignments.setdefault(new_identity, tracker.teams.assignments[old_identity])
            del tracker.teams.assignments[old_identity]
        if old_identity in tracker.teams.votes:
            tracker.teams.votes.setdefault(new_identity, tracker.teams.votes[old_identity])
            del tracker.teams.votes[old_identity]


def on_field(item, field):
    if item.source == "helmet":
        # Without feet, only admit helmets clearly inside the field. Sideline
        # heads can project above the boundary even when their feet are outside.
        if item.ground_anchor is None:
            if field.points is None:
                return True
            left, top, right, bottom = item.box
            distance = cv2.pointPolygonTest(field.points,
                (float((left + right) / 2), float((top + bottom) / 2)), True)
            return distance >= 2 * max(1, bottom - top)
        x, y = item.ground_anchor
        return field.contains((x, y, x, y))
    return field.contains(item.box)


class PlayerTracker:
    def __init__(self, model, person_ids, confidence=0.25, image_size=1280,
                 device=None, tracker="botsort.yaml", buffer=30, target="player",
                 body_model=None, body_classes=None, team_colors=None,
                 confirmation_frames=None, helmet_match_scale=1.0):
        if not 1 <= helmet_match_scale <= 3:
            raise ValueError("helmet_match_scale must be between 1 and 3")
        self.helmet_match_scale = helmet_match_scale
        self.model = model
        self.target = target
        self.body_model = body_model
        self.body_classes = body_classes
        from src.tracking.teams import JerseyTeams
        self.teams = JerseyTeams() if team_colors == "red-white" else None
        self.team_names = {}
        self.confirmation_frames = confirmation_frames or (2 if target == "helmet" else 1)
        self.options = dict(classes=person_ids, conf=confidence, imgsz=image_size,
                            device=device, verbose=False)
        config = SimpleNamespace(**YAML.load(check_yaml(tracker)))
        config.track_buffer = buffer
        self.engine = (BOTSORT if config.tracker_type == "botsort" else BYTETracker)(config)
        self.raw = []
        self.unsuppressed = []
        self._current_observations = []
        self.duplicates_suppressed = 0
        self.tentative_suppressed = 0
        self.missing = []
        self.history = {}
        self.frame_index = 0
        self.excluded = set()
        self.segment = 0
        self.track_hits = {}
        self.body_memory = {}
        self.body_reconnect_gap = min(buffer, 12)
        self.next_body_identity = BODY_ID_BASE
        self.trail_scope = "manual"
        self.manual_trail_ids = set()
        self.route_motion = {}
        self.route_eligible_ids = set()
        self.route_excluded_ids = set()
        self.confirmed_ids = set()
        self.superseded_ids = set()
        self.id_aliases = {}
        self.canonical_memory = {}
        self.appearance_memory = {}
        self.formation_trail_ids = set()
        self.formation_roles = {}
        self.reconnect_buffer = buffer
        # Admit weak detections for the tracker's second association stage.
        # New identities still require the normal confidence threshold.
        if target == "helmet" and helmet_match_scale > 1:
            self.options["conf"] = min(confidence, config.track_low_thresh)
            config.new_track_thresh = max(config.new_track_thresh, confidence)

    def update(self, frame, field=None):
        # Reproject old trail points into the current camera view. If camera
        # motion could not be measured, restart rather than draw a false route.
        if field is not None:
            self.move_history(field.motion)
            self.move_identity_memory(field.motion)
        # predict() does not attach tracking callbacks. Keep raw detections and
        # advance the tracker exactly once, including on frames with no detections.
        result = self.model.predict(frame, **self.options)[0]
        boxes = result.boxes.cpu().numpy()
        self.raw = [Observation(None, float(score), tuple(box), self.target)
                    for box, score in zip(boxes.xyxy, boxes.conf)]
        if self.target == "helmet" and len(boxes):
            boxes = boxes[unique_helmet_indices(boxes.xyxy, boxes.conf)]
            # Tiny helmets can move a full box width between adjacent frames.
            # Expand only the association geometry, then undo it for output.
            data = boxes.data.copy()
            data[:, :4] = scale_boxes(data[:, :4], self.helmet_match_scale)
            boxes = type(boxes)(data, boxes.orig_shape)
        # Track every detection, even outside the field. Filtering the input here
        # used to starve excluded tracks when the boundary moved; they returned
        # under fresh IDs. Field filtering now affects only display/export.
        tracks = self.engine.update(boxes, frame)
        if self.target == "helmet" and len(tracks):
            tracks[:, :4] = scale_boxes(tracks[:, :4], 1 / self.helmet_match_scale)
        self.frame_index += 1
        observations = [Observation(int(row[4]), float(row[5]), tuple(row[:4]), self.target)
                        for row in tracks]
        self.unsuppressed = list(observations)
        observations = suppress_duplicate_observations(observations, self.track_hits)
        kept_ids={item.track_id for item in observations}
        self.superseded_ids.update(item.track_id for item in self.unsuppressed
                                   if item.track_id not in kept_ids)
        from src.tracking.appearance import helmet_appearance
        appearance = {item.track_id: helmet_appearance(frame, item.box, observations)
                      for item in observations} if self.target == "helmet" else {}
        engine_ids = [item.track_id for item in observations]
        observations=canonicalize_track_ids(observations,self.id_aliases,
                                            self.canonical_memory,self.frame_index,
                                            self.reconnect_buffer if self.target == "helmet" else 3,
                                            appearance,self.appearance_memory)
        for engine_id, item in zip(engine_ids, observations):
            if appearance.get(engine_id) is not None:
                self.appearance_memory[item.track_id] = appearance[engine_id]
        self.appearance_memory = {identity: value for identity, value in self.appearance_memory.items()
                                  if identity in self.canonical_memory}
        self.duplicates_suppressed = len(self.unsuppressed)-len(observations)
        for item in observations:
            hits, last = self.track_hits.get(item.track_id, (0, -1))
            hits = hits + 1 if last == self.frame_index - 1 else 1
            item.stable_frames = hits
            self.track_hits[item.track_id] = (hits, self.frame_index)
        if self.target == "helmet" and self.body_model is not None:
            result = self.body_model.predict(frame, classes=self.body_classes, conf=.25,
                                             imgsz=self.options["imgsz"],
                                             device=self.options["device"], verbose=False)[0]
            body_boxes = result.boxes.cpu().numpy()
            body_indices = unique_body_indices(body_boxes.xyxy, body_boxes.conf)
            bodies = body_boxes.xyxy[body_indices].tolist()
            used = associate_helmets_with_bodies(observations, bodies)
            engine_id_by_item = {id(item): engine_id
                                 for engine_id, item in zip(engine_ids, observations)}
            prior_body_matches = continue_tracks_with_bodies(
                [], bodies, set(), self.body_memory, self.frame_index,
                max_gap=self.body_reconnect_gap)
            for item in observations:
                if item.body_index is None:
                    continue
                matches = [support for support in prior_body_matches
                           if support.body_index == item.body_index
                           and support.track_id >= BODY_ID_BASE]
                if len(matches) == 1:
                    old_identity = item.track_id
                    body_identity = matches[0].track_id
                    if old_identity != body_identity:
                        merge_identity_state(old_identity, body_identity, item,
                                             engine_id_by_item[id(item)], self)
                        item.track_id = body_identity
            supports = continue_tracks_with_bodies(observations,bodies,used,
                                                    self.body_memory,self.frame_index,
                                                    max_gap=self.body_reconnect_gap)
            for item in supports:
                hits,last = self.track_hits.get(item.track_id,(0,-1))
                hits = hits + 1 if last == self.frame_index - 1 else 1
                item.stable_frames=hits
                self.track_hits[item.track_id]=(hits,self.frame_index)
            observations.extend(supports)
            for index, body in enumerate(bodies):
                if index in used:
                    continue
                identity = self.next_body_identity
                self.next_body_identity += 1
                left, top, right, bottom = body
                item = Observation(identity, 0.0, tuple(body), "body_support",
                                   (float((left+right)/2), float(bottom)), tuple(body),
                                   body_index=index)
                hits, last = self.track_hits.get(identity, (0, -1))
                item.stable_frames = hits + 1 if last == self.frame_index - 1 else 1
                self.track_hits[identity] = (item.stable_frames, self.frame_index)
                observations.append(item)
                used.add(index)
            for item in observations:
                if item.body_box is not None:
                    self.body_memory[item.track_id]=(item.body_box,self.frame_index)
            self.body_memory={identity:state for identity,state in self.body_memory.items()
                              if self.frame_index-state[1] <= self.body_reconnect_gap}
        if self.target == "player":
            for item in observations:
                item.body_box = item.box
        if self.teams is not None:
            self.teams.update(frame, observations, self.frame_index)
        for item in observations:
            if self.target == "helmet" and item.source == "helmet" and item.body_box is None:
                continue
            points, _ = self.history.get(item.track_id, (deque(), -1))
            box = item.body_box if item.body_box is not None else item.box
            current_anchor = ((box[0] + box[2]) / 2, box[3])
            anchor = smooth_anchor(points[-1] if points else None, current_anchor)
            points.append(anchor)
            self.history[item.track_id] = (points, self.frame_index)
            if self.target == "helmet" and item.body_box is not None:
                state = self.route_motion.get(item.track_id)
                height = max(1.0, item.body_box[3]-item.body_box[1])
                if state is None:
                    self.route_motion[item.track_id] = dict(
                        start=anchor, last=anchor, height=height, travel=0.0,
                        first_frame=self.frame_index)
                else:
                    state["travel"] += np.linalg.norm(np.asarray(anchor)-state["last"])
                    state["last"] = anchor
                    state["height"] = .8*state["height"] + .2*height
                    if self.frame_index-state["first_frame"] >= 12:
                        eligible = route_role_eligible(state, list(self.route_motion.values()))
                        (self.route_eligible_ids if eligible else self.route_excluded_ids).add(item.track_id)
                        if eligible:
                            self.route_excluded_ids.discard(item.track_id)
                        else:
                            self.route_eligible_ids.discard(item.track_id)
        # Missing tracks have no measured position. Never export guessed boxes.
        supported = {item.track_id for item in observations}
        missing=set()
        for track in self.engine.lost_stracks:
            engine_id=track.track_id
            canonical=self.id_aliases.get(engine_id,engine_id)
            if (engine_id not in self.superseded_ids and canonical in self.confirmed_ids
                    and canonical not in self.excluded and canonical not in supported):
                missing.add(canonical)
            active_ids = {item.track_id for item in observations}
            missing.update(identity for identity in self.confirmed_ids
                       if identity >= BODY_ID_BASE and identity not in active_ids
                       and identity in self.history
                       and self.frame_index - self.history[identity][1] <= self.engine.max_frames_lost
                       and identity not in self.excluded)
        self.missing=sorted(missing)
        self.track_hits = {identity: value for identity,value in self.track_hits.items()
                           if self.frame_index-value[1] <= self.engine.max_frames_lost}
        confirmed = [item for item in observations if item.stable_frames >= self.confirmation_frames
                     or item.track_id in self.confirmed_ids]
        self.confirmed_ids.update(item.track_id for item in confirmed)
        self.tentative_suppressed = len(observations)-len(confirmed)
        self._current_observations = confirmed
        return confirmed

    def move_history(self, transform):
        """Move existing path points with a validated background transform."""
        if transform is None:
            return
        for identity, (points, last_seen) in list(self.history.items()):
            if points:
                positions = [index for index, point in enumerate(points) if point is not None]
                if positions:
                    coordinates = np.asarray([points[index] for index in positions], dtype=np.float32)
                    moved = coordinates @ transform[:, :2].T + transform[:, 2]
                    points = list(points)
                    for index, point in zip(positions, moved):
                        points[index] = tuple(point)
                self.history[identity] = (deque(points), last_seen)
        for state in self.route_motion.values():
            for key in ("start", "last"):
                point = np.asarray(state[key], dtype=np.float32)
                state[key] = tuple(transform[:, :2] @ point + transform[:, 2])

    def move_identity_memory(self, transform):
        if transform is None:
            self.canonical_memory.clear()
            self.appearance_memory.clear()
            self.body_memory = {identity: state for identity, state in self.body_memory.items()
                                if identity >= BODY_ID_BASE}
            return
        for memory in (self.canonical_memory, self.body_memory):
            for identity, (box, last_seen) in list(memory.items()):
                x1, y1, x2, y2 = box
                corners = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]], np.float32)
                moved = cv2.transform(corners, transform)[0]
                memory[identity] = ((*moved.min(axis=0), *moved.max(axis=0)), last_seen)

    def reset(self):
        self.engine.reset()
        self.segment += 1
        self.history.clear()
        self.excluded.clear()  # IDs may be reused after a reset.
        self.missing.clear()
        self.track_hits.clear()
        self.body_memory.clear()
        self.next_body_identity = BODY_ID_BASE
        self.route_motion.clear()
        self.route_eligible_ids.clear()
        self.route_excluded_ids.clear()
        self.manual_trail_ids.clear()
        self.formation_trail_ids.clear()
        self.formation_roles.clear()
        self.confirmed_ids.clear()
        self.superseded_ids.clear()
        self.id_aliases.clear()
        self.canonical_memory.clear()
        self.appearance_memory.clear()
        self._current_observations = []
        if self.teams is not None:
            self.teams.reset()

    def trail_ids_to_draw(self):
        identities = set(self.confirmed_ids)
        if self.target == "helmet":
            if self.trail_scope == "manual":
                identities.intersection_update(self.manual_trail_ids)
            elif self.trail_scope == "formation":
                identities.intersection_update(self.formation_trail_ids)
            elif self.trail_scope == "eligible":
                identities.intersection_update(self.route_eligible_ids)
        return identities - self.excluded

    def apply_formation_slots(self, play, frame_shape):
        from src.tracking.formations import match_formation_slots

        assignments, unmatched = match_formation_slots(
            self._current_observations, play, frame_shape, self.team_names)
        self.formation_roles.update({identity: slot["position"]
                                     for identity, slot in assignments.items()})
        self.formation_trail_ids.update(assignments)
        return assignments, unmatched

    def draw(self, frame, observations, trails=False, confidence=False):
        image = frame.copy()
        if trails:
            for identity in self.trail_ids_to_draw():
                state = self.history.get(identity)
                if state is None:
                    continue
                points = [point for point in state[0] if point is not None]
                if len(points) > 1:
                    color = (80 + identity * 37 % 176, 80 + identity * 67 % 176,
                             80 + identity * 97 % 176)
                    cv2.polylines(image, [np.asarray(points, np.int32)], False,
                                  color, 2, cv2.LINE_AA)
        for item in observations:
            box = inferred_head_box(item.box) if item.source == "body_support" else item.box
            x1, y1, x2, y2 = map(int, box)
            identity = item.track_id
            color = ((80 + identity * 37 % 176, 80 + identity * 67 % 176,
                      80 + identity * 97 % 176) if identity is not None else (0, 220, 255))
            label = str(identity) if identity is not None else "det"
            if item.source == "helmet":
                label = "H" + label
                if item.ground_anchor is None:
                    label += "?"  # Field membership is unknown, not out of bounds.
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(image, center, 2, color, -1)
            elif item.source == "body_support":
                label = "B" + label
            if confidence:
                label += f" {item.confidence:.2f}"
            if self.teams is not None:
                label += f" {self.team_names.get(item.team, item.team)}"
            if identity in self.formation_roles:
                label += f" {self.formation_roles[identity]}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y = max(h + 3, y1)
            cv2.rectangle(image, (x1, y - h - 3), (x1 + w + 3, y + 2), (20, 20, 20), -1)
            cv2.putText(image, label, (x1 + 1, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        return image

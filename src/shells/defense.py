"""Defense-only detection adapter and conservative snapshot comparisons."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.tracking.teams import helmet_color, jersey_color, patch_color
from src.shells.baseline import classify_shell, ShellResult


def detection_color(frame, box, target, others=()):
    if target == "helmet":
        return helmet_color(frame, box)
    # Torso evidence is primary. A conservative upper-body patch also works for
    # crouched players, which the upright-only legacy jersey heuristic rejects.
    color = jersey_color(frame, box, others)
    if color != "unknown":
        return color
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    torso = (x1 + .2*w, y1 + .18*h, x2 - .2*w, y1 + .55*h)
    area = (torso[2]-torso[0])*(torso[3]-torso[1])
    if area <= 0:
        return "unknown"
    for a, b, c, d in others:
        overlap = max(0, min(c, torso[2])-max(a, torso[0])) * max(0, min(d, torso[3])-max(b, torso[1]))
        if overlap > .2*area:
            return "unknown"
    return patch_color(frame, torso)


class DefensiveDetector:
    """Filter evidence before the existing tracker sees it; never track offense.

    Uniform colors are supplied explicitly per game. Unknowns are rejected, not
    guessed. Full raw boxes remain available only for background-motion masking.
    """
    def __init__(self, model, color, target="player"):
        if color not in ("red", "white"):
            raise ValueError("Only reviewed red/white uniforms are supported")
        self.model, self.color, self.target = model, color, target
        self.polygon = None
        self.people = []
        self.counts = {}

    def predict(self, frame, **options):
        result = self.model.predict(frame, **options)[0]
        boxes = result.boxes.cpu().numpy()
        rows = boxes.xyxy.tolist()
        self.people = [SimpleNamespace(box=tuple(box)) for box in rows]
        accepted = []
        self.counts = dict(total=len(rows), defense=0, opponent=0, unknown=0, outside=0)
        for index, box in enumerate(rows):
            # Feet for bodies; measured head center for helmets. Neither is
            # substituted for the other in movement measurements.
            anchor = ((box[0]+box[2])/2, box[3] if self.target == "player" else (box[1]+box[3])/2)
            if self.polygon is None or cv2.pointPolygonTest(self.polygon, anchor, False) < 0:
                self.counts["outside"] += 1
                continue
            color = detection_color(frame, box, self.target, rows[:index]+rows[index+1:])
            if color == self.color:
                accepted.append(index)
                self.counts["defense"] += 1
            else:
                self.counts["unknown" if color == "unknown" else "opponent"] += 1
        return [SimpleNamespace(boxes=boxes[np.asarray(accepted, dtype=int)])]


def reference_points(points, transform):
    """Current pixels -> pre-snap reference pixels; no yard claim."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(values):
        return values
    return cv2.perspectiveTransform(values[None], np.asarray(transform, dtype=np.float64))[0]


def shell_from_snapshot(rows, geometry, visible, registration_ok):
    from .deep_safeties import deepest_safety_ids
    basis = np.asarray([geometry["depth_axis"], geometry["lateral_axis"]], float).T
    if abs(np.linalg.det(basis)) < .01:
        raise ValueError("Depth and lateral axes must be independent")
    origin = np.asarray(geometry["los_center"], float)
    points = []
    for row in rows:
        if row.get("anchor_kind") == "feet" and row.get("reference_point") is not None:
            depth, lateral = np.linalg.solve(basis, np.asarray(row["reference_point"])-origin)
            points.append(dict(depth=float(depth), lateral=float(lateral), track_id=row["track_id"]))
    # A candidate can be useful when some box/line defenders were missed, but
    # the accepted result stays unknown until all 11 defensive feet are observed.
    candidate = classify_shell(points, complete=visible and registration_ok,
                               **geometry["thresholds"])
    complete = visible and registration_ok and len(points) == 11
    accepted = candidate if complete else classify_shell(points, complete=False)
    deep_ids = deepest_safety_ids(points,geometry) if registration_ok else []
    selected=[p for p in points if p['track_id'] in deep_ids]
    if selected:
        candidate=classify_shell(selected,complete=visible and registration_ok,**geometry['thresholds'])
        if complete: accepted=candidate
    # A very deep post safety can drift outside the ordinary middle gate as the
    # camera pans. Recover only an extreme, isolated defender; ordinary vertical
    # corners do not satisfy both the depth multiplier and expanded middle gate.
    extreme_ids = [point["track_id"] for point in points
                   if point["depth"] > 1.6*geometry["thresholds"]["depth_threshold"]
                   and abs(point["lateral"]) < 1.5*geometry["thresholds"]["lateral_limit"]]
    if candidate.shell == "0_high" and len(extreme_ids) == 1:
        deep_ids = extreme_ids
        candidate = ShellResult("1_high",1,.55,
            "One extreme-depth post defender recovered outside the ordinary middle gate")
        if complete:
            accepted = candidate
    return dict(shell=accepted.shell, candidate=candidate.shell,
                confidence=accepted.confidence, reason=accepted.reason,
                candidate_reason=candidate.reason, measured_defenders=len(points),
                completeness_confirmed=complete, high_safety_count=len(deep_ids),
                high_safety_ids=deep_ids, geometry_points=points)


def classify_two_high_zone_structure(rows, geometry):
    """Distinguish Cover 2/4 cues after a two-high zone result.

    Cover 4 requires both wide corner candidates to have at least five yards of
    depth and no shallow outside defender occupying either flat. Otherwise the
    safer two-high zone estimate is Cover 2. This is structural evidence, not a
    verified responsibility assignment.
    """
    points=geometry_points_from_rows(rows,geometry)
    scale=geometry["pixels_per_yard"]
    converted=[dict(track_id=i,depth=p["depth"]/scale["depth"],
                    lateral=p["lateral"]/scale["lateral"]) for i,p in points.items()
               if p["anchor_kind"]=="feet"]
    if len(converted)<4:
        return dict(classification="unknown",reason="Too few comparable coverage defenders")
    wide=[p for p in converted if abs(p["lateral"])>=5]
    corners=[]
    for side in (-1,1):
        side_rows=[p for p in wide if p["lateral"]*side>0]
        if side_rows: corners.append(max(side_rows,key=lambda p:abs(p["lateral"])))
    if len(corners)<2:
        return dict(classification="cover_2_zone",corner_depths=[],open_flat_sides=[],
                    reason="Two-high zone without two deep wide corner candidates")
    corner_depths=[p["depth"] for p in corners]
    open=[]
    for side in (-1,1):
        flat=[p for p in converted if p["lateral"]*side>=5 and p["depth"]<5]
        if not flat: open.append(side)
    cover4=all(d>=5 for d in corner_depths) and len(open)==2
    return dict(classification="cover_4_zone" if cover4 else "cover_2_zone",
                corner_depths=corner_depths,open_flat_sides=open,
                reason=("Both corners have quarter depth and both flats are open" if cover4 else
                        "A corner is shallow or a defender occupies a flat; favor Cover 2"))


def geometry_points_from_rows(rows, geometry):
    basis = np.asarray([geometry["depth_axis"], geometry["lateral_axis"]], float).T
    if abs(np.linalg.det(basis)) < .01:
        raise ValueError("Depth and lateral axes must be independent")
    origin = np.asarray(geometry["los_center"], float)
    result = {}
    for row in rows:
        point = row.get("reference_point")
        if row.get("track_id") is None or point is None:
            continue
        depth, lateral = np.linalg.solve(basis, np.asarray(point)-origin)
        result[row["track_id"]] = dict(depth=float(depth), lateral=float(lateral),
                                       anchor_kind=row["anchor_kind"], row=row)
    return result


def classify_corner_movement(before, after, geometry, pixels_per_yard,
                             radius_yards=6.0, corner_ids=None):
    """Compare pre/post corner anchors; incomplete or tied evidence stays unknown."""
    if not 5 <= radius_yards <= 7:
        raise ValueError("Corner bubble radius must be between 5 and 7 yards.")
    pre = geometry_points_from_rows(before, geometry)
    post = geometry_points_from_rows(after, geometry)
    if corner_ids is None:
        thresholds = geometry["thresholds"]
        lateral_limit = thresholds["lateral_limit"]
        lateral_min = geometry.get("corner_lateral_min", .30*lateral_limit)
        corner_limit = geometry.get("corner_lateral_limit", 1.5*lateral_limit)
        depth_max = thresholds["depth_threshold"] + geometry.get("corner_depth_margin", 50)
        selected = {}
        for side in (-1, 1):
            candidates = [(abs(point["lateral"]), identity) for identity, point in pre.items()
                          if abs(point["lateral"]) >= lateral_min
                          and abs(point["lateral"]) <= corner_limit
                          and point["depth"] <= depth_max
                          and point["lateral"]*side > 0]
            if candidates:
                selected[side] = max(candidates)[1]
        corner_ids = [selected[side] for side in (-1, 1) if side in selected]
    else:
        corner_ids = list(corner_ids)

    movements = []
    depth_scale = float(pixels_per_yard["depth"])
    lateral_scale = float(pixels_per_yard["lateral"])
    if depth_scale <= 0 or lateral_scale <= 0:
        raise ValueError("Pixels-per-yard scales must be positive.")
    for identity in corner_ids:
        first, last = pre.get(identity), post.get(identity)
        row = dict(track_id=identity, state="matched", distance_yards=None,
                   stayed_within_bubble=False)
        if first is None or last is None:
            row["state"] = "missing_corner"
        elif first["anchor_kind"] != last["anchor_kind"]:
            row["state"] = "anchor_changed"
        else:
            depth_delta = last["depth"] - first["depth"]
            lateral_delta = last["lateral"] - first["lateral"]
            distance = math.hypot(depth_delta/depth_scale, lateral_delta/lateral_scale)
            row["distance_yards"] = distance
            row["stayed_within_bubble"] = distance <= radius_yards
        movements.append(row)

    total = len(corner_ids)
    measurable = [row for row in movements if row["distance_yards"] is not None]
    enough = total >= 2 and len(measurable) > total/2
    stays = sum(row["stayed_within_bubble"] for row in measurable)
    moves = len(measurable) - stays
    classification = "unknown"
    reason = "Need at least two identified corners and a majority with comparable pre/post anchors."
    if total >= 2 and len(measurable) <= total/2:
        reason = f"Only {len(measurable)} of {total} selected corners had comparable post-snap tracks."
    if enough and stays > total/2:
        classification = "zone_like"
        reason = f"A majority of identified corners stayed within {radius_yards:g} yards."
    elif enough and moves > total/2:
        classification = "man_like"
        reason = f"A majority of identified corners moved beyond {radius_yards:g} yards."
    return dict(classification=classification, reason=reason,
                radius_yards=radius_yards, selected_corner_ids=corner_ids,
                measurable_corners=len(measurable), movements=movements,
                bubble_ids=[row["track_id"] for row in measurable
                            if row["stayed_within_bubble"]] if classification == "zone_like" else [])


def zone_bubble_polygons(rows, corner_result, geometry, pixels_per_yard,
                         reference_to_current=None):
    """Create image-space ellipses for selected zone-like corners."""
    if corner_result["classification"] != "zone_like" or reference_to_current is None:
        return []
    points = geometry_points_from_rows(rows, geometry)
    origin = np.asarray(geometry["los_center"], float)
    depth_axis = np.asarray(geometry["depth_axis"], float)
    lateral_axis = np.asarray(geometry["lateral_axis"], float)
    depth_radius = corner_result["radius_yards"] * pixels_per_yard["depth"]
    lateral_radius = corner_result["radius_yards"] * pixels_per_yard["lateral"]
    angles = np.linspace(0, 2*math.pi, 49)
    polygons = []
    for identity in corner_result["bubble_ids"]:
        point = points.get(identity)
        if point is None:
            continue
        center = origin + depth_axis*point["depth"] + lateral_axis*point["lateral"]
        reference = np.asarray([
            center + depth_axis*depth_radius*math.cos(angle)
            + lateral_axis*lateral_radius*math.sin(angle)
            for angle in angles
        ], dtype=np.float32)
        current = cv2.perspectiveTransform(reference[None],
                                           np.asarray(reference_to_current, np.float64))[0]
        polygons.append((identity, np.round(current).astype(np.int32)))
    return polygons


def compare_snapshots(before, after, max_displacement=500):
    pre = {p["track_id"]: p for p in before}
    post = {p["track_id"]: p for p in after}
    result = []
    for identity in sorted(pre.keys() | post.keys()):
        a, b = pre.get(identity), post.get(identity)
        row = dict(track_id=identity, state="matched", delta_reference_pixels=None,
                   distance_reference_pixels=None)
        if a is None or b is None:
            row["state"] = "new_after_snap" if a is None else "missing_after_snap"
        elif a["anchor_kind"] != b["anchor_kind"]:
            row["state"] = "anchor_changed"
        elif a["reference_point"] is None or b["reference_point"] is None:
            row["state"] = "camera_registration_unavailable"
        else:
            delta = np.asarray(b["reference_point"]) - a["reference_point"]
            distance = float(np.linalg.norm(delta))
            if distance > max_displacement:
                row["state"] = "suspected_id_switch"
            else:
                row["delta_reference_pixels"] = delta.tolist()
                row["distance_reference_pixels"] = distance
        result.append(row)
    return result


def load_coverage_reference(path):
    """Read the structured football reference cues used by the estimator."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (data.get("version") != 1 or data.get("coverage") != "cover_1_man"
            or not isinstance(data.get("shell_requirements"), dict)
            or not isinstance(data.get("man_evidence"), dict)):
        raise ValueError("Unsupported or incomplete Cover 1 Man reference rules.")
    return data


def load_coverage_playbook(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("coverages"), dict):
        raise ValueError("Unsupported or incomplete coverage reference playbook.")
    return data


def coverage_hypotheses(post_shell, corner_motion=None, high_safety_count=None,
                        pre_shell=None, reference=None, playbook=None, zone_structure=None,
                        shell_complete=True):
    candidates = {"0_high": ["cover_0_man"],
                  "1_high": ["cover_1_man", "cover_3_zone"],
                  "2_high": ["cover_2_man", "cover_2_zone", "cover_4_zone"]}
    classification = (corner_motion or {}).get("classification", "unknown")
    if post_shell == "0_high" and shell_complete is not True:
        post_shell = "unknown"
    if post_shell == "2_high" and classification == "zone_like" and zone_structure:
        name=zone_structure.get("classification")
        if name in ("cover_2_zone","cover_4_zone"):
            return dict(coverage=f"{name}_candidate",confidence=.55,candidates=[name],
                        status="provisional_structure_match",reason=zone_structure["reason"],
                        measured_signals=["two_high",classification,name])
    reference_candidates = []
    if playbook is not None:
        for name, rule in playbook["coverages"].items():
            if post_shell not in rule.get("shell_candidates", []):
                continue
            required_motion = rule.get("corner_motion")
            if required_motion is None or classification != required_motion:
                continue
            required_safeties = rule.get("high_safety_count")
            if required_safeties is not None and high_safety_count != required_safeties:
                continue
            reference_candidates.append(name)
    if reference_candidates:
        unique = len(reference_candidates) == 1
        name = reference_candidates[0] if unique else None
        return dict(coverage=f"{name}_candidate" if unique else "unknown_coverage",
                    confidence=0.55 if unique else 0.0,
                    status="provisional_reference_match" if unique else "ambiguous_reference_match",
                    candidates=reference_candidates,
                    reason="Candidate matches saved shell and corner-motion reference cues; unmeasured assignment cues prevent a confirmed coverage call.")
    if reference is not None:
        shell_rules = reference["shell_requirements"]
        man_rules = reference["man_evidence"]
        shell_matches = post_shell == shell_rules["accepted_post_shell"]
        safety_matches = high_safety_count == shell_rules["deep_middle_safety_count"]
        pre_allowed = pre_shell is None or pre_shell in shell_rules["allow_pre_shells"]
        corners_match = classification == man_rules["corner_motion"]
        if shell_matches and safety_matches and pre_allowed and corners_match:
            return dict(coverage="cover_1_man_candidate", confidence=0.55,
                        candidates=["cover_1_man"], status="provisional",
                        measured_signals=["post_snap_one_high", "one_deep_middle_safety",
                                          "corner_majority_man_like"],
                        unmeasured_signals=reference["cues_not_yet_measured"],
                        reason="Shell, safety count, and corner movement match the saved Cover 1 Man reference; receiver assignments remain unverified.")
    if classification in ("zone_like", "man_like"):
        return dict(coverage=classification, confidence=0.6,
                    candidates=candidates.get(post_shell, []),
                    reason=(corner_motion.get("reason", "") +
                            " This is a corner-motion heuristic, not a verified coverage call."))
    return dict(coverage="unknown_coverage", confidence=0.0,
                candidates=candidates.get(post_shell, []),
                reason=(corner_motion or {}).get("reason",
                        "Insufficient corner movement evidence; shell-compatible hypotheses are not classifications."))

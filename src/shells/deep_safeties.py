"""Depth-based safety candidates, independent of a player's nominal position."""
import cv2
import numpy as np
from .defense import detection_color


def deepest_safety_ids(points, geometry):
    thresholds=geometry['thresholds']
    minimum=thresholds['depth_threshold']
    scale=geometry.get('pixels_per_yard',{}).get('depth',minimum/10)
    ball=geometry.get('ball_reference_point',geometry['los_center'])
    basis=np.array([geometry['depth_axis'],geometry['lateral_axis']],float).T
    ball_depth,ball_lateral=np.linalg.solve(basis,np.array(ball)-geometry['los_center'])
    lateral_scale=geometry.get('pixels_per_yard',{}).get('lateral',scale)
    candidates=[p for p in points if p['depth']>minimum
                and abs(p['lateral'])<thresholds['lateral_limit']*(1.5 if p['depth']>1.6*minimum else 1)
                and np.hypot((p['depth']-ball_depth)/scale,
                             (p['lateral']-ball_lateral)/lateral_scale)>=minimum/scale]
    if not candidates: return []
    deepest=max(p['depth'] for p in candidates)
    # Underneath defenders may clear the absolute threshold after the snap.
    # Only the deepest tier supplies safety candidates; do not force a count.
    return [p['track_id'] for p in candidates if p['depth']>=deepest-3*scale]


def deepest_image_safety_ids(rows, geometry):
    """Select the deepest defensive tier when camera registration is unavailable.

    Translation from camera panning does not change relative depth ordering. The
    defensive front supplies the local LOS proxy, so this fallback does not
    depend on the stale pre-snap image origin.
    """
    rows=[r for r in rows if r.get('source')=='fresh_frame_detection']
    if not rows:
        return []
    axis=np.asarray(geometry['depth_axis'],float)
    axis/=np.linalg.norm(axis)
    scale=float(geometry['pixels_per_yard']['depth'])
    scored=[(float(np.dot(np.asarray(r['image_point']),axis)),r['track_id']) for r in rows]
    depths=np.asarray([depth for depth,_ in scored])
    deepest=float(depths.max())
    front=float(np.percentile(depths,25))
    # A safety must be meaningfully removed from the defensive front/ball area.
    if deepest-front < 7*scale:
        return []
    return [identity for depth,identity in scored if depth>=deepest-5*scale]


def refresh_defenders(frame, rows, model, defense_color, replacement_rows=None):
    """Recover missed bodies at the freeze frame; retain the saved coordinate frame.

    The fitted registration is approximate. Without two comparable tracked feet,
    no new reference positions are invented. Fresh IDs are local to this frame.
    """
    anchors=[r for r in rows if r.get('reference_point') is not None and r.get('anchor_kind')=='feet']
    matrix=None
    if len(anchors)>=2:
        matrix,inliers=cv2.estimateAffinePartial2D(np.float32([r['image_point'] for r in anchors]),
                                                np.float32([r['reference_point'] for r in anchors]))
    refreshed=[dict(r) for r in (replacement_rows if replacement_rows is not None else rows)]
    if matrix is not None:
        for r in refreshed:
            r['reference_point']=(matrix@np.array([*r['image_point'],1])).tolist()
            r['anchor_kind']='feet'
    boxes=model.predict(frame,imgsz=1536,conf=.18,classes=[0],verbose=False)[0].boxes.xyxy.cpu().numpy().tolist()
    field_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(field_hsv)
    turf_mask=(h>=25)&(h<=95)&(s>=45)&(v>=40)
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=box
        if y2-y1<20: continue
        cx=(x1+x2)/2
        left=max(0,int(cx)-60);right=min(frame.shape[1],int(cx)+60)
        local_rows=turf_mask[:,left:right].mean(axis=1)
        local_smooth=np.convolve(local_rows,np.ones(5)/5,mode='same')
        field_rows=np.flatnonzero(local_smooth>=.42)
        if len(field_rows) and y2 < int(field_rows[0])+3: continue
        # Exclude bench personnel and officials on the sideline: feet must be
        # surrounded by turf, not just inside the full video rectangle.
        patch=frame[max(0,int(y2)-10):min(frame.shape[0],int(y2)+10),
                    max(0,int(cx)-22):min(frame.shape[1],int(cx)+22)]
        if not patch.size: continue
        h,s,v=cv2.split(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV))
        if np.mean((h>=25)&(h<=95)&(s>=45)&(v>=40))<.4: continue
        color=detection_color(frame,box,'player',boxes[:i]+boxes[i+1:])
        # Distant Chiefs jerseys contain large white numbers, so a generic
        # white-majority classifier can call them white. Meaningful saturated
        # red upper-body evidence overrides that result only for a red defense.
        torso=frame[max(0,int(y1)):int(y1+.65*(y2-y1)),max(0,int(x1)):int(x2)]
        if defense_color=='red' and torso.size:
            h,s,v=cv2.split(cv2.cvtColor(torso,cv2.COLOR_BGR2HSV))
            if np.mean(((h<=12)|(h>=165))&(s>=70)&(v>=55))>=.08:
                color='red'
        if color!=defense_color: continue
        point=np.array([(x1+x2)/2,y2])
        if any(np.linalg.norm(point-r['image_point'])<22 for r in refreshed): continue
        refreshed.append(dict(track_id=f'freeze_{i}',image_point=point.tolist(),
            reference_point=((matrix@np.array([*point,1])).tolist() if matrix is not None else None),
            anchor_kind='feet',
            source='fresh_frame_detection',box=box))
    found=any(r.get('source')=='fresh_frame_detection' for r in refreshed)
    return refreshed,found,matrix is not None

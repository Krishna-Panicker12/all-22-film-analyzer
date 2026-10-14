"""Evidence-aware freeze-frame coverage diagrams (illustrations, not measured assignments)."""
import cv2
import numpy as np

ZONE_ROLES={'CB','S','FS','SS','DB','NICKEL','LB_or_DB'}


def zone_box(point,geometry,width_yards=8):
    scale=geometry['pixels_per_yard']
    depth=np.array(geometry['depth_axis'],float)*scale['depth']*width_yards/2
    lateral=np.array(geometry['lateral_axis'],float)*scale['lateral']*width_yards/2
    center=np.array(point,float)
    return np.rint([center-depth-lateral,center+depth-lateral,center+depth+lateral,center-depth+lateral]).astype(np.int32)


def responsibility_overlay(frame,coverage,defenders,receivers,geometry,pairs=(),width_yards=8,high_safety_ids=()):
    canvas=frame.copy();drawn=[]
    high_safety_ids=set(high_safety_ids)
    if 'zone' in coverage or high_safety_ids:
        tint=canvas.copy()
        for d in defenders:
            # Never turn an arbitrary surviving track into a zone defender. The
            # earlier implementation did this and painted a lineman. A box now
            # requires a reviewed or model-supplied coverage role.
            high=d['track_id'] in high_safety_ids
            if not high and ('zone' not in coverage or d.get('role') not in ZONE_ROLES): continue
            box_width=15 if high else width_yards
            polygon=zone_box(d['image_point'],geometry,box_width)
            cv2.fillConvexPoly(tint,polygon,(110,35,5))
            drawn.append(dict(defender_id=d['track_id'],kind='illustrative_zone',polygon=polygon.tolist(),
                              width_yards=box_width,high_safety=high))
        canvas=cv2.addWeighted(tint,.40,canvas,.60,0)
        for d in drawn: cv2.polylines(canvas,[np.array(d['polygon'],np.int32)],True,(180,75,20),2)
    if 'man' in coverage:
        ds={d['track_id']:d for d in defenders};rs={r['track_id']:r for r in receivers}
        for p in pairs:
            if p.get('status')!='following_candidate': continue
            d=ds.get(p['defender_id']);r=rs.get(p['receiver_candidate_id'])
            if d is None or r is None: continue
            a=tuple(map(round,d['image_point']));b=tuple(map(round,r['image_point']))
            cv2.line(canvas,a,b,(0,225,255),3,cv2.LINE_AA)
            drawn.append(dict(defender_id=d['track_id'],receiver_id=r['track_id'],kind='following_candidate'))
    for d in defenders:
        if not any(o['defender_id']==d['track_id'] for o in drawn): continue
        if 'zone' in coverage and d.get('role') not in ZONE_ROLES and d['track_id'] not in high_safety_ids: continue
        point=tuple(map(round,d['image_point']));cv2.circle(canvas,point,6,(0,220,255),2)
    has_high=any(o.get('high_safety') for o in drawn)
    has_lines=any(o['kind']=='following_candidate' for o in drawn)
    label=(f'Illustrative zones: {width_yards:g} yards wide, approximate scale' if 'zone' in coverage else
           'Lines = observed following candidates' if has_lines else 'No reliable matchup lines available')
    if has_high:
        label='High safety boxes: 15 yards wide (approx.) | '+(
            f'Other zones: {width_yards:g} yards' if 'zone' in coverage else label)
    cv2.rectangle(canvas,(0,canvas.shape[0]-34),(canvas.shape[1],canvas.shape[0]),(20,24,30),-1)
    cv2.putText(canvas,label,(12,canvas.shape[0]-10),0,.6,(255,255,255),1,cv2.LINE_AA)
    return canvas,drawn

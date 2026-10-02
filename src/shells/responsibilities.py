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


def responsibility_overlay(frame,coverage,defenders,receivers,geometry,pairs=(),width_yards=8):
    canvas=frame.copy();drawn=[]
    if 'zone' in coverage:
        tint=canvas.copy()
        for d in defenders:
            # Never turn an arbitrary surviving track into a zone defender. The
            # earlier implementation did this and painted a lineman. A box now
            # requires a reviewed or model-supplied coverage role.
            if d.get('role') not in ZONE_ROLES: continue
            polygon=zone_box(d['image_point'],geometry,width_yards)
            cv2.fillConvexPoly(tint,polygon,(110,35,5))
            drawn.append(dict(defender_id=d['track_id'],kind='illustrative_zone',polygon=polygon.tolist()))
        canvas=cv2.addWeighted(tint,.40,canvas,.60,0)
        for d in drawn: cv2.polylines(canvas,[np.array(d['polygon'],np.int32)],True,(180,75,20),2)
    elif 'man' in coverage:
        ds={d['track_id']:d for d in defenders};rs={r['track_id']:r for r in receivers}
        for p in pairs:
            if p.get('status')!='following_candidate': continue
            d=ds.get(p['defender_id']);r=rs.get(p['receiver_candidate_id'])
            if d is None or r is None: continue
            a=tuple(map(round,d['image_point']));b=tuple(map(round,r['image_point']))
            cv2.line(canvas,a,b,(0,225,255),3,cv2.LINE_AA)
            drawn.append(dict(defender_id=d['track_id'],receiver_id=r['track_id'],kind='following_candidate'))
    for d in defenders:
        if 'zone' in coverage and d.get('role') not in ZONE_ROLES: continue
        point=tuple(map(round,d['image_point']));cv2.circle(canvas,point,6,(0,220,255),2)
    label=(f'Illustrative zones: {width_yards:g} yards wide, approximate scale' if 'zone' in coverage else
           'Lines = observed following candidates, not confirmed assignments' if drawn else 'No reliable matchup lines available')
    cv2.rectangle(canvas,(0,canvas.shape[0]-34),(canvas.shape[1],canvas.shape[0]),(20,24,30),-1)
    cv2.putText(canvas,label,(12,canvas.shape[0]-10),0,.6,(255,255,255),1,cv2.LINE_AA)
    return canvas,drawn

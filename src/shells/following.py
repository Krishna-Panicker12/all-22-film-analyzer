"""Conservative receiver-candidate following evidence from camera-corrected tracks."""
import numpy as np
from .defense import geometry_points_from_rows


def receiver_following(frames, snap_frame, fps, geometry, defender_ids=None):
    """Compare same-ID paths, never equating following with a confirmed man call.

    Opposing-color offensive tracks that release downfield are receiver candidates.
    Coordinates are approximate yards from the existing reviewed local geometry.
    Require temporal coverage, small gaps, sustained proximity, and correlated
    multi-frame motion. Competing receivers or defenders make matching ambiguous.
    """
    if fps <= 0: raise ValueError('FPS must be positive')
    window=[f for f in frames if f['frame_index'] > snap_frame]
    result=dict(status='unknown',pairs=[],classification='unknown',
        diagnostics=dict(post_snap_frames=len(window),registered_frames=0,
                         defender_tracks=0,receiver_candidate_tracks=0,
                         pairs_rejected_for_track_coverage=0),
        limitations=['Offensive release tracks are receiver candidates, not verified eligible receivers.',
                    'Local yard scale and camera compensation are approximate.',
                    'Following can occur in match-zone; absence of following is not zone proof.'])
    if not window: return result
    tracks={'defenders':{},'receivers':{}}
    scale=geometry['pixels_per_yard']
    for frame in window:
        if not frame.get('registration_ok',False): continue
        result['diagnostics']['registered_frames'] += 1
        for side in tracks:
            for identity,p in geometry_points_from_rows(frame.get(side,[]),geometry).items():
                if p['anchor_kind'] != 'feet': continue
                if side=='defenders' and defender_ids is not None and identity not in defender_ids: continue
                xy=np.array([p['depth']/scale['depth'],p['lateral']/scale['lateral']])
                if np.isfinite(xy).all():
                    tracks[side].setdefault(identity,{})[frame['frame_index']]=xy
    result['diagnostics']['defender_tracks']=len(tracks['defenders'])
    result['diagnostics']['receiver_candidate_tracks']=len(tracks['receivers'])
    result['diagnostics']['longest_defender_track_frames']=max((len(p) for p in tracks['defenders'].values()),default=0)
    result['diagnostics']['longest_receiver_track_frames']=max((len(p) for p in tracks['receivers'].values()),default=0)
    candidates={}
    for did,dpath in tracks['defenders'].items():
        scored=[]
        for rid,rpath in tracks['receivers'].items():
            common=sorted(dpath.keys() & rpath.keys())
            if (len(common)<max(8,.7*len(window)) or
                    (common[-1]-common[0])/fps < 1 or max(np.diff(common))/fps > .2):
                result['diagnostics']['pairs_rejected_for_track_coverage'] += 1
                continue
            d=np.array([dpath[i] for i in common]); r=np.array([rpath[i] for i in common])
            # Suppress stationary linemen/QB and tiny movement correlated by camera jitter.
            route=float(np.linalg.norm(r[-1]-r[0])); travel=float(np.linalg.norm(d[-1]-d[0]))
            if route < 3 or r[-1,0]-r[0,0] < 1.5 or travel < 2: continue
            lag=max(1,round(.2*fps)); pairs=[(i,j) for i in common for j in [i+lag] if j in dpath and j in rpath]
            if len(pairs)<5: continue
            dv=np.array([dpath[j]-dpath[i] for i,j in pairs]); rv=np.array([rpath[j]-rpath[i] for i,j in pairs])
            dn=np.linalg.norm(dv,axis=1); rn=np.linalg.norm(rv,axis=1); moving=(dn>.25)&(rn>.25)
            if moving.sum()<5: continue
            alignment=float(np.mean(np.sum(dv[moving]*rv[moving],axis=1)/(dn[moving]*rn[moving])))
            distance=np.linalg.norm(d-r,axis=1)
            proximity=float(np.mean(distance<=5)); spread=float(np.std(distance))
            follows=alignment>=.75 and proximity>=.8 and spread<=2
            if follows:
                scored.append(dict(defender_id=did,receiver_candidate_id=rid,
                    alignment=alignment,proximity_fraction=proximity,separation_std_yards=spread,
                    median_separation_yards=float(np.median(distance)),duration_seconds=(common[-1]-common[0])/fps,
                    samples=len(common),score=alignment*proximity/(1+spread),status='following_candidate'))
        scored.sort(key=lambda p:p['score'],reverse=True)
        if scored:
            best=scored[0]
            if len(scored)>1 and best['score']-scored[1]['score']<.15:
                best['status']='ambiguous_receiver'
            candidates[did]=best
    pairs=list(candidates.values())
    for p in pairs:
        if sum(q['receiver_candidate_id']==p['receiver_candidate_id'] for q in pairs)>1:
            p['status']='ambiguous_shared_receiver'
    accepted=[p for p in pairs if p['status']=='following_candidate']
    result.update(pairs=pairs,status='measured' if pairs else 'unknown',
        classification='following_evidence' if accepted else 'unknown')
    return result

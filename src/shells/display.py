"""Large readable estimates and explicit manual playback controls."""
import cv2
from .coverage_stages import FAMILIES


def pretty(name):
    return str(name).replace('_candidate','').replace('_',' ').upper()


def display_estimate(shell=None, coverage=None, review=None, post_snap=False):
    shell=shell or {}; coverage=coverage or {}; review=review or {}
    shell_name=shell.get('candidate','unknown')
    if shell_name=='0_high' and not shell.get('completeness_confirmed',False):
        shell_name='unknown'  # Missing detections cannot establish no deep safety.
    shell_source='tracking candidate'
    if shell_name=='unknown' and review.get('pre_shell','unknown')!='unknown':
        shell_name=review['pre_shell']; shell_source='pre-snap film review'
    names=coverage.get('candidates',[]) if post_snap else FAMILIES.get(shell_name,[])
    source='tracking candidate' if post_snap else 'shell-compatible possibilities'
    best=coverage.get('coverage','unknown_coverage')
    # Never display a coverage whose required deep shell contradicts the shell
    # printed directly above it. This prevents one-high + Cover 0 combinations.
    required={'cover_0_man':'0_high','cover_1_man':'1_high','cover_1_man_candidate':'1_high',
              'cover_2_man':'2_high','cover_2_man_candidate':'2_high',
              'cover_2_zone':'2_high','cover_2_zone_candidate':'2_high',
              'cover_4_zone':'2_high','cover_4_zone_candidate':'2_high',
              'cover_3_zone':'1_high','cover_3_zone_candidate':'1_high'}
    incompatible=required.get(best) not in (None,shell_name)
    if incompatible: best='unknown_coverage'
    if post_snap and best not in ('unknown_coverage','zone_like','man_like'):
        estimate=pretty(best)
    elif (post_snap and review.get('coverage','unknown_coverage')!='unknown_coverage'
          and required.get(review.get('coverage')) in (None,shell_name)):
        estimate=pretty(review['coverage']); source='WHOLE-PLAY FILM REVIEW candidate; automatic estimate unresolved'
    elif post_snap and best in ('man_like','zone_like'):
        estimate=pretty(best)+'; COVERAGE FAMILY UNKNOWN'
        source='corner-motion heuristic; provisional'
    else:
        estimate=' / '.join(pretty(n) for n in names) or 'UNKNOWN - insufficient evidence'
    return dict(shell=pretty(shell_name),estimate=estimate,shell_source=shell_source,source=source)


def banner(frame, estimate, play_id, seconds, held=False):
    height=220
    canvas=cv2.copyMakeBorder(frame,height,0,0,0,cv2.BORDER_CONSTANT,value=(20,24,30))
    width=canvas.shape[1]
    lines=[(f"COVERAGE SHELL: {estimate['shell']}",48,1.3,(0,235,255)),
           (f"BEST ESTIMATE: {estimate['estimate']}",102,1.25,(255,255,255)),
           (f"{estimate['shell_source']} | {estimate['source']}",142,.65,(190,200,210)),
           (f"{play_id} | snap {seconds:+.2f}s | "+('PAUSED - Q: next play / close | ESC: quit batch' if held else 'SPACE: pause | Q: next play / close | ESC: quit batch'),188,.7,(100,230,150))]
    for text,y,size,color in lines:
        measured=cv2.getTextSize(text,0,size,2)[0][0]
        size=min(size,size*(width-32)/max(1,measured))
        cv2.putText(canvas,text,(16,y),0,size,color,2,cv2.LINE_AA)
    return canvas


def wait_for_manual_exit(window):
    """Only Q advances, Escape or closing the window aborts the batch."""
    while cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>=1:
        key=cv2.waitKey(100)&0xFF
        if key in (ord('q'),ord('Q')): return 'next'
        if key==27: return 'quit'
    return 'quit'

"""Ordered, provisional coverage estimates; shell is not a play-call label."""

FAMILIES = {
    '0_high': ['cover_0_man'],
    '1_high': ['cover_1_man', 'cover_3_zone'],
    '2_high': ['cover_2_man', 'cover_2_zone', 'cover_4_zone', 'cover_6_zone'],
}


def staged_coverage(pre, post, corner_motion, playbook=None, following=None):
    """Consume tracked snapshots, preserving unknowns and unverified roles.

    Safety identities are geometric candidates. The rotation is only measured
    when pre-snap high candidates have comparable post-snap geometry points.
    Corner displacement supplies weak evidence, not receiver assignment proof.
    """
    def estimate(snapshot):
        shell = snapshot.get('candidate', 'unknown')
        return dict(shell=shell, safety_candidate_ids=snapshot.get('high_safety_ids', []),
                    candidates=list(FAMILIES.get(shell, [])),
                    status='provisional' if shell in FAMILIES else 'unknown')

    before, after = estimate(pre), estimate(post)
    pre_ids = set(before['safety_candidate_ids'])
    post_ids = set(after['safety_candidate_ids'])
    observed = {p['track_id'] for p in post.get('geometry_points', [])}
    comparable = bool(pre_ids) and pre_ids <= observed
    rotation = dict(status='measured_candidate' if comparable else 'unknown',
                    stayed_high_ids=sorted(pre_ids & post_ids) if comparable else [],
                    dropped_from_high_ids=sorted(pre_ids - post_ids) if comparable else [],
                    new_high_ids=sorted(post_ids - pre_ids) if comparable else [],
                    shell_transition=before['shell']+' -> '+after['shell'])
    motion = corner_motion.get('classification', 'unknown')
    candidates = list(after['candidates'])
    if motion == 'man_like':
        candidates = [c for c in candidates if c.endswith('_man')]
    elif motion == 'zone_like':
        candidates = [c for c in candidates if c.endswith('_zone')]
    following = following or {'classification':'unknown','pairs':[]}
    follows = following.get('classification') == 'following_evidence'
    if follows:
        # A measured follow can contradict a small-radius zone heuristic.
        # Retain match-zone alternatives instead of forcing a man classification.
        candidates = list(after['candidates'])
    references = []
    for name, rule in (playbook or {}).get('coverages', {}).items():
        if after['shell'] in rule.get('shell_candidates', []):
            references.append(dict(coverage=name, images=rule.get('reference_images', []),
                                   unmeasured=rule.get('unmeasured_assignments', [])))
    return dict(pre_snap=before, post_snap=dict(**after, rotation=rotation),
                corner_refinement=dict(candidates=candidates, motion=motion,
                    receiver_following=following,
                    status='provisional' if candidates else 'unknown',
                    alternative_candidates=after['candidates'],
                    reason=('Sustained receiver-candidate following observed; man and match-zone remain possible.' if follows else
                            'No reliable receiver-follow evidence; displacement remains weak evidence and absence does not establish zone.')),
                reference_context=references,
                limitations=['Geometric safety candidates are not verified roster positions.',
                             'Match-zone can follow receivers; man defenders may move little.',
                             'Buzz, cloud, and invert require assignment evidence beyond shell counts.'])

# Helper-only regression. Main extraction NEVER uses these minima.
import itertools
import numpy as np
import legacy as fa


def exact_plateau_runs(signal_values):
    y = np.asarray(signal_values)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('Expected a finite one-dimensional signal')
    runs = []
    a, n = 0, len(y)
    while a < n:
        b = a
        while b + 1 < n and y[b + 1] == y[a]:
            b += 1
        rep = (a + b) // 2
        left = a > 0
        right = b + 1 < n
        trough = bool(left and right and y[a - 1] > y[a] and y[b + 1] > y[a])
        runs.append({'bounds_inclusive': [a, b], 'value': float(y[a]), 'value_hex': float(y[a]).hex(),
                     'representative': rep, 'left_exterior_exists': left, 'right_exterior_exists': right,
                     'left_value_hex': float(y[a-1]).hex() if left else None,
                     'right_value_hex': float(y[b+1]).hex() if right else None,
                     'is_trough': trough, 'representative_in_margin': 10 <= rep <= n - 11,
                     'eligible_minimum': trough and 10 <= rep <= n - 11})
        a = b + 1
    return runs


def ranked_pairs(minima, lag, n):
    pairs = [(a, b) for a in minima for b in minima if 80 <= b - a <= 200 and abs(b - a - lag) <= 0.1 * lag]
    pair = min(pairs, key=lambda ab: (abs(ab[1] - ab[0] - lag), abs((ab[0] + ab[1]) / 2 - (n - 1) / 2), ab[0], ab[1])) if pairs else None
    return [list(p) for p in pairs], None if pair is None else list(pair)


def plateau_minima_and_pair(y, lag):
    minima = [r['representative'] for r in exact_plateau_runs(y) if r['eligible_minimum']]
    pairs, pair = ranked_pairs(minima, lag, len(y))
    return minima, pair, len(pairs)


def independent_groupby_reference(y):
    """Independent maximal-group reference, no candidate calls."""
    if np.ndim(y) != 1 or not np.isfinite(y).all():
        raise ValueError('nonfinite/non1D')
    answer = []
    for value, grouped in itertools.groupby(enumerate(y), key=lambda item: item[1]):
        indices = [item[0] for item in grouped]
        first, last = indices[0], indices[-1]
        middle = indices[(len(indices) - 1) // 2]
        if first > 0 and last < len(y) - 1 and value < min(y[first-1], y[last+1]) and 10 <= middle < len(y)-10:
            answer.append(middle)
    return answer


def synthetic_helper_checks():
    cases = []
    def add(name, edits, expected):
        y = np.full(32, 3.0)
        for a, b, value in edits:
            y[a:b+1] = value
        runs = exact_plateau_runs(y)
        actual = [r['representative'] for r in runs if r['eligible_minimum']]
        reference = independent_groupby_reference(y)
        assert actual == reference == expected, name
        strict = fa.minima_and_pair(y, 80)[0]
        singleton = [r['representative'] for r in runs if r['eligible_minimum'] and r['bounds_inclusive'][0] == r['bounds_inclusive'][1]]
        assert strict == singleton, name
        cases.append({'name': name, 'input': y.tolist(), 'expected': expected, 'candidate': actual, 'independent_reference': reference, 'pass': True})
    add('constant', [], [])
    add('singleton', [(15, 15, 1)], [15])
    add('even_plateau', [(14, 15, 1)], [14])
    add('odd_plateau', [(14, 16, 1)], [15])
    add('separated_equal_troughs', [(12, 12, 1), (18, 18, 1)], [12, 18])
    add('boundary_runs', [(0, 1, 1), (30, 31, 1)], [])
    add('left_margin_outside', [(9, 10, 1)], [])
    add('left_margin_inside', [(10, 11, 1)], [10])
    add('right_margin_inside', [(21, 22, 1)], [21])
    add('right_margin_outside', [(22, 23, 1)], [])
    add('margin_spanning_left', [(8, 13, 1)], [10])
    add('margin_spanning_right', [(20, 23, 1)], [21])
    add('equal_shoulder_not_below_both', [(14, 15, 1), (16, 16, 1), (17, 17, 0)], [17])
    for name, value in (('nonfinite_nan', np.nan), ('nonfinite_posinf', np.inf), ('nonfinite_neginf', -np.inf)):
        y = np.full(32, 3.0)
        y[15] = value
        rejected = []
        for helper in (exact_plateau_runs, independent_groupby_reference):
            try:
                helper(y)
            except ValueError:
                rejected.append(True)
        assert rejected == [True, True], name
        cases.append({'name': name, 'nonfinite_at_index': 15, 'value_label': name.removeprefix('nonfinite_'),
                      'candidate_and_reference_reject': True, 'pass': True})
    return {'status': 'pass', 'fixed_case_count': len(cases), 'cases': cases,
            'scope': 'helper conventions only; no fixture-panel expansion and no old correlation-nearmax checks'}

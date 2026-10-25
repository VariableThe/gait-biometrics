#!/usr/bin/env python3
"""One frozen, exclusively reserved, outcome-informed 12-fixture regression."""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'cycle-length-feasibility-r1'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for key in THREAD_KEYS:
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
START_WALL = time.perf_counter()

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import platform
import signal
import traceback
import types

PREP_CPU = json.loads((ROOT / 'preparation.json').read_text())['process_cpu_seconds']
IS_RUN = '--run' in sys.argv
FREEZE_CPU = json.loads((ROOT / 'freeze.json').read_text())['process_cpu_seconds'] if IS_RUN else 0.0
HARD_CPU = min(39, math.floor(59 - PREP_CPU - FREEZE_CPU)) if IS_RUN else 10
assert HARD_CPU >= 5, 'Insufficient remaining cumulative CPU budget; preserve directory'
resource.setrlimit(resource.RLIMIT_CPU, (HARD_CPU - 1, HARD_CPU))


def cpu_signal(_signum, _frame):
    raise TimeoutError('CPU cap reached; preserve exclusive attempt and partial evidence')


signal.signal(signal.SIGXCPU, cpu_signal)

import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
import frozen_algorithm as fa

ARMS = ('strict', 'plateau')
PREREG = ('prepare.py', 'runner.py', 'frozen_algorithm.py', 'README.md', 'provenance.json',
          'cached_panel.json', 'input_windows.npy', 'cached_correlations.npy', 'cached_features.npy',
          'cached_feature_valid_mask.npy', 'old_files_snapshot.json', 'preparation.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def typed(value):
    """Exact JSON types, float bits (including -0), and strict string keys."""
    kind = type(value)
    if kind is dict:
        assert all(type(k) is str for k in value), 'JSON keys must already be strings'
        return ('dict', tuple((k, typed(value[k])) for k in sorted(value)))
    if kind is list:
        return ('list', tuple(typed(v) for v in value))
    if kind is float:
        assert math.isfinite(value), 'Nonfinite JSON forbidden; use explicit masks/NPY'
        return ('float', value.hex())
    assert kind in (str, int, bool, type(None)), f'Unsupported JSON type: {kind}'
    return (kind.__name__, value)


def canonical(value):
    original = typed(value)
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert typed(json.loads(encoded)) == original, 'Prehash JSON type/float round-trip failure'
    return encoded


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save_json(name, value):
    encoded = canonical(value)
    with (ROOT / name).open('xb') as f:
        f.write(encoded + b'\n')
        f.flush()
        os.fsync(f.fileno())
    reread = json.loads((ROOT / name).read_bytes())
    assert typed(reread) == typed(value) and canonical(reread) == encoded, 'Post-save JSON selfcheck'


def budget():
    assert time.process_time() < HARD_CPU - 2, 'Cooperative CPU stop; preserve partial attempt'
    assert PREP_CPU + FREEZE_CPU + time.process_time() < 56, 'Combined cooperative CPU cap'


def runtime_identity():
    import numpy._core._multiarray_umath as nc
    import scipy.interpolate._cubic as cubic
    import scipy.interpolate._ppoly as ppoly
    import scipy.signal._sosfilt as sos
    files = {'python': sys.executable, 'numpy_init': np.__file__, 'numpy_binary': nc.__file__,
             'scipy_init': scipy.__file__, 'natural_spline': cubic.__file__, 'ppoly': ppoly.__file__, 'sos': sos.__file__}
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'float64_dtype': np.dtype(np.float64).str, 'byteorder': sys.byteorder,
            'file_sha256': {k: sha(v) for k, v in files.items()},
            'threads': {k: os.environ[k] for k in THREAD_KEYS}, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'cpu_affinity': sorted(os.sched_getaffinity(0))}


def make_manifest(panel):
    return {'experiment': 'ONE outcome-informed numerical regression; no feasibility or promotion study',
            'panel': panel['fixtures'], 'fixtures': 12, 'rows': 48, 'arms_in_order': list(ARMS),
            'lag_rules_in_order': list(fa.RULES), 'length_only': 768, 'period_only': 180, 'phases_only': [19, 20, 21],
            'input': 'exact saved six-axis windows; no generator executed', 'known_outcome_motivation':
            'alternating/P180/phase20 strict minima [67,157,608,698] have no pair; ties247/248,337/338,427/428,517/518',
            'candidate': {'runs': 'maximal exactly equal runs on full finite 1D signal',
                          'trough': 'both exterior neighbors exist and both are strictly higher',
                          'representative': 'floor((a+b)/2), lower-middle', 'margin': '10..N-11 on representative only',
                          'epsilon': None, 'ULP_tolerance': None},
            'lag_selection': {'lags': list(range(80, 201)), 'pearson': 'np.corrcoef(y[:-lag], y[lag:])[0,1]',
                              'exact': 'lowest lag among exact finite maxima',
                              'near': 'lowest finite lag abs(r-max)<=1e-10, rtol0; unchanged'},
            'filter': panel['saved_filter'],
            'pair_rule': '80<=b-a<=200 and abs(b-a-lag)<=0.1*lag; rank(abs discrepancy, abs midpoint-(N-1)/2, a,b)',
            'features': 'raw magnitudes; inclusive natural200 CubicSpline per channel; center/population std; concatenate/unitnorm; rawRMS zero or std<=1e-12*rawRMS rejects',
            'gate': 'ALL24 strict rows ALL cached fields exact type/float.hex, correlation/feature/mask bytes BEFORE any candidate checks',
            'synthetic_helpers_only': ['constant', 'singleton', 'even_plateau', 'odd_plateau', 'separated_equal_troughs',
                                       'boundary_runs', 'left_margin_outside', 'left_margin_inside', 'right_margin_inside',
                                       'right_margin_outside', 'margin_spanning_left', 'margin_spanning_right',
                                       'equal_shoulder_not_below_both', 'nonfinite_nan', 'nonfinite_posinf', 'nonfinite_neginf'],
            'nonfinite_helper_contract': 'Reject nonfinite or non1D input with ValueError; never bridge NaN/Inf. Real panel is frozen finite.',
            'quality_cutoff': None, 'biometric_scores': False, 'finite_aliases_discarded': False,
            'cpu': {'cumulative_budget_seconds': 60, 'single_allowed_cpu': True, 'threads': 1, 'cuda': False},
            'canonical362': 'retained unchanged; no repository state modified',
            'scope_limits': ['Twelve selected saved noiseless synthetic windows only; outcome-informed selection.',
                             'Putative repeated-interval normalization, not anatomical stride annotation.',
                             'Balanced acceleration P/2 ambiguous; gyro contains P; no joint-unidentifiability claim.',
                             'IDNet arxiv1606.03238 section3 correlation templates, updates and eight channels absent.',
                             'Neither IDNet nor Derawi2010 prescribes this convention; engineering rule motivated by float failure.',
                             'Passing panel supports unit regression and scoping further work only; no real DEV authorization.',
                             'Root independent review required before publication; no publication performed.',
                             'Original eighteen correlation-nearmax checks unrelated and not repeated.'],
            'prohibited': ['old-file modification', 'repo/git', 'installs', 'SSH', 'children', 'realdata', 'fits',
                           'fullgrid', 'parameter search', 'seed search', 'new generators', 'new phases', 'new lengths',
                           'candidate changes after unexpected outcomes', 'rescued guards']}


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


def compatible_record(base, feature, fixture, ri, window, mags, smooth, rs, roots):
    """Reproduce the entire saved N768 row schema, including original row IDs."""
    family, phase, period = fixture['family'], fixture['phase_index'], 180
    fi, n = fixture['source_fixture_index'], 768
    rec = dict(base)
    rec.update(row_id=fi*4+2+ri, fixture_index=fi, fixture_key=f'{family}/P{period}/phase{phase:02d}',
               length=n, rule=fa.RULES[ri], correlation_row=fi*2+1,
               family=family, period_samples=period, phase_index=phase, phase_cycles=phase/32)
    rec.update(fa.truth(period, phase, rec, n))
    cs = fa.correlation_summary(rs)
    rec.update(all_nearmax_lags=cs['all_nearmax_lags'], correlation_gaps=cs['correlation_gaps'],
               finite_correlation_count=cs['finite_correlation_count'], undefined_correlation_count=cs['undefined_correlation_count'],
               selected_lag_correlation=None if rec['selected_lag'] is None else float(rs[rec['selected_lag']-80]),
               selected_interval_interpretation='putative repeat interval; anatomical stride not inferred',
               status='finite_pre_quality' if feature is not None else rec['failure_reason'],
               input_finite_count=int(np.isfinite(window).sum()), input_sample_value_count=6*n,
               magnitude_finite_count=int(np.isfinite(mags).sum()), filtered_acc_finite_count=int(np.isfinite(smooth).sum()),
               guard_evaluated=rec['pair_bounds_inclusive'] is not None,
               channel_population_std=rec.get('channel_population_std'), channel_raw_rms=rec.get('channel_raw_rms'),
               channel_degenerate=rec.get('channel_degenerate'), feature_norm=rec.get('feature_norm'),
               feature_channel_means=rec.get('feature_channel_means'), explicit_no_quality_cutoffs=True,
               quality_cutoff=None, quality_trained=False, quality_applied=False,
               acceleration_harmonic_ambiguity=family == fa.FAMILIES[2], full_six_axis_signal_half_periodic=False,
               joint_magnitudes_half_periodic=False, gyro_contains_physical_P=True,
               extracted_interval_alias=rec['selected_pair_alias'] not in ('unavailable', 'one_stride'),
               finite_extracted_interval_alias=rec['valid_pre_quality'] and rec['selected_pair_alias'] != 'one_stride',
               unique_nearmax_is_stride_evidence=False,
               geometric_reference=fa.geometric_reference(family, period, phase, n, rec['selected_lag'], roots))
    return rec


def counts(rows):
    return {'rows': len(rows), 'finite': sum(r['finite_features'] for r in rows),
            'failures': dict(Counter(r['failure_reason'] or 'none' for r in rows)),
            'lag_aliases': dict(Counter(r['selected_lag_alias'] for r in rows)),
            'pair_aliases': dict(Counter(r['selected_pair_alias'] for r in rows)),
            'finite_pair_aliases': dict(Counter(r['selected_pair_alias'] for r in rows if r['finite_features'])),
            'finite_half_aliases': sum(r['finite_features'] and r['selected_pair_alias'] == 'half_stride' for r in rows),
            'finite_double_aliases': sum(r['finite_features'] and r['selected_pair_alias'] == 'double_stride' for r in rows)}


def verify_saved(rows, arrays, panel, runs_by_fixture):
    """Check saved evidence and pair convention without filtering/correlation/spline extraction."""
    disk = [json.loads(line) for line in (ROOT / 'records.jsonl').read_text().splitlines()]
    assert typed(disk) == typed(rows) and len(disk) == 48
    saved = {k: np.load(ROOT / 'arrays' / (k + '.npy'), allow_pickle=False) for k in arrays}
    for k in arrays:
        assert saved[k].tobytes() == arrays[k].tobytes(), k
    assert saved['row_present_mask'].all() and saved['fixture_present_mask'].all() and saved['correlation_present_mask'].all()
    assert np.array_equal(saved['feature_finite_mask'], np.isfinite(saved['features']))
    assert np.isfinite(saved['features'][saved['feature_valid_mask']]).all()
    assert np.isnan(saved['features'][~saved['feature_valid_mask']]).all()
    assert np.max(np.abs(np.linalg.norm(saved['features'][saved['feature_valid_mask']], axis=1) - 1)) < 1e-14
    assert np.array_equal(saved['correlation_finite_mask'], np.isfinite(saved['correlations']))
    for r in disk:
        i, fi, ai, ri = r['row_id'], r['fixture_index'], r['arm_index'], r['rule_index']
        assert saved['row_mapping'][i].tolist() == [fi, ai, ri, fi, r['source_fixture_index'], r['source_row_id']]
        assert bool(saved['feature_valid_mask'][i]) == r['finite_features']
        rs = saved['correlations'][fi]
        finite = np.isfinite(rs)
        mx = float(np.max(rs[finite]))
        exact = fa.LAGS[finite & (rs == mx)].tolist()
        near = fa.LAGS[finite & (np.abs(rs-mx) <= 1e-10)].tolist()
        assert (exact, near) == (r['exact_max_ties'], r['all_nearmax_lags'])
        assert r['selected_lag'] == min(exact if ri == 0 else near)
        minima = r['minima_representatives']
        pairs, pair = ranked_pairs(minima, r['selected_lag'], 768)
        assert pairs == r['all_eligible_pair_bounds_inclusive'] and pair == r['pair_bounds_inclusive']
        assert len(pairs) == r['eligible_pair_count']
        if ai == 1:
            assert minima == [v['representative'] for v in runs_by_fixture[fi] if v['eligible_minimum']]
        if r['guard_evaluated']:
            rms, std = np.array(r['channel_raw_rms']), np.array(r['channel_population_std'])
            degenerate = (rms == 0) | (std <= 1e-12*rms)
            assert degenerate.tolist() == r['channel_degenerate']
            assert bool(degenerate.any()) == (r['failure_reason'] == 'channel_degenerate')
        assert r['quality_cutoff'] is None and not r['quality_evaluated'] and not r['quality_applied'] and not r['quality_trained']
        assert r['declared_P_error_samples']['lag'] == r['selected_lag'] - 180
        assert r['declared_P_error_samples']['pair'] == (None if pair is None else pair[1] - pair[0] - 180)
    windows = np.load(ROOT / 'input_windows.npy', allow_pickle=False)
    for fi, runs in enumerate(runs_by_fixture):
        y = saved['filtered_acc_magnitude'][fi]
        next_index = 0
        for run in runs:
            a, b = run['bounds_inclusive']
            assert a == next_index and b >= a and np.all(y[a:b+1] == y[a])
            assert a == 0 or y[a-1] != y[a]
            assert b == 767 or y[b+1] != y[b]
            assert run['value_hex'] == float(y[a]).hex() and run['representative'] == (a+b)//2
            trough = bool(a > 0 and b < 767 and y[a-1] > y[a] and y[b+1] > y[a])
            assert trough == run['is_trough'] and run['eligible_minimum'] == (trough and 10 <= (a+b)//2 <= 757)
            next_index = b + 1
        assert next_index == 768
    with np.load(ROOT / 'evidence.npz', allow_pickle=False) as z:
        assert z['input_windows'].tobytes() == windows.tobytes()
        for name in saved:
            assert z[name].tobytes() == saved[name].tobytes()
    return {'status': 'pass', 'saved_rows': 48, 'saved_windows': 12, 'saved_run_partitions': 12,
            'new_fixture_extractions': 0, 'checks': ['lossless arrays/NPZ/row map', 'all runs and representative margins',
            'lag sets and all eligible pairs', 'finite/missing/guard masks', 'normalization and aliases retained', 'no quality or scores']}


def execute(panel, frozen):
    with (ROOT / 'attempt.started.json').open('xb') as f:
        f.write(canonical({'source_sha256': frozen['source_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
                           'freeze_file_sha256': sha(ROOT / 'freeze.json'), 'unix_time_ns': time.time_ns(),
                           'exclusive': True, 'retry_authorized': False}) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    arrays, rows, runs_by_fixture = {}, [], []
    state = {'status': 'running_incomplete', 'stage': 'strict_baseline', 'baseline_rows_passed': 0, 'rows_saved': 0}
    journal = None
    def checkpoint():
        for a in arrays.values():
            a.flush()
        if journal:
            journal.flush()
            os.fsync(journal.fileno())
        state.update(rows_saved=len(rows), process_cpu_seconds=time.process_time())
        pending = ROOT / 'run_state.pending.json'
        pending.write_bytes(canonical(state) + b'\n')
        pending.replace(ROOT / 'run_state.json')
    try:
        (ROOT / 'arrays').mkdir()
        specs = {'magnitudes': ((12, 2, 768), np.float64), 'filtered_acc_magnitude': ((12, 768), np.float64),
                 'correlations': ((12, 121), np.float64), 'features': ((48, 400), np.float64),
                 'feature_valid_mask': ((48,), np.bool_), 'feature_finite_mask': ((48, 400), np.bool_),
                 'correlation_finite_mask': ((12, 121), np.bool_), 'row_present_mask': ((48,), np.bool_),
                 'fixture_present_mask': ((12,), np.bool_), 'correlation_present_mask': ((12,), np.bool_),
                 'row_mapping': ((48, 6), np.int64)}
        for name, (shape, dtype) in specs.items():
            arrays[name] = np.lib.format.open_memmap(ROOT / 'arrays' / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
            arrays[name][:] = np.nan if dtype == np.float64 else (-1 if dtype == np.int64 else False)
        windows = np.load(ROOT / 'input_windows.npy', allow_pickle=False)
        cached_rs = np.load(ROOT / 'cached_correlations.npy', allow_pickle=False)
        cached_features = np.load(ROOT / 'cached_features.npy', allow_pickle=False)
        cached_valid = np.load(ROOT / 'cached_feature_valid_mask.npy', allow_pickle=False)
        assert windows.shape == (12, 6, 768) and windows.dtype == np.float64 and np.isfinite(windows).all()
        sos = butter(2, 3.0, btype='lowpass', fs=100.0, output='sos')
        assert sos.tobytes() == np.asarray(panel['saved_filter']['sos'], dtype=np.float64).tobytes()
        journal = (ROOT / 'records.jsonl').open('x')
        reproduced = []
        def add(compat, feature, fi, ai, ri):
            rec = dict(compat)
            source_row, source_fi = rec.pop('row_id'), rec.pop('fixture_index')
            rec.pop('correlation_row')
            minima = rec.pop('strict_minima')
            rowid = fi*4+ai*2+ri
            pairs, chosen = ranked_pairs(minima, rec['selected_lag'], 768)
            assert chosen == rec['pair_bounds_inclusive'] and len(pairs) == rec['eligible_pair_count']
            rec.update(row_id=rowid, fixture_index=fi, arm_index=ai, arm=ARMS[ai], rule_index=ri,
                       source_row_id=source_row, source_fixture_index=source_fi, correlation_row=fi,
                       minima_representatives=minima, all_eligible_pair_bounds_inclusive=pairs,
                       declared_P_error_samples={'lag': rec['selected_lag'] - 180,
                           'pair': None if rec['pair_length_samples'] is None else rec['pair_length_samples'] - 180},
                       declared_P_relative_error={'lag': rec['selected_lag']/180 - 1,
                           'pair': None if rec['pair_length_samples'] is None else rec['pair_length_samples']/180 - 1})
            assert not arrays['row_present_mask'][rowid]
            arrays['row_mapping'][rowid] = [fi, ai, ri, fi, source_fi, source_row]
            if feature is not None:
                arrays['features'][rowid] = feature
                arrays['feature_finite_mask'][rowid] = np.isfinite(feature)
            arrays['feature_valid_mask'][rowid] = feature is not None
            arrays['row_present_mask'][rowid] = True
            rows.append(rec)
            journal.write(canonical(rec).decode() + '\n')
        for fi, fixture in enumerate(panel['fixtures']):
            budget()
            window = windows[fi]
            mags = np.stack([np.linalg.norm(window[:3], axis=0), np.linalg.norm(window[3:], axis=0)])
            smooth = sosfiltfilt(sos, mags[0], padtype='odd', padlen=9)
            rs = fa.correlations(smooth)
            arrays['magnitudes'][fi], arrays['filtered_acc_magnitude'][fi], arrays['correlations'][fi] = mags, smooth, rs
            arrays['correlation_finite_mask'][fi] = np.isfinite(rs)
            arrays['fixture_present_mask'][fi] = arrays['correlation_present_mask'][fi] = True
            assert rs.tobytes() == cached_rs[fi].tobytes(), f'Cached correlation bits mismatch fixture{fi}'
            for ri, rule in enumerate(fa.RULES):
                base, feature = fa.extract(mags, smooth, rs, rule)
                compat = compatible_record(base, feature, fixture, ri, window, mags, smooth, rs, panel['analytic_minimum_root_phases_cycles'])
                add(compat, feature, fi, 0, ri)
                expected = panel['records'][fi*2+ri]
                mismatch = [k for k in sorted(set(compat) | set(expected))
                            if k not in compat or k not in expected or typed(compat[k]) != typed(expected[k])]
                actual_vector = arrays['features'][fi*4+ri]
                bits_ok = actual_vector.tobytes() == cached_features[fi*2+ri].tobytes()
                mask_ok = bool(arrays['feature_valid_mask'][fi*4+ri]) is bool(cached_valid[fi*2+ri])
                if mismatch or not bits_ok or not mask_ok:
                    save_json('baseline_mismatch.json', {'fixture_index': fi, 'rule': rule, 'keys': mismatch,
                              'feature_bits_equal': bits_ok, 'mask_equal': mask_ok, 'reproduced': compat, 'cached': expected})
                    raise AssertionError('Strict baseline mismatch; STOP before candidate checks')
                reproduced.append(compat)
                state['baseline_rows_passed'] += 1
            checkpoint()
        comparison = {'status': 'pass', 'rows': 24, 'all_cached_fields_compared': True,
                      'numeric_comparison': 'exact JSON types and float.hex; correlations/features/masks bytes',
                      'cached_fields_sha256': fingerprint(panel['records']), 'reproduced_fields_sha256': fingerprint(reproduced),
                      'before_any_candidate_checks': True, 'reproduced_cached_fields': reproduced}
        assert comparison['cached_fields_sha256'] == comparison['reproduced_fields_sha256']
        save_json('baseline_comparison.json', comparison)
        state['stage'] = 'baseline_passed_before_candidate'
        checkpoint()
        print(json.dumps({'baseline_all24_passed': True, 'process_cpu_seconds': time.process_time()}), flush=True)
        budget()
        helpers = synthetic_helper_checks()
        save_json('helper_tests.json', helpers)
        candidate = types.FunctionType(fa.extract.__code__, {**fa.extract.__globals__, 'minima_and_pair': plateau_minima_and_pair}, 'plateau_extract')
        assert candidate.__code__ is fa.extract.__code__
        state['stage'] = 'fixed_candidate'
        with (ROOT / 'runs.jsonl').open('x') as run_journal:
            for fi, fixture in enumerate(panel['fixtures']):
                budget()
                mags, smooth, rs = (np.asarray(arrays[k][fi]) for k in ('magnitudes', 'filtered_acc_magnitude', 'correlations'))
                runs = exact_plateau_runs(smooth)
                runs_by_fixture.append(runs)
                singleton = [r['representative'] for r in runs if r['eligible_minimum'] and r['bounds_inclusive'][0] == r['bounds_inclusive'][1]]
                strict = fa.minima_and_pair(smooth, 180)[0]
                assert strict == singleton, 'Every strict singleton must stay identical'
                run_journal.write(canonical({'fixture_index': fi, 'fixture_key': fixture['fixture_key'],
                                   'all_runs': runs, 'strict_singletons_unchanged': True}).decode() + '\n')
                run_journal.flush()
                os.fsync(run_journal.fileno())
                for ri, rule in enumerate(fa.RULES):
                    base, feature = candidate(mags, smooth, rs, rule)
                    compat = compatible_record(base, feature, fixture, ri, windows[fi], mags, smooth, rs, panel['analytic_minimum_root_phases_cycles'])
                    add(compat, feature, fi, 1, ri)
                checkpoint()
        journal.close()
        journal = None
        assert len(rows) == 48 and arrays['row_present_mask'].all()
        mapping_columns = np.array(['fixture_index', 'arm_index', 'rule_index', 'correlation_row', 'source_fixture_index', 'source_row_id'])
        with (ROOT / 'evidence.npz').open('xb') as f:
            np.savez(f, **arrays, input_windows=windows, row_mapping_columns=mapping_columns,
                     fixture_keys=np.array([f['fixture_key'] for f in panel['fixtures']]),
                     arms=np.array(ARMS), rules=np.array(fa.RULES), lags=fa.LAGS)
        budget()
        audit = verify_saved(rows, arrays, panel, runs_by_fixture)
        summary = [{'arm': arm, 'rule': rule, **counts([r for r in rows if r['arm'] == arm and r['rule'] == rule])}
                   for arm in ARMS for rule in fa.RULES]
        by_family = [{'arm': arm, 'rule': rule, 'family': family,
                      **counts([r for r in rows if r['arm'] == arm and r['rule'] == rule and r['family'] == family])}
                     for arm in ARMS for rule in fa.RULES for family in fa.FAMILIES]
        index = {r['row_id']: r for r in rows}
        paired = []
        for fi, fixture in enumerate(panel['fixtures']):
            for ri, rule in enumerate(fa.RULES):
                a, b = index[fi*4+ri], index[fi*4+2+ri]
                paired.append({'fixture_index': fi, 'fixture_key': fixture['fixture_key'], 'rule': rule,
                               'strict_row': a['row_id'], 'plateau_row': b['row_id'],
                               'outcome': ('finite' if a['finite_features'] else a['failure_reason']) + ' -> ' +
                                          ('finite' if b['finite_features'] else b['failure_reason']),
                               'lag_unchanged': a['selected_lag'] == b['selected_lag'],
                               'pair_unchanged': a['pair_bounds_inclusive'] == b['pair_bounds_inclusive'],
                               'feature_bits_unchanged': arrays['features'][a['row_id']].tobytes() == arrays['features'][b['row_id']].tobytes(),
                               'strict_pair': a['pair_bounds_inclusive'], 'plateau_pair': b['pair_bounds_inclusive'],
                               'added_minima': sorted(set(b['minima_representatives']) - set(a['minima_representatives']))})
        old_snapshot = json.loads((ROOT / 'old_files_snapshot.json').read_text())
        after_old = {p.relative_to(OLD).as_posix(): sha(p) for p in sorted(OLD.rglob('*')) if p.is_file()}
        assert after_old == old_snapshot, 'Old evidence changed; preserve attempt'
        assert frozen['preregistration_sha256'] == {name: sha(ROOT / name) for name in PREREG}
        audit.update(old_files_unchanged=True, frozen_sources_unchanged=True, independent_review=False)
        save_json('saved_evidence_audit.json', audit)
        artifacts = {name: sha(ROOT / name) for name in ('attempt.started.json', 'records.jsonl', 'runs.jsonl', 'baseline_comparison.json',
                     'helper_tests.json', 'evidence.npz', 'saved_evidence_audit.json')}
        array_hashes = {p.name: sha(p) for p in sorted((ROOT / 'arrays').glob('*.npy'))}
        measurement_cpu = time.process_time()
        scientific = {'source_sha256': frozen['source_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
                      'freeze_file_sha256': sha(ROOT / 'freeze.json'), 'runtime_sha256': frozen['runtime_sha256'],
                      'baseline_all24_cached_fields_bitwise': True, 'helper_case_count': helpers['fixed_case_count'],
                      'total_counts': counts(rows), 'summary': summary, 'by_family': by_family, 'paired_outcomes': paired,
                      'paired_outcome_counts': dict(Counter(r['outcome'] for r in paired)),
                      'paired_counts_by_rule': [{'rule': rule, 'pairs': 12,
                          'outcomes': dict(Counter(r['outcome'] for r in paired if r['rule'] == rule))} for rule in fa.RULES],
                      'artifact_sha256': artifacts, 'array_sha256': array_hashes,
                      'scope_limits': frozen['manifest']['scope_limits'], 'canonical362_retained': True,
                      'old_files_unchanged': True, 'measurement_process_cpu_seconds': measurement_cpu}
        science_hash = fingerprint(scientific)
        report = {'status': 'complete', 'scientific': scientific, 'scientific_fingerprint_sha256': science_hash,
                  'fingerprint_contract': 'strict string keys and type/float round-trip BEFORE hash, explicit post-save recomputation',
                  'cpu': {'preparation_seconds': PREP_CPU, 'freeze_seconds': FREEZE_CPU,
                          'measurement_process_seconds': measurement_cpu,
                          'subtotal_seconds': PREP_CPU + FREEZE_CPU + measurement_cpu,
                          'run_hard_cpu_seconds': HARD_CPU, 'cumulative_cap_seconds': 60,
                          'single_cpu': sorted(os.sched_getaffinity(0))},
                  'wall_seconds_after_initial_setup': time.perf_counter() - START_WALL}
        save_json('report.json', report)
        loaded = json.loads((ROOT / 'report.json').read_text())
        assert typed(loaded) == typed(report) and fingerprint(loaded['scientific']) == loaded['scientific_fingerprint_sha256'] == science_hash
        save_json('fingerprint_selfcheck.json', {'prehash_type_roundtrip': True, 'strict_string_keys': True,
                  'postsave_exact_type_roundtrip': True, 'postsave_scientific_fingerprint_equal': True,
                  'scientific_fingerprint_sha256': science_hash, 'report_sha256': sha(ROOT / 'report.json')})
        state.update(status='complete', stage='saved_evidence_and_fingerprint_verified')
        checkpoint()
        elapsed = {'preparation_cpu_seconds': PREP_CPU, 'freeze_cpu_seconds': FREEZE_CPU,
                   'run_cpu_seconds_through_verification': time.process_time(),
                   'combined_cpu_seconds_through_verification': PREP_CPU + FREEZE_CPU + time.process_time(),
                   'run_wall_seconds_after_initial_setup': time.perf_counter() - START_WALL,
                   'CPU_budget_seconds': 60, 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'CUDA_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
        assert elapsed['combined_cpu_seconds_through_verification'] < 60
        save_json('timings.json', elapsed)
        lines = ['# Outcome-informed plateau regression', '',
                 'One exclusive attempt completed on the twelve saved P180 phase19/20/21 N768 fixtures.',
                 'All24 strict cached rows match every field, type, float bit, correlation, feature and mask before candidate checks.',
                 f'All{helpers["fixed_case_count"]} fixed helper conventions and the saved-evidence audit pass.', '']
        for item in summary:
            lines.append(f'- {item["arm"]} / {item["rule"]}: {item["finite"]}/12 finite; failures {item["failures"]}; '
                         f'finite half aliases {item["finite_half_aliases"]}, double aliases {item["finite_double_aliases"]}.')
        lines += ['', f'Paired outcomes across24 comparisons: {scientific["paired_outcome_counts"]}.',
                  f'Combined preparation/freeze/run CPU through verification: {elapsed["combined_cpu_seconds_through_verification"]:.9f}s; '
                  f'run CPU {elapsed["run_cpu_seconds_through_verification"]:.9f}s; one allowed CPU, threads1, CUDAoff, cap60s.',
                  f'Runner SHA256: `{frozen["source_sha256"]}`.', f'Manifest SHA256: `{frozen["manifest_sha256"]}`.',
                  f'Scientific fingerprint: `{science_hash}`.', '',
                  'The panel was selected after observing a floating equality failure. Results support a unit regression and scoping further work only.',
                  'This is repeated-interval normalization, not anatomical stride annotation. Finite aliases remain finite and gyro guard failures remain failures.',
                  'Balanced acceleration is half-period ambiguous; gyro contains P. No joint-unidentifiability claim is made.',
                  'Canonical362 is retained unchanged. No real DEV authorization or publication is implied. Root independent review remains required.',
                  'All old files and frozen sources verified unchanged. Lossless evidence and exact mappings are in evidence.npz, arrays/, records.jsonl and runs.jsonl.',
                  'report.json, freeze.json, baseline_comparison.json, helper_tests.json, saved_evidence_audit.json and fingerprint_selfcheck.json document the checks.']
        with (ROOT / 'RESULTS.md').open('x') as f:
            f.write('\n'.join(lines) + '\n')
        files = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
        save_json('artifact_index.json', {'files_sha256': files, 'index_self_excluded': True, 'one_attempt': True})
        print(json.dumps({'status': 'complete', 'summary': summary, 'paired_outcome_counts': scientific['paired_outcome_counts'],
                          'source_sha256': frozen['source_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
                          'scientific_fingerprint_sha256': science_hash, 'timings': elapsed}), flush=True)
    except BaseException as error:
        state.update(status='failed_preserved', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        checkpoint()
        save_json('failure.json', state)
        raise
    finally:
        if journal:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-only', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert (platform.python_version(), np.__version__, scipy.__version__) == ('3.14.8', '2.5.4', '1.18.1')
    panel = json.loads((ROOT / 'cached_panel.json').read_text())
    runtime = runtime_identity()
    manifest = make_manifest(panel)
    prereg = {name: sha(ROOT / name) for name in PREREG}
    if args.freeze_only:
        save_json('manifest.json', manifest)
        frozen = {'status': 'frozen_before_measurement', 'measurement_started': False,
                  'manifest': manifest, 'manifest_sha256': fingerprint(manifest),
                  'manifest_file_sha256': sha(ROOT / 'manifest.json'), 'source_sha256': prereg['runner.py'],
                  'preregistration_sha256': prereg, 'runtime_identity': runtime, 'runtime_sha256': fingerprint(runtime),
                  'process_cpu_seconds': time.process_time(), 'wall_seconds': time.perf_counter() - START_WALL}
        save_json('freeze.json', frozen)
        print(json.dumps({k: frozen[k] for k in ('status', 'source_sha256', 'manifest_sha256', 'runtime_sha256', 'process_cpu_seconds')}))
        return
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    assert frozen['status'] == 'frozen_before_measurement' and not frozen['measurement_started']
    assert frozen['preregistration_sha256'] == prereg and typed(frozen['manifest']) == typed(manifest)
    assert frozen['manifest_sha256'] == fingerprint(manifest) and frozen['manifest_file_sha256'] == sha(ROOT / 'manifest.json')
    assert frozen['runtime_sha256'] == fingerprint(runtime)
    execute(panel, frozen)


if __name__ == '__main__':
    main()

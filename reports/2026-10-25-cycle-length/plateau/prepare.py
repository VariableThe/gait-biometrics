#!/usr/bin/env python3
"""Read saved evidence and copy selected pure functions; never execute old source."""
import os
import resource
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_CPU, (9, 10))
START_WALL = time.perf_counter()

import ast
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'cycle-length-feasibility-r1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save_array(name, value):
    with (ROOT / name).open('xb') as f:
        np.save(f, value, allow_pickle=False)


def main():
    before = {p.relative_to(OLD).as_posix(): sha(p) for p in sorted(OLD.rglob('*')) if p.is_file()}
    source = (OLD / 'prototype.py').read_text()
    report = json.loads((OLD / 'report.json').read_text())
    assert sha(OLD / 'prototype.py') == report['source_sha256'] == 'a8dd5b7572207cdf1f80b12febfef7a6dcd1906fa766c699354390b4138b2c5c'
    assert sha(OLD / 'report.json') == 'bd890b2065009c961c7f8dc53624d85435e50ea44186d86465323673b45761da'
    for name in ('records.jsonl',):
        assert sha(OLD / name) == report['artifact_sha256'][name]
    consumed_arrays = ('windows_768', 'extractor_correlations', 'features', 'feature_valid_mask', 'fixture_present_mask', 'correlation_present_mask', 'record_present_mask')
    for name in consumed_arrays:
        assert sha(OLD / 'arrays' / (name + '.npy')) == report['array_sha256'][name + '.npy']
    names = ('correlations', 'select', 'correlation_summary', 'minima_and_pair', 'features_from_pair', 'extract', 'alias', 'truth', 'geometric_reference')
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    provenance = []
    sections = []
    for name in names:
        node = nodes[name]
        segment = ast.get_source_segment(source, node)
        sections.append(segment)
        provenance.append({'name': name, 'source_start_line': node.lineno, 'source_end_line': node.end_lineno,
                           'utf8_sha256': hashlib.sha256(segment.encode()).hexdigest(), 'copy': 'verbatim AST-delimited function text'})
    header = '''# Pure function copies only: no old module import or old top-level execution.
import math
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
FAMILIES = ("gravity_periodic_6axis", "alternating_step_harmonic", "pure_periodic_balanced_peaks", "weak_degenerate_gyro")
RULES = ("exact_max_smallest_exact_tie", "nearmax_atol_1e-10_smallest")
LAGS = np.arange(80, 201)
GUARD = 1e-12
ATOL = 1e-10
'''
    with (ROOT / 'frozen_algorithm.py').open('x') as f:
        f.write(header + '\n\n' + '\n\n\n'.join(sections) + '\n')
    save_json('provenance.json', {'source': str(OLD / 'prototype.py'), 'source_sha256': sha(OLD / 'prototype.py'),
              'source_top_level_executed': False, 'function_copies': provenance,
              'candidate_only_change': 'runner.plateau_extract clones extract globals and replaces minima_and_pair only; code object unchanged'})
    families = report['manifest']['families_in_order']
    expected = [(family, 180, phase) for family in families for phase in (19, 20, 21)]
    allrows = [json.loads(line) for line in (OLD / 'records.jsonl').read_text().splitlines()]
    selected = {(r['family'], r['period_samples'], r['phase_index'], r['rule']): r for r in allrows
                if r['length'] == 768 and r['period_samples'] == 180 and r['phase_index'] in (19, 20, 21)}
    rules = report['manifest']['rules_in_order']
    rows = [selected[(*key, rule)] for key in expected for rule in rules]
    assert len(rows) == len(selected) == 24
    fixtures = [{'fixture_index': i, 'source_fixture_index': rows[2*i]['fixture_index'],
                 'fixture_key': rows[2*i]['fixture_key'], 'family': family, 'period_samples': p, 'phase_index': phase}
                for i, (family, p, phase) in enumerate(expected)]
    cache = {name: np.load(OLD / 'arrays' / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in consumed_arrays}
    fis = [f['source_fixture_index'] for f in fixtures]
    crs = [rows[2*i]['correlation_row'] for i in range(12)]
    ris = [r['row_id'] for r in rows]
    assert cache['fixture_present_mask'][fis].all() and cache['correlation_present_mask'][crs].all() and cache['record_present_mask'][ris].all()
    for filename, value in (
        ('input_windows.npy', cache['windows_768'][fis]),
        ('cached_correlations.npy', cache['extractor_correlations'][crs]),
        ('cached_features.npy', cache['features'][ris]),
        ('cached_feature_valid_mask.npy', cache['feature_valid_mask'][ris]),
    ):
        save_array(filename, value)
    save_json('cached_panel.json', {'fixtures': fixtures, 'rules': rules, 'records': rows,
              'analytic_minimum_root_phases_cycles': report['analytic_minimum_root_phases_cycles'],
              'saved_filter': report['manifest']['filter'], 'scope': 'Only saved P180 phases19/20/21 N768; no generators or new roots executed'})
    after = {p.relative_to(OLD).as_posix(): sha(p) for p in sorted(OLD.rglob('*')) if p.is_file()}
    assert before == after
    save_json('old_files_snapshot.json', before)
    result = {'status': 'prepared_without_fixture_measurement', 'saved_windows': 12, 'cached_rows': 24,
              'old_files_unchanged': True, 'old_file_count': len(before), 'process_cpu_seconds': time.process_time(),
              'wall_seconds': time.perf_counter() - START_WALL, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'hard_cpu_seconds': 10}
    save_json('preparation.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

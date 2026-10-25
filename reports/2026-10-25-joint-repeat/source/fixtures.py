"""Fixed input inventory and synthetic-only generators. No extraction or fitting."""

import common as c
import legacy
import numpy as np


def keys():
    answer = []
    for fi, (family, period, phase) in enumerate(
        (f, p, k) for f in legacy.FAMILIES for p in legacy.PERIODS for k in range(32)
    ):
        for li, n in enumerate(c.LENGTHS):
            answer.append(
                {
                    "kind": "legacy",
                    "family": family,
                    "period_samples": period,
                    "phase_index": phase,
                    "length": n,
                    "source_fixture_index": fi,
                    "source_length_index": li,
                    "condition": "original",
                    "transformation": "identity",
                    "key": f"legacy/{family}/P{period}/phase{phase:02d}/N{n}",
                }
            )
    for period in c.STRESS_PERIODS:
        for phase in c.PHASES:
            for n in c.LENGTHS:
                for condition in c.CONDITIONS:
                    answer.append(
                        {
                            "kind": "stress",
                            "family": "analytic_positive",
                            "period_samples": period,
                            "phase_index": phase,
                            "length": n,
                            "condition": condition,
                            "transformation": "identity",
                            "key": f"stress/P{period}/phase{phase:02d}/N{n}/{condition}",
                        }
                    )
    for family in legacy.FAMILIES:
        for phase in c.PHASES:
            for n in c.LENGTHS:
                for transform in c.TRANSFORMS:
                    answer.append(
                        {
                            "kind": "invariance",
                            "family": family,
                            "period_samples": 160,
                            "phase_index": phase,
                            "length": n,
                            "condition": "original",
                            "transformation": transform,
                            "key": f"invariance/{family}/P160/phase{phase:02d}/N{n}/{transform}",
                        }
                    )
    for index, item in enumerate(answer):
        item["input_index"] = index
    c.require(
        len(answer) == 2880 and len({r["key"] for r in answer}) == 2880,
        "Input inventory mismatch",
    )
    return answer


def stress_long(period, phase):
    j = c.STRESS_PERIODS.index(period)
    rng = np.random.Generator(np.random.PCG64(20261025 + 1000 * j + phase))
    # One row-major 768x2 sequence per P/actual phase, shared across all eight conditions.
    noise = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(768, 2))
    theta = 2 * np.pi * (np.arange(768, dtype=float) / period + phase / 32)
    sigma_a = 1 / np.sqrt(2.0)
    sigma_g = 1 / (2 * np.sqrt(2.0))
    a, g = 10 + np.cos(2 * theta), 4 + 0.5 * np.cos(theta)
    na, ng = sigma_a * noise[:, 0], sigma_g * noise[:, 1]
    vectors = (
        (a, g),
        (a + 0.1 * na, g + 0.1 * ng),
        (a, g + 4 * ng),
        (a, 4 + 4 * ng),
        (10 + 4 * na, 4 + 4 * ng),
        (a, np.full(768, 4.0)),
        (a, 4 + 1e-13 * np.cos(theta)),
        (a, 4 + 0.5 * np.cos(2 * theta)),
    )
    output = {}
    for condition, (acc, gyro) in zip(c.CONDITIONS, vectors):
        c.require(
            bool(np.all(acc > 0) and np.all(gyro > 0)),
            "Positive analytic lift violated",
        )
        zero = np.zeros(768)
        output[condition] = np.asarray([zero, zero, acc, zero, zero, gyro])
    return output


def rotation(name):
    if name == "common_Rz30":
        angle = np.pi / 6
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    if name == "common_Rx90":
        # Exact proper quarter-turn avoids introducing a nonzero cos(pi/2) coefficient.
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    raise ValueError(name)


def generate(item):
    n, p, k = item["length"], item["period_samples"], item["phase_index"]
    if item["kind"] == "stress":
        return stress_long(p, k)[item["condition"]][:, :n].copy()
    base = legacy.fixture(item["family"], p, k, 768)
    if item["kind"] == "legacy":
        prefix = base[:, :n].copy()
        independent = legacy.fixture(item["family"], p, k, n)
        c.require(
            prefix.tobytes() == independent.tobytes(),
            "Legacy independent-prefix mismatch",
        )
        return prefix
    transform = item["transformation"]
    if transform.startswith("common_R"):
        r = rotation(transform)
        transformed = np.vstack((r @ base[:3], r @ base[3:]))
    else:
        gains = (0.9, 1.1) if transform == c.TRANSFORMS[2] else (1.1, 0.9)
        transformed = np.vstack((gains[0] * base[:3], gains[1] * base[3:]))
    return transformed[:, :n].copy()


def manifest():
    return {
        "study": "joint-repeat-feasibility-r1",
        "preparation_only_until_root_freeze": True,
        "counts": {
            "inputs": 2880,
            "legacy_inputs": 2560,
            "stress_inputs": 192,
            "invariance_inputs": 128,
            "extractions": 8640,
            "baseline_gate_rows": 5120,
        },
        "arms": list(c.ARMS),
        "lengths": list(c.LENGTHS),
        "families": list(legacy.FAMILIES),
        "periods": list(legacy.PERIODS),
        "phases": list(range(32)),
        "dtype": "float64",
        "ordering": "legacy: family,period,phase,length; stress: period,phase,length,condition; invariance: family,phase,length,transform; each row=input_index*3+arm_index",
        "legacy_generator": "Verbatim fixture in legacy.py; generate768 then prefix; independently generate same N and compare exact bytes",
        "stress": {
            "periods": list(c.STRESS_PERIODS),
            "phases": list(c.PHASES),
            "conditions": list(c.CONDITIONS),
            "theta": "2*pi*(arange(768)/P+phase/32)",
            "a": "10+cos(2theta)",
            "g": "4+0.5*cos(theta)",
            "sigmas": ["1/sqrt(2)", "1/(2sqrt(2))"],
            "noise": "Generator(PCG64(20261025+1000*j+actual_phase_index)).uniform(-sqrt(3),sqrt(3),size=(768,2)), C-order; acc column0 gyro column1; same draw for eight conditions; NO sample recentering/rescaling",
            "replacement": "DC-preserving a=10+4*sigma_a*u_a and/or g=4+4*sigma_g*u_g; amplitudes 4 frozen before any test",
            "prefix": "N256 exact prefix of N768 before filtering; lengths filtered independently",
            "lift": "(0,0,a),(0,0,g); analytic DC lower bounds positive",
        },
        "invariance": {
            "period": 160,
            "phases": list(c.PHASES),
            "transforms": list(c.TRANSFORMS),
            "rotations": "Rz30 cos(pi/6)/sin(pi/6); Rx90 exact signed permutation; common proper rotations of both modalities; no untransformed extra inputs",
        },
        "filter": {
            "order": 2,
            "cutoff_hz": 3.0,
            "fs_hz": 100.0,
            "padtype": "odd",
            "padlen": 9,
            "sos": c.SOS.tolist(),
        },
        "candidate": {
            "lags": [80, 200],
            "vectors": "filtered prefix[:-lag],suffix[lag:] of equal length N-lag",
            "pearson": "np.corrcoef(x,y)[0,1] after independent vector guards; standard numpy clipping",
            "guard": "all finite AND uncentered RMS q>0 AND population std>1e-12*q for EACH vector in each modality",
            "joint": "(guarded_acc_pearson+guarded_gyro_pearson)/2; undefined if either modality undefined",
            "selection": "smallest lag with abs(J-maxJ)<=1e-10, rtol=0; retain exact ties, nearsets, gaps",
            "whole_window_guard": "record raw and filtered modality guards regardless of first extraction failure",
        },
        "main_boundaries": "STRICT old acc minima only; margin10; 80<=duration<=200; abs(duration-lag)<=0.1*lag; rank(abs(duration-lag),abs(midpoint-(N-1)/2),a,b)",
        "features": "Old unchanged raw magnitude inclusive endpoints natural cubic 200 each; center population-standardize; relative1e-12 RMS guard; 400 concat unit L2; invalid NaNs plus bool mask",
        "quality_cutoff": None,
        "fallback_imputation_reweighting": False,
        "diagnostics": {
            "lags": [40, 200],
            "anchors": "arange(N-200): 0..N-201 inclusive, same anchors at every lag",
            "channels": [
                "raw_acc",
                "raw_gyro",
                "filtered_acc",
                "filtered_gyro",
                "raw_joint",
                "filtered_joint",
            ],
            "guards": "same vector guard as candidate, all per-vector RMS/std/masks saved",
            "effect_on_extraction": "NONE",
        },
        "gates": "Exclusive attempt consumed FIRST; validate freeze/source/runtime/parents; reproduce all2560 original inputs x both controls with all cached fields, exact type/float.hex, original arrays/masks/feature/correlation bits; expected counts EACH length/rule BEFORE any joint correlation/extraction/diagnostic/helper candidate execution",
        "expected_control_counts": {
            "256": {
                "finite": 848,
                "missing_pair": 192,
                "channel_degenerate": 240,
                "finite_half_alias": 96,
            },
            "768": {
                "finite": 959,
                "missing_pair": 1,
                "channel_degenerate": 320,
                "finite_half_alias": 96,
            },
        },
        "fixed_checks": {
            "legacy_lag_vectors": 18,
            "plateau_helpers": 16,
            "new": "Pearson/guard identical,negative,positive-affine,zero,constant,DC5e-13/DC2e-12,NaN,+inf,-inf; analytic unfiltered P180phase0 both lengths; centered-dot identity distance squared=2(1-J)",
        },
        "reporting": "Each length separately; all inputs/denominators; 96 balanced P160/180/200 x32 settings per length primary; selected-lag abs error<=1 vs declared P and finite-oneP extraction separate; transitions for every original finite half output; 1280 legacy finite-oneP coverage/lag/pair errors; aliases; undefined corr; failures; modality corr; noise-only false valid; invariance exact lag/pair/validity and feature errors, changes logged never repaired",
        "invariance_feature_log_tolerance": {
            "atol": 1e-10,
            "rtol": 0.0,
            "purpose": "descriptive numerical comparison only; not a selection/quality threshold; exact equality and max absolute error also retained",
        },
        "budget": {
            "exclusive_attempts": 1,
            "cpu_seconds": 600,
            "wall_seconds": 900,
            "cooperative_cpu_seconds": 595,
            "cooperative_wall_seconds": 895,
            "cpu_rlimit_soft": 599,
            "cpu_rlimit_hard": 600,
            "numerical_threads": 1,
            "cpu_affinity_count": 1,
            "cuda_off": True,
            "includes": "CPU: execute Python process including interpreter/imports/validation/baseline/candidate/verification/packaging; uv CPU outside Python process. Wall: shell reservation immediately before uv through final output. Preparation separately recorded.",
        },
        "scope": "Synthetic feasibility ONLY; N768 DIAGNOSTIC ONLY (inventory one768 block,zero both-days people); canonical362 WCCN/rawmean_multi/policyC retained; no DEV/data/model/scores/fitting/quality cutoff/publication/installs/children/git",
        "literature": {
            "IDNet": "https://arxiv.org/html/1606.03238#S3",
            "Barth2015": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4435165/",
            "MatrixProfileVI": "https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf",
            "claim": "JOINT PEARSON is our adaptation, reproducing none of those complete pipelines",
        },
        "unresolved_brief_ambiguities": [],
        "implementation_conventions_for_root_review": [
            "Canonical legacy half/one/double/other labels retained as saved metadata only, never anatomical claims.",
            "Primary selected-lag resolution tolerance exactly<=1sample; finite oneP coverage uses legacy +/-10% pair-alias label plus valid feature, with all exact errors retained.",
            "Transitions give failure priority missing_pair, then degenerate (including whole-window joint guard), then otherfailure; valid aliases classified oneP/otheralias.",
            "Diagnostics include both raw and filtered modalities (and their means), never used to select or repair.",
            "No additional untransformed invariance input: compare to same legacy family/P160/phase/length/arm.",
        ],
    }

"""Bounded source/synthetic diagnosis. No data/artifact loads, fitting, or candidates.

Run from gait-biometrics with the invocation recorded in summary.json. The only
output is summary.json next to this script. All fixtures are fixed below, before
any measurements; no parameter/seed search or outcome-driven fixture selection.
"""

import os
import sys

sys.dont_write_bytecode = True
for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["UV_OFFLINE"] = "1"

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path("/home/sherlock/Code/github/VariableThe/gait-biometrics")
OUT = Path(__file__).resolve().parent
assert OUT == ROOT.parent / "artifacts/noise-gain-diagnostic-r1"
sys.path.insert(0, str(ROOT))

from data.imu_perturbations import IMUPerturbation, apply_imu_perturbation
from models.gait_canonical import extract_canonical_features
from models.gait_features import extract_gait_features
from models.gait_invariants import extract_invariant_features, invariant_feature_names
from models.lag_covariance import extract_lag_covariance_features
from models.session_robust import extract_lagged_covariance_features

FROZEN = "108a559bc1a83355dc85335e9f640786846f894a"
HEAD = "1c861bfc2104b204c7262d931d8cffa19cc6e676"
FILES = (
    "models/gait_canonical.py", "models/gait_features.py",
    "models/gait_invariants.py", "models/session_robust.py",
    "models/gait_metric.py", "models/lag_covariance.py",
    "scripts/canonical_orientation_study.py",
    "scripts/mentor_backend_study.py", "scripts/covariance_robustness_study.py",
    "scripts/multisensor_study.py", "data/imu_perturbations.py",
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def fixtures():
    # Four complete cycles in 2.56 seconds: fundamental 1.5625 Hz, harmonics
    # through 12.5 Hz. Acceleration m/s^2; angular velocity degrees/second.
    # Full-rank harmonic mixture, followed by an idealized planar single-mode
    # motion. Neither fixture is a model of an individual or a biometric trial.
    p = 2 * np.pi * 4 * np.arange(256) / 256
    full = np.array([
        1.4*np.sin(p) + .28*np.sin(2*p+.3) + .12*np.cos(4*p) + .08*np.sin(7*p),
        .55*np.cos(p+.2) + .18*np.cos(3*p) + .07*np.sin(5*p) + .05*np.cos(8*p),
        9.81 + .7*np.cos(2*p+.4) + .2*np.sin(3*p) + .09*np.cos(6*p),
        18*np.cos(p+.1) + 4*np.sin(3*p) + 1.7*np.cos(5*p) + 1.2*np.sin(7*p),
        11*np.sin(p+.4) + 3*np.cos(2*p) + 1.5*np.sin(6*p) + .8*np.cos(8*p),
        7*np.cos(p+.7) + 2*np.sin(4*p) + 1.1*np.cos(6*p) + .6*np.sin(7*p),
    ])
    deficient = np.array([
        1.4*np.sin(p), np.zeros(256), 9.81+.7*np.cos(p),
        18*np.cos(p), np.zeros(256), 7*np.sin(p),
    ])
    return np.stack((full, deficient))


def layout():
    blocks = {
        "axis_amplitude": [], "axis_lag_correlation": [], "axis_spectrum": [],
        "invariant_amplitude": [], "invariant_covariance_fractions": [],
        "invariant_lag_cosine": [], "invariant_centered_lag_correlation": [],
        "invariant_spectra": [], "lag_log_eigenvalues": [], "lag_log_trace": [],
    }
    names = []
    stats = ("mean", "std", "rms", "min", "max", "q10", "q25", "q50", "q75", "q90", "mean_abs_difference")
    spectral = ("band0_0.5", "band0.5_1", "band1_2", "band2_3", "band3_5", "band5_inf", "entropy", "dominant_hz", "centroid_hz")
    for c, name in enumerate(("acc_fwd", "acc_lateral", "acc_gravity", "gyro_fwd", "gyro_lateral", "gyro_gravity", "acc_norm", "gyro_norm")):
        names.extend(f"{name}.{s}" for s in (*stats, *(f"corr_{lag}" for lag in (10,20,30,50,75,100)), *spectral))
        for block, lo, hi in (("axis_amplitude",0,11), ("axis_lag_correlation",11,17), ("axis_spectrum",17,26)):
            blocks[block].extend(range(26*c+lo,26*c+hi))
    names.extend(invariant_feature_names(6))
    for g in range(2):
        for block, lo, hi in (("invariant_amplitude",0,31), ("invariant_covariance_fractions",31,34), ("invariant_lag_cosine",34,40), ("invariant_centered_lag_correlation",40,46), ("invariant_spectra",46,64)):
            blocks[block].extend(range(208+64*g+lo,208+64*g+hi))
    for g in range(2):
        names.extend([*(f"lag_triad{g}.log_eigenvalue{k}" for k in range(12)), f"lag_triad{g}.log_trace"])
        blocks["lag_log_eigenvalues"].extend(range(336+13*g,348+13*g))
        blocks["lag_log_trace"].append(348+13*g)
    assert len(names) == 362
    assert sorted(i for indices in blocks.values() for i in indices) == list(range(362))
    return blocks, names


BLOCKS, NAMES = layout()


def feature362(x):
    canonical = extract_canonical_features(x, 100)
    # Exact registered composition, avoiding study runners and all data loaders.
    return np.concatenate((canonical.require_all_valid().astype(float), extract_lagged_covariance_features(x).astype(float)), axis=1), canonical


def fusion_in_fixed_frames(x, frames):
    # Diagnostic attribution only; this is not a proposed candidate or score.
    projected = np.einsum("nsji,ngjt->nsgit", frames, x.reshape(len(x),2,3,256)).reshape(-1,6,256)
    axes = extract_gait_features(projected,100)
    channels = np.concatenate((projected,np.linalg.norm(projected.reshape(-1,2,3,256),axis=2)),axis=1)
    centered = channels-channels.mean(axis=-1,keepdims=True)
    power = np.abs(np.fft.rfft(centered*np.hanning(256),axis=-1))**2
    power[...,0] = 0
    peak = power.max(axis=-1,keepdims=True)
    relative_peak = np.divide(power,peak,out=np.zeros_like(power),where=peak>0)
    hz = np.fft.rfftfreq(256,d=.01)
    dominant = hz[np.argmax(relative_peak >= 1-64*np.finfo(float).eps,axis=-1)]
    axes[:,26*np.arange(8)+24] = np.where(power.sum(axis=-1)>1e-24,dominant,0)
    inv = extract_invariant_features(projected,100)
    fusion = np.concatenate((axes,inv),axis=1).reshape(len(x),2,336).astype(float).mean(axis=1).astype(np.float32)
    return fusion.astype(float)


def lag_reference(x):
    results, features = [], []
    for triad in x.reshape(2,3,256):
        stacked = np.concatenate([triad[:,lag:lag+236] for lag in (0,5,10,20)])
        z = stacked-stacked.mean(axis=1,keepdims=True)
        # Independent direct covariance and SVD checks at ordinary magnitudes.
        cov = np.cov(stacked,ddof=1)
        np.testing.assert_allclose(cov,z@z.T/235,atol=2e-12,rtol=2e-13)
        trace = float(np.trace(cov))
        eig = np.linalg.eigvalsh(cov/max(trace,1e-8))
        svd_eig = np.linalg.svd(z,compute_uv=False)[::-1]**2/235/max(trace,1e-8)
        np.testing.assert_allclose(eig,svd_eig,atol=4e-16,rtol=2e-6)
        feat = np.r_[np.log(np.maximum(eig,1e-8)),np.log(max(trace,1e-8))]
        features.extend(feat)
        results.append({
            "rank": int(np.linalg.matrix_rank(z)), "trace": trace,
            "trace_floor_active": bool(trace<=1e-8),
            "normalized_eigenvalues": eig.tolist(),
            "eigenvalues_at_or_below_1e_8": int(np.count_nonzero(eig<=1e-8)),
            "svd_cov_eigen_max_abs_error": float(np.max(np.abs(eig-svd_eig))),
            "regularized_matrix_min_eigenvalue": float(.8*max(eig[0],0)+.2/12),
        })
    return np.array(features), results


def spectrum_diagnostics(x, frames):
    b = frames[0]
    projected = np.vstack((b.T@x[:3],b.T@x[3:]))
    channels = np.vstack((projected,np.linalg.norm(projected.reshape(2,3,256),axis=1)))
    centered = channels-channels.mean(axis=-1,keepdims=True)
    p = np.abs(np.fft.rfft(centered*np.hanning(256),axis=-1))**2
    p[:,0] = 0
    total = p.sum(axis=-1)
    peaks = np.sort(p,axis=-1)[:,-2:]
    gaps = np.divide(peaks[:,1]-peaks[:,0],peaks[:,1],out=np.zeros(8),where=peaks[:,1]>0)
    return {"axis_norm_total_power":total.tolist(), "axis_norm_below_1e_24":int(np.count_nonzero(total<=1e-24)), "relative_top_two_power_gap":gaps.tolist()}


def geometry_record(g, row, clean):
    # Pair-match before comparing, since the forward/lateral signs have no order.
    distance = min(float(np.linalg.norm(g.frames[row,order]-clean.frames[row])) for order in ([0,1],[1,0]))
    return {"valid":bool(g.valid[row]), "reason":str(g.reason[row]),
            "gravity_ratio":float(g.gravity_ratio[row]),
            "horizontal_energy_ratio":float(g.horizontal_energy_ratio[row]),
            "eigengap_ratio":float(g.eigengap_ratio[row]),
            "frame_pair_frobenius_distance_to_clean":distance}


def delta_record(before, after):
    delta = after-before
    records=[]
    for block, columns in BLOCKS.items():
        d=delta[columns]
        records.append({"block":block,"n":len(columns),"l2":float(np.linalg.norm(d)),"rms_per_coordinate":float(np.sqrt(np.mean(d*d))),"max_abs":float(np.max(np.abs(d)))})
    records.sort(key=lambda r:r["l2"],reverse=True)
    top=np.argsort(-np.abs(delta),kind="stable")[:5]
    return {"block_ranking_by_raw_l2":records,"top_coordinates":[{"column":int(i),"name":NAMES[i],"clean":float(before[i]),"perturbed":float(after[i]),"delta":float(delta[i])} for i in top]}


def lag_matrix_reference(x, logarithm):
    out=[]
    for triad in x.reshape(2,3,256):
        z=np.concatenate([triad[:,l:l+236] for l in (0,5,10,20)])
        cov=np.cov(z,ddof=1)
        trace=float(np.trace(cov))
        s=.8*cov/max(trace,1e-8)+.2*np.eye(12)/12
        if logarithm:
            lam,q=np.linalg.eigh(s)
            s=(q*np.log(lam))@q.T
        a,b=np.triu_indices(12)
        vec=s[a,b].copy()
        vec[a!=b]*=np.sqrt(2)
        out.extend(np.r_[vec,np.log(max(trace,1e-8))])
    return np.array(out)


def main():
    assert git("rev-parse","HEAD").decode().strip()==HEAD
    status_before=git("status","--porcelain=v1")
    hashes={}
    for file in FILES:
        frozen=git("show",f"{FROZEN}:{file}")
        assert frozen==git("show",f"{HEAD}:{file}")== (ROOT/file).read_bytes(),file
        hashes[file]=hashlib.sha256(frozen).hexdigest()
    x=fixtures()
    x.setflags(write=False)
    clean,c=feature362(x)
    # Fixed reference from the predeclared full-rank clean synthetic window.
    # This scalar calculation is not estimator fitting or real TRAIN access.
    dynamic=x[0]-x[0].mean(axis=-1,keepdims=True)
    scale=np.sqrt(np.mean(dynamic.reshape(2,3,256)**2,axis=(1,2)))
    # Analytic independent reference: each distinct full-cycle sinusoid has
    # variance amplitude^2/2; triad RMS includes the 1/3 axis average.
    coefficients=([
        1.4,.28,.12,.08,.55,.18,.07,.05,.7,.2,.09,
    ],[18,4,1.7,1.2,11,3,1.5,.8,7,2,1.1,.6])
    analytic=np.array([np.sqrt(np.sum(np.array(a)**2)/6) for a in coefficients])
    np.testing.assert_allclose(scale,analytic,atol=3e-15,rtol=1e-14)
    base_fixed=fusion_in_fixed_frames(x,c.geometry.frames)
    np.testing.assert_array_equal(base_fixed,clean[:,:336])
    amp=BLOCKS["axis_amplitude"]+BLOCKS["invariant_amplitude"]
    energy=BLOCKS["lag_log_trace"]
    shape=[i for i in range(362) if i not in amp+energy]
    conditions={"clean":x,"gain09":.9*x,"gain11":1.1*x}
    for fraction in (.01,.05):
        name="noise01" if fraction==.01 else "noise05"
        spec=IMUPerturbation(name=name,noise_fraction=(fraction,fraction),reference_std=tuple(np.repeat(scale,3)),reference_id="fixed-synthetic-full-rank-dynamic-RMS",units=("m/s^2","degrees/sec"))
        conditions[name]=apply_imu_perturbation(x,spec,window_keys=("synthetic/full_rank","synthetic/rank_deficient"),seed=20261025).windows
    rows=[]
    matrix_comparison={}
    independent_max=0.
    for condition,signal in conditions.items():
        f,can=feature362(signal)
        fixed=fusion_in_fixed_frames(signal,c.geometry.frames)
        for row,label in enumerate(("full_rank","rank_deficient")):
            reference,eig=lag_reference(signal[row])
            error=float(np.max(np.abs(reference-f[row,336:])))
            independent_max=max(independent_max,error)
            np.testing.assert_allclose(reference,f[row,336:],atol=2e-6,rtol=2e-7)
            item={"row_key":f"synthetic/{label}","condition":condition,
                  "geometry":geometry_record(can.geometry,row,c.geometry),
                  "lag_triads":eig,
                  "spectrum":spectrum_diagnostics(signal[row],can.geometry.frames[row]),
                  "independent_lag_max_abs_error":error,
                  "canonical336_total_delta_l2":float(np.linalg.norm(f[row,:336]-clean[row,:336])),
                  "clean_frame_held336_delta_l2":float(np.linalg.norm(fixed[row]-clean[row,:336])),
                  "reestimated_vs_held_frame336_l2":float(np.linalg.norm(f[row,:336]-fixed[row])),
                  **delta_record(clean[row],f[row])}
            if condition.startswith("gain"):
                g=.9 if condition=="gain09" else 1.1
                expected=clean[row].copy()
                expected[amp]*=g
                expected[energy]+=2*np.log(g)
                item["gain_identity_check"]={
                    "gain":g,"expected_log_trace_shift":float(2*np.log(g)),
                    "max_abs_amplitude_error":float(np.max(np.abs(f[row,amp]-g*clean[row,amp]))),
                    "max_abs_shape_delta":float(np.max(np.abs(f[row,shape]-clean[row,shape]))),
                    "actual_log_trace_shifts":(f[row,energy]-clean[row,energy]).tolist(),
                    "all362_max_abs_prediction_error":float(np.max(np.abs(f[row]-expected))),
                }
                np.testing.assert_allclose(f[row],expected,atol=3e-6,rtol=3e-7)
            if condition.startswith("noise"):
                frac=.01 if condition=="noise01" else .05
                item["noise_std_native_per_triad"]=(frac*scale).tolist()
                noise=(signal[row]-x[row]).reshape(2,3,256)
                item["realized_noise_rms_native_per_triad"]=np.sqrt(np.mean(noise**2,axis=(1,2))).tolist()
                # For iid time/axis noise the uncentered increment is sigma^2
                # I12. Finite overlapping lag means add a small off-diagonal
                # correction; the simple eigenvalue formula neglects O(1/L).
                # A single draw also contains signal/noise cross terms.
                _,clean_eig=lag_reference(x[row])
                approximation=[]
                for triad in range(2):
                    trace=clean_eig[triad]["trace"]
                    eta=(frac*scale[triad])**2
                    old=np.array(clean_eig[triad]["normalized_eigenvalues"])
                    estimated=(old*trace+eta)/(trace+12*eta)
                    approximation.append({"sigma_squared":float(eta),"predicted_normalized_min_eigenvalue":float(estimated[0]),"predicted_max_log_eigen_delta":float(np.max(np.abs(np.log(np.maximum(estimated,1e-8))-np.log(np.maximum(old,1e-8)))))})
                item["noise_expected_covariance_approximation"]=approximation
            rows.append(item)
        for logarithm in (False,True):
            got=extract_lag_covariance_features(signal,logarithm=logarithm)
            refs=np.stack([lag_matrix_reference(s,logarithm) for s in signal])
            np.testing.assert_allclose(got,refs,atol=2e-12,rtol=1e-11)
            if condition=="clean":
                matrix_comparison[str(logarithm)]={"shape":list(got.shape),"reference_max_abs_error":float(np.max(np.abs(got-refs))),"clean":got}
            else:
                base=matrix_comparison[str(logarithm)]["clean"]
                delta=got-base
                matrix_comparison[str(logarithm)][condition]={"matrix_only_l2_per_row":np.linalg.norm(np.delete(delta,[78,157],axis=1),axis=1).tolist(),"log_trace_shifts":delta[:,[78,157]].tolist()}
    assert [r["rank"] for r in rows[0]["lag_triads"]]==[12,12]
    assert [r["rank"] for r in rows[1]["lag_triads"]]==[2,2]
    assert all(row["geometry"]["valid"] for row in rows)
    for entry in matrix_comparison.values():
        del entry["clean"]
    # Minimal exact illustrations, deliberately separate from all real WCCN.
    epsilon=.01
    v=np.array([1.,0.]); w=np.array([1.,epsilon]); projection=np.diag([1.,100.])
    cosine=lambda a,b:float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    metric_illustration={"label":"illustrative fixed manual 2D projection ONLY; no WCCN fit or saved metric loaded", "clean":v.tolist(),"perturbed":w.tolist(),"projection_diagonal":[1,100],"cosine_identity":cosine(v,w),"cosine_fixed_projection":cosine(v@projection,w@projection),"cosine_isotropic_times100":cosine(100*v,100*w)}
    np.testing.assert_allclose(metric_illustration["cosine_fixed_projection"],1/np.sqrt(2),atol=2e-16)
    tiny=np.array([1.,.5]); gained=np.array([1.1,.5])
    metric_illustration["amplitude_plus_shape_gain_cosine"]=cosine(tiny,gained)
    noise_findings=[]
    for row in rows:
        if row["condition"].startswith("noise"):
            clean_row=next(r for r in rows if r["row_key"]==row["row_key"] and r["condition"]=="clean")
            noise_findings.append({
                "row_key":row["row_key"],"condition":row["condition"],
                "clean_lag_eigen_coordinates_at_floor":sum(t["eigenvalues_at_or_below_1e_8"] for t in clean_row["lag_triads"]),
                "noisy_lag_eigen_coordinates_at_floor":sum(t["eigenvalues_at_or_below_1e_8"] for t in row["lag_triads"]),
                "clean_zero_power_axis_norm_channels":clean_row["spectrum"]["axis_norm_below_1e_24"],
                "noisy_zero_power_axis_norm_channels":row["spectrum"]["axis_norm_below_1e_24"],
                "raw_top_three_blocks":[r["block"] for r in row["block_ranking_by_raw_l2"][:3]],
                "max_lag_log_eigen_delta":next(r["max_abs"] for r in row["block_ranking_by_raw_l2"] if r["block"]=="lag_log_eigenvalues"),
            })
    result={
        "scope":"One bounded source and deterministic synthetic diagnosis; not a candidate, fit, biometric evaluation, or causal explanation of observed DEV errors.",
        "frozen_source_commit":FROZEN,"evidence_head":HEAD,"source_sha256":hashes,
        "runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"threads":1,"cuda_visible_devices":"","offline":True,"seed":20261025},
        "invocation":"cd /home/sherlock/Code/github/VariableThe/gait-biometrics && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 uv run --no-project --python ../.venv/bin/python python -B ../artifacts/noise-gain-diagnostic-r1/diagnose.py",
        "fixture_spec":{"samples":256,"sampling_rate_hz":100,"fundamental_hz":1.5625,"native_units":["m/s^2","degrees/sec"],"definition":"fixed trigonometric expressions in fixtures(); never changed based on output","reference":"clean full-rank synthetic window, fixed across rows and noise levels; not actual TRAIN","dynamic_rms_native_per_triad":scale.tolist(),"analytic_dynamic_rms_native_per_triad":analytic.tolist()},
        "coverage":{"input_rows_per_condition":2,"conditions":list(conditions),"expected_rows":10,"returned_rows":len(rows),"valid_rows":sum(row["geometry"]["valid"] for row in rows),"dropped_or_imputed":0,"columns":362,"disjoint_feature_block_columns":{k:len(v) for k,v in BLOCKS.items()}},
        "raw_delta_ranking_limit":"Raw block L2 is descriptive and mixes native amplitudes, log coordinates, dimensionless quantities and Hz; neither WCCN-scaled importance nor causal attribution. RMS per coordinate is also reported. Held clean frames are an internal no-fit attribution diagnostic only.",
        "synthetic_findings":{
            "gain_shape_max_abs_delta":max(row["gain_identity_check"]["max_abs_shape_delta"] for row in rows if "gain_identity_check" in row),
            "gain_prediction_max_abs_error":max(row["gain_identity_check"]["all362_max_abs_prediction_error"] for row in rows if "gain_identity_check" in row),
            "trace_floor_crossings":0,
            "noise":noise_findings,
            "frame_note":"Noise frame-pair motion is small (Frobenius distance <0.013) in these fixtures; holding the clean frame still gives large deficient-row spectral changes. Small frame motion can also switch dominant-frequency bins in weak channels. These two fixtures do not exclude geometry sensitivity on other windows.",
        },
        "mathematical_identities":[
            "For positive shared gain g, frame pair B(gX)=B(X) and all three validity ratios are unchanged, up to roundoff; the internal numerical rescaling is undone for amplitudes.",
            "Above inherited energy cutoffs, f(gX) has 150 amplitude coordinates multiplied by g, 210 shape/lag-eigenspectrum coordinates unchanged, and 2 log-trace coordinates shifted by 2 ln(g). Thus canonical362 is mathematically NOT gain invariant.",
            "Lag26: G=Zc Zc^T/235, q_i=lambda_i(G/max(tr G,1e-8)), h_i=ln max(q_i,1e-8); append ln max(tr G,1e-8), separately per native triad. Positive gain cancels in q_i only above the trace floor. A floored q_i is flat until it crosses 1e-8, then dh/dq=1/q.",
            "For independent time/axis white noise, E[G(X+noise)]=G(X)+sigma^2 K. K has diagonal 1; entries for different axes are zero; same-axis distinct lags separated by d have K_ij=-(L-d)/(L(L-1)) for d<L, L=236. Thus K=I+O(1/L). Approximate q_i'=(lambda_i+sigma^2)/(tr G+12 sigma^2) neglects overlapping-mean corrections. Finite-draw signal/noise cross terms remain.",
            "If sigma=p*r for this same window and r^2=mean_axis_time(centered triad^2), then tr G approximately 12r^2 and a null q_i rises to about p^2/[12(1+p^2)]: 8.33e-6 at 1%, 2.08e-4 at 5%. This is not an actual TRAIN/probe scale estimate.",
            "First differences of independent white noise have variance 2 sigma^2 per axis. A full-cycle sinusoid x=A sin(omega t) has Var(diff x)=2 A^2 sin^2(omega/2)=4 sin^2(omega/2) Var(x), ignoring the one-sample endpoint imbalance. Slow clean motion can make differences disproportionately noise sensitive.",
            "For a zero centered channel, normalized spectral features are zero; arbitrarily small nonzero noise above total-power cutoff has a unit-sum random spectrum. Spectrum normalization can expose noise without a large absolute signal change. Entropy log guards alone do not establish amplification.",
            "New full/log covariance extractor is NOT canonical362: S=.8 G/max(tr G,1e-8)+(.2/12)I; its eigenvalues are >=1/60, limiting log derivative to 60 with respect to S eigenvalues; it still appends the gain-sensitive log trace.",
            "Fixed WCCN: z=((f-mu)/s)P, s_j=max(TRAIN std_j,1e-6), P=Q diag(max(lambda_i,max(1e-5 lambda_max,1e-6))^(-1/2))Q^T, e=z/||z||. delta z=(delta f/s)P; first-order delta e=delta z(I-e^T e)/||z||. Actual directional gain depends on s,P,z and perturbation alignment, not dimension count alone.",
        ],
        "source_locations":{
            "composition":"scripts/canonical_orientation_study.py:114-129; scripts/mentor_backend_study.py:173-197",
            "geometry_and_amplitude":"models/gait_canonical.py:122-209,235-286; docs/gait_canonical.md",
            "axis_features_and_floors":"models/gait_features.py:41-87; models/gait_canonical.py:212-232",
            "invariant_features_and_floors":"models/gait_invariants.py:89-183",
            "actual_lag26":"models/session_robust.py:27-96",
            "distinct_regularized_covariances":"models/lag_covariance.py:13-33,93-147; docs/lag_covariance.md",
            "standardization_and_WCCN":"models/gait_metric.py:46-94,96-124",
            "noise_reference_and_generation":"scripts/covariance_robustness_study.py:112-171; data/imu_perturbations.py:327-367",
        },
        "minimal_numeric_checks":{"log_trace_gain09":float(2*np.log(.9)),"log_trace_gain11":float(2*np.log(1.1)),"log_floor":float(np.log(1e-8)),"null_mode_1pct_expected_log_shift":float(np.log(.01**2/(12*(1+.01**2)))-np.log(1e-8)),"null_mode_5pct_expected_log_shift":float(np.log(.05**2/(12*(1+.05**2)))-np.log(1e-8)),"independent_lag_max_abs_error":independent_max,"illustrative_metric":metric_illustration},
        "failed_or_unsupported_hypotheses":[
            "Pure shared positive gain rotates an otherwise identifiable PCA frame: contradicted mathematically and checked numerically here; gain changes amplitude/log-energy features instead.",
            "Every valid canonical row has well-conditioned lag covariance: contradicted by valid rank-2 lag-stack fixtures. Frame guards check gravity/horizontal geometry, not lag-spectrum rank or spectral stability.",
            "The newer .2-shrunk matrix logarithm is the canonical362 lag block: false by source composition. It cannot explain canonical362 errors.",
            "All noise sensitivity must be frame instability or eigenvalue logs: unsupported; held-frame and full-rank measurements are recorded and axis spectra/differences have separate sensitivity mechanisms.",
            "A large scalar WCCN gain necessarily changes cosine scores: false; isotropic scaling cancels in L2 normalization. Actual directional amplification cannot be determined without the frozen metric state and DEV perturbation vectors.",
        ],
        "future_registered_DEV_only_diagnostics":[
            {"hypothesis":"Gain errors are mediated by amplitude/log-trace changes under a frozen metric, rather than PCA reorientation.","diagnostic":"Register a diagnostic-only paired replay with the existing frozen clean metric/threshold/enrollment: record delta f, (delta f/s)P, embedding norms and genuine-score shifts for amplitude, log-trace, and remaining blocks, separately and jointly. Keep every row/claim; compare to the analytic gain prediction; do not refit or promote a new candidate.","falsifier":"Gain mainly changes invariant shape columns, frames, or score shifts persist after diagnostic restoration of both amplitude and log-trace blocks to clean values (within numerical tolerances)."},
            {"hypothesis":"Noise score loss is partly mediated by low lag modes crossing the log floor and/or previously weak channel spectra, with learned metric amplification.","diagnostic":"Before a future DEV-only run, register paired floor-crossing counts, spectral powers, canonical eigengaps and frame motion; raw versus standardized/projected block deltas; clean-metric block-restoration diagnostics for lag26, axis spectra/differences and remaining blocks. Preserve all rows and fixed scoring/thresholds; count abstentions explicitly; no stress-based selection or fitting.","falsifier":"Little real-DEV floor crossing/weak-spectrum activation occurs, those deltas project weakly, or restoring those blocks fails to recover genuine-score shifts. Large held-frame versus reestimated-frame differences would redirect attribution toward geometry."},
        ],
        "interpretation":"The user-supplied clean/gain/noise FAR/FRR values are reused-DEV observations, not causally explained by this source/synthetic probe. No raw archive/dataset, NPZ/large numerical artifacts, original CALIB/TEST data or results were opened. No real-data or synthetic metric fitting, dependencies, Git changes, production edits, or new biometric candidate.",
        "regularized_covariance_comparison_only":matrix_comparison,
        "rows":rows,
    }
    assert git("status","--porcelain=v1")==status_before
    data=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    (OUT/"summary.json").write_text(data)
    print(json.dumps({"summary":str(OUT/"summary.json"),"bytes":len(data),"coverage":result["coverage"],"checks":"passed"}))


if __name__=="__main__":
    main()

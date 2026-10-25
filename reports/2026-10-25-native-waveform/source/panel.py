"""Fixed 1440-input N446 panel. keys()/proposal() do not generate numeric inputs."""
import numpy as np
import common as c
import legacy

N = 446
PHASES = (0, 8, 16, 24)
STRESS_PERIODS = (160, 180, 200)
CONDITIONS = ("clean", "noise_both_0.1_acrms", "noise_gyro_4_acrms",
              "gyro_replaced_4_acrms_noise", "both_replaced_4_acrms_noise",
              "constant_gyro_4", "near_constant_gyro_4_plus_1e-13_cos", "both_half_period")
TRANSFORMS = ("common_Rz30", "common_Rx90", "modality_gains_0.9_1.1", "modality_gains_1.1_0.9")


def keys():
    result = []
    def add(kind, family, period, phase, condition="original", transformation="identity"):
        result.append(dict(input_index=len(result), kind=kind, family=family,
                           period_samples=period, phase_index=phase, length=N,
                           condition=condition, transformation=transformation,
                           key=f"{kind}/{family}/P{period}/phase{phase:02d}/N446/{condition}/{transformation}"))
    for family in legacy.FAMILIES:
        for period in legacy.PERIODS:
            for phase in range(32):
                add("legacy", family, period, phase)
    for period in STRESS_PERIODS:
        for phase in PHASES:
            for condition in CONDITIONS:
                add("stress", "analytic_positive", period, phase, condition)
    for family in legacy.FAMILIES:
        for phase in PHASES:
            for transform in TRANSFORMS:
                add("invariance", family, 160, phase, transformation=transform)
    c.require(len(result) == len({r["key"] for r in result}) == 1440, "Fixed key inventory")
    return result


def rotation(name):
    if name == "common_Rz30":
        angle = np.pi / 6
        return np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    if name == "common_Rx90":
        return np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    raise ValueError(name)


def stress_setting(period, phase):
    # Exactly one row-major 768x2 draw per P/phase, shared by all conditions.
    rng = np.random.Generator(np.random.PCG64(20261025 + 1000 * STRESS_PERIODS.index(period) + phase))
    noise = rng.uniform(-np.sqrt(3.), np.sqrt(3.), size=(768, 2))[:N]
    theta = 2 * np.pi * (np.arange(N, dtype=float) / period + phase / 32)
    a, g = 10 + np.cos(2 * theta), 4 + .5 * np.cos(theta)
    sigma_a, sigma_g = 1 / np.sqrt(2.), 1 / (2 * np.sqrt(2.))
    na, ng = sigma_a * noise[:, 0], sigma_g * noise[:, 1]
    values = ((a, g), (a + .1 * na, g + .1 * ng), (a, g + 4 * ng),
              (a, 4 + 4 * ng), (10 + 4 * na, 4 + 4 * ng),
              (a, np.full(N, 4.)), (a, 4 + 1e-13 * np.cos(theta)),
              (a, 4 + .5 * np.cos(2 * theta)))
    zero = np.zeros(N)
    result = {}
    for condition, (acc, gyro) in zip(CONDITIONS, values):
        c.require(bool(np.all(acc > 0) and np.all(gyro > 0)), "Positive analytic lift")
        result[condition] = np.asarray([zero, zero, acc, zero, zero, gyro])
    return result


def generate(item, stress_cache):
    c.require(item["length"] == N, "N446 only")
    if item["kind"] == "stress":
        setting = (item["period_samples"], item["phase_index"])
        if setting not in stress_cache:
            stress_cache[setting] = stress_setting(*setting)
        return stress_cache[setting][item["condition"]].copy()
    # Original arithmetic, analytically generated at N446; no concatenate/prefix input assembly.
    base = legacy.fixture(item["family"], item["period_samples"], item["phase_index"], N)
    if item["kind"] == "legacy":
        return base
    transform = item["transformation"]
    if transform.startswith("common_R"):
        r = rotation(transform)
        return np.vstack((r @ base[:3], r @ base[3:]))
    gains = (.9, 1.1) if transform == TRANSFORMS[2] else (1.1, .9)
    return np.vstack((gains[0] * base[:3], gains[1] * base[3:]))


def proposal():
    return dict(study="native-waveform-feasibility-r1", N=N, sampling_hz=100,
        status="PROPOSED_NOT_FROZEN_NOT_MEASURED", adaptive_exploratory=True,
        adaptation="Post previous outcomes; exact plateau convention outcome-informed by prior 12-fixture study. Prospectively fixed for this addendum; blinded to latest joint-study outcomes.",
        interpretation="New synthetic addendum, not IDNet/Barth reproduction; canonical 362 nominee unchanged; no DEV fit or execution.",
        metadata_rationale=dict(eligible_segments=299, lengths=[446,931],
            ideal_phase_geometry_hole=False, maximum_period=200, margin=10,
            numerical_minima_and_correlations="unknown",
            completed_result_sha256="80be0bc4e54ff1244a2ff8cf909dccf5ca7c3c2ee5189f58f64e9991e32aa25b",
            secondary_arithmetic="artifacts/complete-support-root-metadata-analysis-20261025.json"),
        counts=dict(total=1440,legacy=1280,nondegenerate_legacy=960,weak_gyro_legacy=320,
                    balanced_half_admissible=96,stress=96,invariance=64),
        ordering="legacy family/period/phase; stress period/phase/condition; invariance family/phase/transform",
        families=list(legacy.FAMILIES), periods=list(legacy.PERIODS), phases=list(range(32)),
        generator="Verbatim legacy.fixture, analytic N446. Root execute full 1280 N256-prefix bit gate BEFORE any N446 candidate; no parent measurements needed.",
        stress=dict(periods=list(STRESS_PERIODS),phases=list(PHASES),conditions=list(CONDITIONS),
                    theta="2*pi*(arange(446)/P+phase/32)",a="10+cos(2theta)",g="4+.5*cos(theta)",
                    noise="ONE Generator(PCG64(20261025+1000*j+actual_phase_index)).uniform(-sqrt3,sqrt3,size=(768,2)) row-major per setting; first446; sigma_a=1/sqrt2 sigma_g=1/(2sqrt2); DC preserved; no noise centering/scaling",amplitude_replacement=4),
        invariance=dict(period=160,phases=list(PHASES),transforms=list(TRANSFORMS)),
        filter=dict(order=2,cutoff_hz=3.,sampling_hz=100.,padtype="odd",padlen=9,sos=c.SOS.tolist()),
        lag=dict(range_inclusive=[80,200],pearson="np.corrcoef on SAME filtered prefix/suffix lag per modality",joint="equal modality mean; either undefined => joint undefined",selection="smallest ABS-nearmax <=1e-10, rtol0",guard="every vector finite, q=uncentered RMS>0, population std>1e-12*q"),
        boundary=dict(convention="maximal EXACT equal run [u,v], both exterior neighbors strictly higher; floor((u+v)/2)",margin=[10,435],pair="ALL ordered a<b including intervening minima; 80<=D=b-a<=200; integer 10*abs(D-lag)<=lag",rank="(abs(D-lag),abs(a+b-445),a,b)",strict="diagnostic only, cannot rescue/select"),
        mapping=dict(source="UNFILTERED raw magnitudes at integer knots a..b inclusive",samples=200,
                     coordinates="a+(b-a)*k/200 for k=0..199; right endpoint excluded from outputs, included as interpolation knot",
                     interpolation="piecewise LINEAR; no wrap/extrapolation/refinement/independent alignment",
                     normalization="per-channel center/population std, same finite/RMS/relative std guard; acc first then gyro; NO global normalization; theoretical norm20",
                     phase_zero="selected acceleration minimum, not anatomical event"),
        gates=dict(fixed_numerical_checks=True,nondegenerate_legacy="ALL960 valid AND abs(lag-P)<=1 AND abs(duration-P)<=1, including96 balanced half-admissible",
                   invalid="ALL320 weak gyro plus constant/nearlyconstant stress invalid",
                   invariance="same validity/lag/pair; both valid feature absdelta<=1e-10 rtol0",
                   mapping_means_atol=1e-12,mapping_variance_atol=1e-12,mapping_norm_atol=1e-10,
                   map_checks="verification only, never extraction guards",noise="all stress diagnostics unconditionally reported; no noise quality gate",
                   consequence="ANY failure stops; no outcome-dependent change/search/fallback/retry. Pass only permits DEV REGISTRATION PREPARATION eligibility."),
        quality_cutoff=None, duration_amplitude_quality_features=False,
        denominator_policy="all1440 retained, noise-only false-valid labeled not_walking, bothhalf repeat P/2 no anatomical requirement",
        runtime=dict(cpu_process_seconds=600,wall_seconds_including_uv=900,affinity_count=1,numerical_threads=1,cuda=False,offline=True,installs=False,retry=False),
        preparation_scope="fixed scalar handchecks + one nonprimary gravity P100 phase0 N446 fixture, its N256 prefix and four transforms only; no primary96/full panel",
        unresolved_brief_ambiguities=[],root_review_required=True)

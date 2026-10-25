"""Sole N446 candidate. No generator or measurement side effects."""
import numpy as np
import common as c
import guards
import legacy
import plateau_helpers


def pairs(minima, lag, n):
    if lag is None:
        return [], None
    eligible = [[a,b] for a in minima for b in minima
                if 80 <= b-a <= 200 and 10*abs(b-a-lag) <= lag]
    chosen = min(eligible, key=lambda ab:(abs(ab[1]-ab[0]-lag),abs(ab[0]+ab[1]-(n-1)),*ab)) if eligible else None
    return eligible, chosen


def linear_mapping(mags, pair):
    a, b = pair
    c.require(type(a) is int and type(b) is int and 0 <= a < b < mags.shape[1], "Integer mapping knots")
    coordinates = a + (b-a)*np.arange(200,dtype=np.float64)/200
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower+1, b)
    fraction = coordinates-lower
    # The b endpoint remains a knot even though no output coordinate equals b.
    values = mags[:,lower] + (mags[:,upper]-mags[:,lower])*fraction
    stats, good, finite = [], [], []
    for channel in values:
        ok, stat, fin = guards.vector_guard(channel)
        stats.append(stat); good.append(ok); finite.append(fin)
    stats = np.asarray(stats)
    means = np.mean(values,axis=1)
    feature = None
    if all(good):
        feature = ((values-means[:,None])/stats[:,1,None]).reshape(400)
        if not np.isfinite(feature).all():
            feature = None
    return dict(coordinates=coordinates,lower=lower,upper=upper,fraction=fraction,
                samples=values,stats=stats,means=means,valid=np.asarray(good),
                finite=np.asarray(finite),feature=feature)


def mapping_checks(feature):
    if feature is None:
        return None
    values = feature.reshape(2,200)
    means = np.mean(values,axis=1)
    variances = np.var(values,axis=1)
    norm = float(np.linalg.norm(feature))
    return dict(channel_means=means.tolist(),channel_variances=variances.tolist(),norm=norm,
                pass_verification_only=bool(np.all(np.abs(means)<=1e-12) and
                    np.all(np.abs(variances-1)<=1e-12) and abs(norm-20)<=1e-10))


def alias(value, period):
    if value is None:
        return "unavailable"
    targets = ((period/2,"half"),(period,"one"),(2*period,"double"))
    target, label = min(targets,key=lambda p:abs(value-p[0]))
    return label if abs(value-target)<=.1*target else "other"


def decide(item, mags, filtered, correlations):
    """Saved-evidence decision path: no generation, filtering or fitting."""
    lag, best, ties, near = legacy.select(correlations[2],legacy.RULES[1])
    finite_acc = bool(np.isfinite(filtered[0]).all())
    runs = plateau_helpers.exact_plateau_runs(filtered[0]) if finite_acc else []
    minima = [r["representative"] for r in runs if r["eligible_minimum"]]
    strict = [i for i in range(10,filtered.shape[1]-10)
              if filtered[0,i]<filtered[0,i-1] and filtered[0,i]<filtered[0,i+1]] if finite_acc else []
    eligible, pair = pairs(minima,lag,filtered.shape[1])
    strict_pairs, strict_pair = pairs(strict,lag,filtered.shape[1])
    mapped = linear_mapping(mags,pair) if pair is not None else None
    feature = None if mapped is None else mapped["feature"]
    failure = "no_joint_lag" if lag is None else "no_plateau_pair" if pair is None else "mapping_vector_guard" if feature is None else None
    duration = None if pair is None else pair[1]-pair[0]
    period = item["period_samples"]
    whole = guards.whole_window(mags,filtered)  # independent of first failure
    record = dict(**item,selected_lag=lag,maximum_joint_correlation=best,exact_max_ties=ties,
        nearmax_lags=near,correlation_summary=legacy.correlation_summary(correlations[2]),
        selected_modality_correlations=None if lag is None else correlations[:2,lag-80].tolist(),
        plateau_run_count=len(runs),plateau_trough_runs=[r for r in runs if r["is_trough"]],
        plateau_minima=minima,eligible_pairs=eligible,selected_pair=pair,
        duration=duration,strict_diagnostic=dict(minima=strict,eligible_pairs=strict_pairs,
                selected_pair=strict_pair,status="no_lag" if lag is None else "no_pair" if strict_pair is None else "pair_available",can_select_or_rescue=False),
        whole_window=whole,whole_window_degenerate=any(row["degenerate"] for row in whole),
        valid=feature is not None,failure=failure,quality_cutoff=None,quality_evaluated=False,
        lag_error=None if lag is None else lag-period,duration_error=None if duration is None else duration-period,
        lag_alias=alias(lag,period),duration_alias=alias(duration,period),
        mapping_guards=None if mapped is None else [dict(finite=bool(mapped["finite"][i]),valid=bool(mapped["valid"][i]),
             q=float(mapped["stats"][i,0]) if np.isfinite(mapped["stats"][i,0]) else None,
             std_population=float(mapped["stats"][i,1]) if np.isfinite(mapped["stats"][i,1]) else None) for i in range(2)],
        mapping_verification=mapping_checks(feature),
        noise_only=item["condition"]=="both_replaced_4_acrms_noise",
        noise_only_false_valid=bool(item["condition"]=="both_replaced_4_acrms_noise" and feature is not None),
        walking_label="not_walking" if item["condition"]=="both_replaced_4_acrms_noise" else "synthetic_no_walking_claim",
        bothhalf_available_repeat=period/2 if item["condition"]=="both_half_period" else None,
        phase_zero="selected_acc_minimum",anatomical_event=False)
    return record, mapped


def extract(item, raw):
    c.require(raw.shape==(6,446),"Only 6x446 candidate input")
    mags = guards.magnitudes(raw)
    filtered = guards.filter_modalities(mags)
    corr,stats,valid,finite = guards.joint_correlations(filtered)
    record,mapped = decide(item,mags,filtered,corr)
    return record,dict(raw=raw,magnitudes=mags,filtered=filtered,correlations=corr,
                       correlation_stats=stats,correlation_valid=valid,correlation_finite=finite),mapped

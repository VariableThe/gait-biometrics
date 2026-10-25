# Exact nuisance-study promotion gates

`engine.robustness_selection` fits no models and loads no files. The caller
supplies twenty registered DEV validation triples, the fifteen DEV IDs and all
280 outer rows: two arms (`baseline`, `nuisance`) by seven conditions by twenty
folds. Every person must occur four times; claim graphs, person denominators and
all six directed pairs must match within a fold across arms and conditions.

`ranking_metrics(labels, scores, probe_ids)` groups tied scores before deriving
ROC endpoints. It calculates EER by exact rational interpolation between the
crossing endpoints and trapezoidal AUC with exact fractions. It returns canonical
rational strings overall and per probe person. Both binary classes must occur
in every group. This matches the historical ROC interpolation, while avoiding
floating-point promotion decisions. These are ranking diagnostics, not new
decision thresholds.

`summarize_nuisance_study(rows, *, dev_ids, folds)` requires the historical row
fields `fold`, `arm`, `condition`, `metrics`, `pair_errors`, plus `ranking` from
the above API and `decision_sha256`, a hash of the ordered acceptance vector at
the frozen threshold. The caller must independently bind these values to source
keys and replayed score caches. A valid hash string alone proves no decisions.
Stored display rates and floating EER do not drive selection.

Promote only if **all** gates pass:

- Clean equal-fold FAR is no worse than baseline and at most 1%; the number of
  folds above 2% FAR, equal-fold FRR, and total repeated genuine rejects do not
  increase. At least ten people have no worse mean FRR; nobody gains more than
  one repeated reject or becomes completely rejected after being usable.
- Clean mean exact EER does not increase. Report per-person EER and AUC as well.
- Equally weighted mean FRR over gain0.9, gain1.1 and noise1% improves by at least
  five percentage points; none of these condition means increases. At least ten
  people have no worse averaged stress FRR; nobody worsens by over five points.
- Every condition has mean FAR at most 1%, no additional folds above 2% FAR,
  and no increase in repeated false accepts targeting any claimant.
- Both fixed rotation conditions retain that candidate's ordered clean
  decisions. The severe noise5% endpoint is reported and has the same FAR gates,
  but does not enter the FRR improvement objective.

Fold rates have equal weight. Person rates are means over their four dependent
appearances. Pooled repeated counts are explicitly separate. Conditions and
gates were designed after inspecting prior DEV stress failures; all evidence
remains exploratory. Neither rational arithmetic nor these gates provides
independent confirmation or a population FAR guarantee. Original CALIB/TEST
must stay closed for this experiment.

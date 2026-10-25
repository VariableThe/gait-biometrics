# Locked synthetic length recreation

Completed one exclusive frozen attempt. Root review precedes publication or any real DEV plan.

Legacy control: all 1,280 cached records, all cached fields, match exactly; 1,280 six-axis nested prefixes are bitwise identical.

| N | Rule | Finite / 1280 | Missing pair | Degenerate gyro | Finite half aliases |
|---|---|---:|---:|---:|---:|
| 256 | exact_max_smallest_exact_tie | 848 | 192 | 240 | 96 |
| 256 | nearmax_atol_1e-10_smallest | 848 | 192 | 240 | 96 |
| 768 | exact_max_smallest_exact_tie | 959 | 1 | 320 | 96 |
| 768 | nearmax_atol_1e-10_smallest | 959 | 1 | 320 | 96 |

Run process CPU: 64.069865s; wall after imports: 66.662143s.

All 5,120 record rows and 1,280 four-channel long-window diagnostics are retained. All fixed numerical checks pass; exact-rule divergences are expected.

- Only four noiseless analytic fixtures, two preregistered lengths and two rules; these counts do not estimate gait, subject, session, authentication or real-DEV performance.
- N768's 620-sample geometric rationale is not an empirical optimum or sufficient condition for correct stride extraction or finite features.
- Balanced acceleration has period P/2; physical stride is unidentifiable from acceleration alone. Its gyro magnitude has period P, so joint magnitudes in THIS fixture are not unidentifiable.
- In general identical joint magnitudes can be ambiguous, but no additional generator was planned or tested. Full-signal and extracted-interval aliases are separately flagged.
- Nearmax selection stabilizes only the fixed numerical checks. A unique nearmax set, high correlation, or finite feature does not identify anatomical stride.
- Long-window harmonic diagnostics use common fixed anchors and never rescue extraction. Filter boundaries can change observed minima relative to independent raw-minimum geometry.
- Hard near-flat gyro guards remain failures without imputation. Aliases remain extraction successes when features are finite; no quality cutoff is trained or applied.
- IDNet section3's eight channels, orientation transform and adaptive templates are absent; this two-magnitude experiment is not an IDNet reproduction. Derawi paper is 2010 DOI10.1109/IIHMSP.2010.84.
- Independent root review is required before publication or any real DEV plan. No such work is performed here.

Hashes and scientific fingerprint are in report.json. Named lossless audit_arrays.npz includes six raw axes, correlations, vectors, masks and row maps. audit.py performs saved-artifact checks without new extraction.

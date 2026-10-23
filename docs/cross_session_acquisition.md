# Santos cross-day acquisition and preregistered descriptor protocol

Research checked 2026-10-23 UTC. The population split and descriptor protocol below were specified from metadata before reading test signals. The implemented reader is documented in [santos_dataset.md](santos_dataset.md); experiment commands are in [cross_day_verification.md](cross_day_verification.md). Results, when available, belong to the separate cross-day report.

The official v7 archive was subsequently downloaded and verified against the published size/MD5. Its measured SHA-256 is `ed944fa88a98ce1058327ff4b3c8f919fbf58626fc7e3c5256360ee080a3a304`. Filename-only inspection found all 25 expected participants and 499 processed MCU trials: `user12/day2/trial0009` is absent. Inspection of one **development** file confirmed the exact seven-column header `frame,acc x,acc y,acc z,gyro x,gyro y,gyro z`. The primary protocol still uses only the three acceleration channels; gyroscope readings are not an undeclared extra feature variant. No held-out signal values were needed to resolve the schema.

## Acquisition decision and exact official manifest

The [version-pinned Figshare API](https://api.figshare.com/v2/articles/14727231/versions/7) returns **one ZIP**, with no separately downloadable CSV samples and an empty `folder_structure`. Acquisition reads compressed archive bytes for integrity; only explicitly selected subject files are parsed by the loader.

- Dataset DOI: [10.6084/m9.figshare.14727231.v7](https://doi.org/10.6084/m9.figshare.14727231.v7).
- Dataset title: *A multi-sensor human gait dataset captured through an optical system and inertial measurement units*.
- Authors: Geise Santos, Marcelo M. Wanderley, Tiago Tavares, Anderson Rocha.
- Figshare article ID/version: `14727231`, `7`; file ID: `35882624`.
- Exact file name: `A multi-sensor human gait dataset.zip`.
- Stable file download URL: <https://ndownloader.figshare.com/files/35882624>.
- Exact size: **1,523,321,221 bytes** (1.523 decimal GB; approximately **1.419 GiB**, explaining the earlier “1.42 GB” display).
- Supplied MD5: `35adf0860202f2aefa37560dc0748ebd`.
- Figshare-computed MD5: `35adf0860202f2aefa37560dc0748ebd`.
- MIME type: `application/zip`; public, download enabled, not embargoed, not link-only.
- Version publication time: `2022-06-14T22:12:45Z`.
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution, version DOI, license link, and descriptions of derived transformations.

The API response and filename inventory are retained with the cross-day evidence. MD5 and size are publisher-reported; the SHA-256 above was measured from bytes matching both. The reader uses selected ZIP members directly and never extracts paths to disk. A reusable downloader validates containment, rejects destination symlinks and checks the pinned SHA-256 before promoting a partial download.

```bash
uv run python -m scripts.setup_santos_dataset
# Verify/reuse an existing archive without downloading:
uv run python -m scripts.setup_santos_dataset --verify-only
```

HEAD on the stable URL returns a 302 redirect to Figshare's S3 storage. Following that redirect with HEAD, and HEAD on the unsigned storage path, returned 403. This does **not** establish that the normal signed GET download fails; the signed redirect may depend on the request method. No ZIP body, range request, member payload, or GPU work was performed. Sanitized HEAD evidence is in `artifacts/cross_session_research/figshare_35882624_head.txt` and `figshare_35882624_storage_head.txt`. Local DNS initially returned `0.0.0.0` for Figshare; the API and redirect HEAD succeeded using a per-request public-DNS-resolved address with normal TLS verification. Do not hard-code those short-lived addresses in a downloader.

Acquisition and descriptor evaluation use CPU resources. The three-channel acceleration protocol is explicit despite the six-axis source. It requires no additional framework or interruption of the GPU scaling study.

## Authentic subjects, sessions, and metadata

The [Scientific Data descriptor](https://www.nature.com/articles/s41597-022-01638-2) documents 25 participants, two sessions on different days, and ten trials per session. The per-participant identifiers range within `01`–`30` and are nonconsecutive. The labels `day1` and `day2` mean that participant's first and second acquisition days; they are not global calendar dates shared across people.

The first author's [companion analysis repository](https://github.com/geisekss/gait_reliability_investigation), pinned at commit `23930c7d3ba69d63e94b9de503b0d3096efb7875`, exposes the following directory names through its [contents API](https://api.github.com/repos/geisekss/gait_reliability_investigation/contents/csv_files?ref=23930c7d3ba69d63e94b9de503b0d3096efb7875):

```text
user01 user03 user05 user06 user07 user09 user11 user12 user13
user14 user15 user16 user17 user18 user19 user21 user22 user23
user24 user25 user26 user27 user28 user29 user30
```

This is an observed 25-person author-provided listing, saved as `artifacts/cross_session_research/author_companion_participant_listing.json`. It is **supporting evidence, not an audit of the v7 ZIP**. Check the archive's participant directories against this exact set using filenames only before accepting the proposed split. Do not silently add the absent IDs `02`, `04`, `08`, `10`, or `20`, relabel gaps into new people, or assume matching numeric IDs in unrelated datasets identify the same person.

The companion `user01` listing confirms real names such as `capture_user01_day1_0001_imu_walk.csv` and `capture_user01_day2_0001_imu_walk.csv`, and trials `0001`–`0010` on both days. Its metadata is saved in `author_companion_user01_file_metadata.json`; no corresponding signal body was read. It does not prove that the companion repository and v7 contain byte-identical data, and its Git blob SHA is not an MD5 for an official Figshare file.

Published participant metadata has limits:

- The descriptor's abstract and Figshare description report ages 18–47; its Participants section says 18–45. Retain this discrepancy rather than selecting an unsupported participant-level maximum.
- The descriptor reports 12 women, 12 men, and one undeclared gender. The authors' [related Scientific Reports study](https://www.nature.com/articles/s41598-022-12452-6) describes 12 male, 12 female, and one non-binary participant. These aggregate descriptions do not provide an ID-to-gender mapping.
- Participants reported no leg injuries or conditions affecting gait/posture. Collection occurred December 2019–February 2020. Exact per-person dates and day gaps are not exposed by the API or descriptor.
- The related study explicitly says height and weight were not collected. No per-subject demographics table was found in the checked sources; an uninspected archive might contain additional documentation.
- The paper specifies five trials with the band on the left leg and five on the right leg per day. The checked sources do not establish a trial-number-to-leg mapping. **Do not assume `0001`–`0005` means left.** Retain `placement_side=unknown` until an explicit source verifies it. A study pooling all ten trials must report a mixture of leg placements rather than claim matched-side control.

## Raw versus processed data and sensor schema

The descriptor's Data Records section distinguishes:

- `raw_data/<participant>/`: complete trials, including preparation, taps, walking, and the ending. Documented trial files are `capture_userID_dayD_TT_qtm.c3d` and `capture_userID_dayD_TT_imu.csv`.
- `processed_data/<participant>/`: walking sections selected using optical trajectories. Documented names are `capture_userID_dayD_TT_qtm_walk.mat`, `capture_userID_dayD_TT_qtm_acc_walk.mat`, `capture_userID_dayD_TT_imu_walk.csv`, and `capture_userID_dayD_TT_nexus_walk.csv`.

Here `userID` corresponds to names such as `user01`, `dayD` is literally `day1` or `day2`, and `TT` is a four-digit trial number `0001`–`0010`. The filename and its containing participant folder must agree. Preserve the original path and all labels; a recording key is `(dataset_version, participant, day, trial, sensor, processing_level)`.

The MCU uses an ESP8266 and MPU-9250; the phone is a Nexus 5 with MPU-6515. Acquisition is nominally **100 Hz**. [Paper Table 2](https://www.nature.com/articles/s41597-022-01638-2/tables/3) documents MCU column order as `Frame`, `Acc x`, `Acc y`, `Acc z`, `Gyro x`, `Gyro y`, `Gyro z`: integer trajectory frame index, three accelerations in **m/s²**, and three angular velocities in **degrees/s**. MCU accelerations were converted from g using **9.81 m/s²**, so converting an already converted CSV again would be wrong. Frame numbering need not start at one.

[Paper Table 3](https://www.nature.com/articles/s41597-022-01638-2/tables/4) documents three phone acceleration columns in **m/s²**. Android acceleration includes gravity. Phone capture was continuous over a day's trials and subsequently aligned by correlation to MCU walking sections; processed phone CSVs have no corresponding trajectory frame numbers. MCU and optical capture were synchronized by the integration system. Do not describe phone acquisition as hardware-synchronous, or its rows as original MCU frames without evidence. Optical positions are in mm; optical derived accelerations are in mm/s², which are different modalities and units from the sensor readings.

**Resolved source difference:** the authors' [published extraction script](https://github.com/geisekss/motion_capture_analysis/blob/15c3f3c24a41e725b6044d1e58e37f8195e020f9/data_analysis/extract_walkSection_accelerations.py), lines 34–38, writes only acceleration indexed by `frame`. The pinned companion CSV also has three axes, whereas the official v7 development CSV has all six. These sources are not byte-identical. The implemented loader accepts only the observed official seven-column schema, validates all sensor fields and explicitly returns acceleration only. The Python extraction slice excludes its ending index whereas the paper describes inclusion; original processed rows and frames are preserved without attempting to reconstruct removed endpoints.

Header inspection used **development subjects only**. The loader checks header spelling/order, field count, finite sensors and frame continuity; malformed rows, duplicate/reversed frames and gaps break window continuity. It neither invents missing readings nor treats frame indices as a signal. No unit conversion or resampling is applied.

## Proposed frozen 15/5/5 population split

The parent requested seed **2026**, fifteen development subjects, five calibration subjects, and five untouched test subjects. To remove PRNG ambiguity, this proposal uses Python standard-library `random.Random(2026).shuffle(sorted_subject_strings)`; first 15 are development, next 5 calibration, last 5 test. The exact sets below were generated from **directory metadata only**, without opening signal values:

```text
development (permutation order):
user07 user16 user11 user09 user29 user27 user05 user13
user03 user01 user25 user24 user14 user19 user17

calibration:
user21 user18 user23 user12 user30

untouched test:
user26 user28 user22 user15 user06
```

Treat these explicit IDs as authoritative if adopting this proposal; NumPy and other seed-2026 generators need not yield the same permutation. Verify disjointness, complete 25-person coverage, and archive ID equality using metadata first. The official inventory check must not inspect test values. Keep the adapter and experiment opt-in and keep existing within-recording studies separate.

Population separation and day separation answer different questions. For every development-validation, calibration, and test subject, construct enrollment **only from day1**, and score probes **only from day2**. Never fit identity labels across development/calibration/test pools, concatenate different days into one recording, generate genuine enrollment/probe pairs from the same day, or use held-out identities as training negatives.

For descriptor/metric selection, use five folds over the 15 development subjects: split the development permutation above into consecutive groups of three for validation, with the other twelve as fold training. All fitted feature normalization, feature selection, PCA, covariance/shrinkage estimates, metric weights, or learned encoders use **only the twelve training identities** in that fold. Training identities may contribute both days to learn cross-day relationships. Fold validation uses only held-out identities' day1 enrollment and day2 probes. Compare a small prespecified set of descriptor/metric candidates using development folds, then freeze the winner and refit its learned transforms on all fifteen development subjects. Calibration subjects select only the operating threshold; they do not select descriptors or metric hyperparameters. Neither stage accesses test values.

External source pretraining, if used, needs its own declared cohort provenance and channel/placement compatibility. Cross-dataset participant numeric IDs alone cannot prove cohort overlap or independence. The minimal descriptor study requires no pretraining; neither calibration nor test participants may enter pretraining, fitted preprocessing, or tuning.

## Windowing, enrollment, and frozen-threshold scoring contract

The following are proposed design choices, not claims of reproducing the authors' experiment:

1. Start with walking-only processed **MCU acceleration**, pending development-header verification. Keep all ten trials per day; report pooled leg placement unless explicit metadata supports stratification. Any six-axis extension is a separately declared variant after units and actual channels are confirmed.
2. Preserve each trial as a separate recording. At native 100 Hz use **256 samples (2.56 s)** and hop **256** for enrollment and probe evaluation. Drop incomplete tails, report exclusion counts, and never pad across trial/day/subject/sensor boundaries. Any alternative window length must be chosen using development folds and frozen before calibration or test. The paper uses 256 samples/75% overlap, but the companion `generate_frames.py` actually sets `STEP=32` (87.5% overlap); copying it is not an unambiguous reproduction.
3. Carry original row offsets and MCU frame intervals with every window. Keep raw and processed versions of the same trial out of different evaluation roles. All transforms fitted across recordings use only the relevant training population. If resampling for an existing model is later proposed, anti-alias within each trial before windowing, record the rate and conversion, and validate compatibility on development subjects. The existing 50 Hz/nine-channel HAR representation is not automatically compatible with a 100 Hz MCU CSV.
4. Specify descriptor equations, channel order, normalization, distance direction, and template aggregation before calibration. A concrete simple enrollment policy is: average window descriptors within each day1 trial, then average those trial descriptors with equal trial weights to form one template per identity. This avoids automatically giving longer trials greater enrollment weight. Freeze whether unit normalization occurs before/after averaging and how zero norms are handled.
5. Use each day2 trial's mean window descriptor as the **primary probe unit**, scored against its own subject's day1 template for genuine decisions and every other subject template in the same held-out population for impostor decisions. Report optional window-level results separately. This uses the same aggregation on calibration and test and avoids pretending overlapping or repeated scores are independent trials. A subject needs at least one eligible trial on both days; missing coverage is reported under this rule, not repaired by exchanging identities or selecting favorable test windows.
6. After descriptor/metric selection and refit, calibrate the fixed threshold on the five calibration identities' day1 templates/day2 probes. For a requested 1% empirical calibration FAR, choose the least restrictive attainable threshold satisfying the target with an explicit acceptance inequality and tie handling. Freeze it before loading test values. Report the actual attained calibration FAR and counts; 1% is a target, not a guarantee.
7. Freeze a protocol artifact containing split IDs, feature/metric settings, window/exclusion rules, template/probe aggregation, calibration target, scoring inequality, tie handling, training provenance, code/config hashes, and calibration threshold before the first test evaluation. This research document supplies the population/session proposal; it is **not** a substitute for that implementation-specific freeze.

With five identities and ten eligible day2 trials per identity, the primary test has at most **50 genuine** and **200 directed impostor decisions**. The nominal FAR resolution is then 1/200 = **0.5 percentage points**; exclusions reduce support. Scores reuse subjects, templates, and trials, so these are not 250 independent observations. Report per-subject errors and denominators, a clearly specified subject-cluster uncertainty procedure where feasible, and the limitations of just five evaluation identities. Do not claim extreme low-FAR assurance from zero observed errors.

Report genuine/impostor counts, trial/window coverage, fixed-threshold test FAR and FRR, per-subject errors, and descriptive ROC/AUC/EER. A test-derived EER threshold is not the deployed/calibration threshold and must never be fed back into selection. Neither high AUC nor different recording days alone establishes passive continuous authentication, takeover delay, cross-device transfer, or population independence. The supported claim after a successful implementation would be **held-out-subject cross-day verification on this controlled, leg-mounted, short-walk dataset**, with the exact protocol attached.

## Acquisition and execution checks

The implemented acceleration study follows these checks:

- Verify the official v7 ZIP and compare filename-only participant/day/trial inventory before reading test values.
- Resolve schema using development files, preserve original frames and keep unknown leg placement explicit.
- Freeze the exact population split and implementation before fitting. Calibrate only after development selection; verify the lock before the single test attempt.
- Reject unknown layouts and keep every fitted descriptor transform inside the declared training population.

The initial acquisition research accessed metadata and documentation only. The separate implementation and measured report record subsequent data access, tests and results; they must not be inferred from this design document.

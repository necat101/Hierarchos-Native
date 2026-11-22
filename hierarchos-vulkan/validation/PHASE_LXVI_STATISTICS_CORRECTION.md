# Phase LXVI — dated statistical correction

Generated 2026-11-19 from the preserved raw samples by
`phase_lxvi_historical_ci.py`. No historical receipt, sample or hash was
modified.

## What was wrong

wmma_abba_analyze.py used a 12-entry two-tailed 95% t table and T95.get(df, 1.96); phase_lxv_stats.py extended the table to 32 degrees of freedom and kept the same 1.96 fallback. Both silently replace a finite-sample t quantile with the normal quantile, which makes intervals too narrow.

## What was done

Each pair's statistic is the median of that run's per-iteration samples (unchanged). The interval is the two-sided 95% Student-t interval over the per-pair ratios with df = pairs - 1, using a computed quantile. The point estimate and the geomean are unchanged.

The single implementation is `hierarchos-vulkan/validation/abba_stats.py`. It computes the
quantile from the regularized incomplete beta function, refuses degrees of
freedom outside its supported range, and contains no `1.96` fallback on any
live path.

## Recomputable sweeps

abba_a_vs_a_null, abba_a_vs_final, abba_full_main_vs_final, abba_main_vs_final

## Not recomputable here

* **phase_lxi / phase_lxii / phase_lxiv** — no raw per-iteration ABBA samples are present in this tree; only reported summary numbers survive. Those intervals are therefore not recomputed here and are not used for any new decision. Re-running the sweeps would be required.

## Old and corrected intervals

`legacy` is the interval the superseded `T95.get(df, 1.96)` rule printed.
`lxv` is the interval the Phase LXV in-tree table printed (it mattered only
where the table was missing). `corrected` is the canonical t interval.

| Sweep | Shape | n | legacy CI | lxv CI | corrected CI | widen | conclusion |
|---|---|---:|---|---|---|---:|---|
| abba_a_vs_a_null | 128x8192x768 | 16 | [0.9947, 1.0600] | [0.9918, 1.0629] | [0.9918, 1.0629] | 2.86e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 16x128x32 | 16 | [0.9891, 1.0049] | [0.9884, 1.0056] | [0.9884, 1.0056] | 6.94e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 2048x768x2048 | 16 | [0.9923, 1.0047] | [0.9918, 1.0053] | [0.9918, 1.0053] | 5.42e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 512x2048x768 | 16 | [0.9805, 1.0077] | [0.9793, 1.0089] | [0.9793, 1.0089] | 1.19e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 512x768x768 | 16 | [0.9976, 1.1802] | [0.9896, 1.1882] | [0.9896, 1.1882] | 7.99e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 64x64x64 | 16 | [0.9916, 1.0153] | [0.9906, 1.0163] | [0.9906, 1.0163] | 1.04e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 128x8192x768 | 16 | [0.9975, 1.0041] | [0.9972, 1.0044] | [0.9972, 1.0044] | 2.91e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 16x128x32 | 16 | [0.9807, 1.0078] | [0.9795, 1.0090] | [0.9795, 1.0090] | 1.19e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 2048x768x2048 | 16 | [0.9879, 1.0328] | [0.9859, 1.0348] | [0.9859, 1.0348] | 1.97e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 512x2048x768 | 16 | [0.9963, 1.0042] | [0.9959, 1.0045] | [0.9959, 1.0045] | 3.46e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 512x768x768 | 16 | [0.9778, 1.0075] | [0.9766, 1.0088] | [0.9765, 1.0088] | 1.30e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 64x64x64 | 16 | [0.9963, 1.0209] | [0.9952, 1.0220] | [0.9952, 1.0220] | 1.08e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 128x8192x768 | 16 | [0.9960, 1.0048] | [0.9956, 1.0052] | [0.9956, 1.0052] | 3.85e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 16x128x32 | 16 | [0.9850, 1.0143] | [0.9837, 1.0156] | [0.9837, 1.0156] | 1.28e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 2048x768x2048 | 16 | [0.9785, 1.0360] | [0.9760, 1.0385] | [0.9760, 1.0385] | 2.51e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 512x2048x768 | 16 | [1.0025, 1.1805] | [0.9947, 1.1883] | [0.9947, 1.1883] | 7.79e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 512x768x768 | 16 | [0.9691, 1.1200] | [0.9626, 1.1266] | [0.9625, 1.1266] | 6.60e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_a_null | 64x64x64 | 16 | [0.9831, 1.0115] | [0.9819, 1.0127] | [0.9819, 1.0127] | 1.24e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 128x8192x768 | 24 | [0.9479, 0.9868] | [0.9469, 0.9878] | [0.9469, 0.9878] | 1.08e-03 | REGRESSION (CI excludes 1.0) |
| abba_a_vs_final | 16x128x32 | 24 | [0.9950, 1.0162] | [0.9944, 1.0168] | [0.9944, 1.0167] | 5.86e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 2048x768x2048 | 24 | [0.9967, 1.0042] | [0.9965, 1.0044] | [0.9965, 1.0044] | 2.08e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 512x2048x768 | 24 | [0.9839, 1.0097] | [0.9832, 1.0105] | [0.9832, 1.0105] | 7.16e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 512x768x768 | 24 | [0.9327, 1.0524] | [0.9294, 1.0557] | [0.9294, 1.0557] | 3.32e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 64x64x64 | 24 | [0.9773, 1.0177] | [0.9762, 1.0188] | [0.9762, 1.0188] | 1.12e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 128x8192x768 | 24 | [0.9965, 1.0030] | [0.9964, 1.0032] | [0.9964, 1.0032] | 1.79e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 16x128x32 | 24 | [0.9916, 1.0190] | [0.9908, 1.0198] | [0.9908, 1.0198] | 7.61e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 2048x768x2048 | 24 | [0.9939, 1.0067] | [0.9935, 1.0071] | [0.9935, 1.0071] | 3.57e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 512x2048x768 | 24 | [0.9975, 1.0094] | [0.9972, 1.0097] | [0.9972, 1.0097] | 3.29e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 512x768x768 | 24 | [0.9955, 1.0259] | [0.9946, 1.0267] | [0.9946, 1.0267] | 8.42e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 64x64x64 | 24 | [0.9867, 1.0182] | [0.9858, 1.0190] | [0.9858, 1.0190] | 8.73e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 128x8192x768 | 24 | [0.9958, 1.0054] | [0.9955, 1.0056] | [0.9955, 1.0056] | 2.66e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 16x128x32 | 24 | [0.9911, 1.0253] | [0.9901, 1.0262] | [0.9901, 1.0262] | 9.48e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 2048x768x2048 | 24 | [0.9810, 1.0093] | [0.9802, 1.0101] | [0.9802, 1.0101] | 7.84e-04 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 512x2048x768 | 24 | [0.9650, 1.0794] | [0.9619, 1.0826] | [0.9619, 1.0826] | 3.17e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 512x768x768 | 24 | [0.9786, 1.0429] | [0.9768, 1.0446] | [0.9768, 1.0446] | 1.78e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_a_vs_final | 64x64x64 | 24 | [0.9860, 1.0227] | [0.9850, 1.0237] | [0.9850, 1.0237] | 1.02e-03 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 128x8192x768 | 12 | [0.9815, 1.0483] | [0.9815, 1.0483] | [0.9815, 1.0483] | 2.25e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 16x128x32 | 12 | [0.9881, 1.0066] | [0.9881, 1.0066] | [0.9881, 1.0066] | 6.24e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 2048x768x2048 | 12 | [0.9881, 0.9975] | [0.9881, 0.9975] | [0.9881, 0.9975] | 3.20e-08 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 512x2048x768 | 12 | [1.0055, 1.0467] | [1.0055, 1.0467] | [1.0055, 1.0467] | 1.39e-07 | SPEEDUP (CI excludes 1.0) |
| abba_full_main_vs_final | 512x768x768 | 12 | [0.8031, 1.1424] | [0.8031, 1.1424] | [0.8031, 1.1424] | 1.14e-06 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 64x64x64 | 12 | [0.9776, 1.0261] | [0.9776, 1.0261] | [0.9776, 1.0261] | 1.63e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 128x8192x768 | 12 | [0.9858, 0.9989] | [0.9858, 0.9989] | [0.9858, 0.9989] | 4.44e-08 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 16x128x32 | 12 | [0.9850, 1.0088] | [0.9850, 1.0088] | [0.9850, 1.0088] | 8.05e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 2048x768x2048 | 12 | [0.9906, 1.0047] | [0.9906, 1.0047] | [0.9906, 1.0047] | 4.75e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 512x2048x768 | 12 | [1.0012, 1.0121] | [1.0012, 1.0121] | [1.0012, 1.0121] | 3.71e-08 | SPEEDUP (CI excludes 1.0) |
| abba_full_main_vs_final | 512x768x768 | 12 | [0.9854, 1.0509] | [0.9854, 1.0509] | [0.9854, 1.0509] | 2.21e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 64x64x64 | 12 | [0.9787, 1.0220] | [0.9787, 1.0220] | [0.9787, 1.0220] | 1.46e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_full_main_vs_final | 128x8192x768 | 12 | [0.1316, 0.1326] | [0.1316, 0.1326] | [0.1316, 0.1326] | 3.31e-09 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 16x128x32 | 12 | [0.5922, 0.6100] | [0.5922, 0.6100] | [0.5922, 0.6100] | 5.99e-08 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 2048x768x2048 | 12 | [0.0940, 0.0950] | [0.0940, 0.0950] | [0.0940, 0.0950] | 3.33e-09 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 512x2048x768 | 12 | [0.1169, 0.1359] | [0.1169, 0.1359] | [0.1169, 0.1359] | 6.40e-08 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 512x768x768 | 12 | [0.1091, 0.1514] | [0.1091, 0.1514] | [0.1091, 0.1514] | 1.43e-07 | REGRESSION (CI excludes 1.0) |
| abba_full_main_vs_final | 64x64x64 | 12 | [0.6046, 0.6166] | [0.6046, 0.6166] | [0.6046, 0.6166] | 4.06e-08 | REGRESSION (CI excludes 1.0) |
| abba_main_vs_final | 128x8192x768 | 12 | [0.9540, 1.0338] | [0.9540, 1.0338] | [0.9540, 1.0338] | 2.69e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 16x128x32 | 12 | [0.9875, 1.0038] | [0.9875, 1.0038] | [0.9875, 1.0038] | 5.47e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 2048x768x2048 | 12 | [0.9940, 1.0038] | [0.9940, 1.0038] | [0.9940, 1.0038] | 3.33e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 512x2048x768 | 12 | [0.9586, 1.0111] | [0.9586, 1.0111] | [0.9586, 1.0111] | 1.77e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 512x768x768 | 12 | [0.8564, 1.2133] | [0.8564, 1.2133] | [0.8564, 1.2133] | 1.20e-06 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 64x64x64 | 12 | [0.9882, 1.0263] | [0.9882, 1.0263] | [0.9882, 1.0263] | 1.28e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 128x8192x768 | 12 | [1.0018, 1.0117] | [1.0018, 1.0117] | [1.0018, 1.0117] | 3.33e-08 | SPEEDUP (CI excludes 1.0) |
| abba_main_vs_final | 16x128x32 | 12 | [0.9886, 1.0129] | [0.9886, 1.0129] | [0.9886, 1.0129] | 8.21e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 2048x768x2048 | 12 | [0.9866, 1.0103] | [0.9866, 1.0103] | [0.9866, 1.0103] | 7.99e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 512x2048x768 | 12 | [0.9963, 1.0039] | [0.9963, 1.0039] | [0.9963, 1.0039] | 2.56e-08 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 512x768x768 | 12 | [0.9712, 1.0266] | [0.9712, 1.0266] | [0.9712, 1.0266] | 1.87e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 64x64x64 | 12 | [0.9995, 1.0375] | [0.9995, 1.0375] | [0.9995, 1.0375] | 1.28e-07 | INCONCLUSIVE (CI includes 1.0) |
| abba_main_vs_final | 128x8192x768 | 12 | [1.3820, 1.3969] | [1.3820, 1.3969] | [1.3820, 1.3969] | 5.01e-08 | SPEEDUP (CI excludes 1.0) |
| abba_main_vs_final | 16x128x32 | 12 | [1.0252, 1.0481] | [1.0252, 1.0481] | [1.0252, 1.0481] | 7.71e-08 | SPEEDUP (CI excludes 1.0) |
| abba_main_vs_final | 2048x768x2048 | 12 | [1.5745, 1.6734] | [1.5745, 1.6734] | [1.5745, 1.6734] | 3.33e-07 | SPEEDUP (CI excludes 1.0) |
| abba_main_vs_final | 512x2048x768 | 12 | [1.4537, 1.8121] | [1.4537, 1.8121] | [1.4537, 1.8121] | 1.21e-06 | SPEEDUP (CI excludes 1.0) |
| abba_main_vs_final | 512x768x768 | 12 | [1.3656, 1.5468] | [1.3656, 1.5468] | [1.3656, 1.5468] | 6.11e-07 | SPEEDUP (CI excludes 1.0) |
| abba_main_vs_final | 64x64x64 | 12 | [1.1100, 1.1340] | [1.1100, 1.1340] | [1.1100, 1.1340] | 8.10e-08 | SPEEDUP (CI excludes 1.0) |

## Did any conclusion change?

* abba_a_vs_a_null / 512x2048x768: SPEEDUP (CI excludes 1.0) → INCONCLUSIVE (CI includes 1.0)

## Did the non-inferiority gate change?

* abba_a_vs_a_null / 512x768x768: True → False

## Where the corrected numbers are used

* every new performance decision, including the Phase LXVI gate;
* `check_phase_lxv.py` verifies the archived receipt against this record,
  so the historical receipt stays byte-identical while the live comparison
  uses the corrected interval.

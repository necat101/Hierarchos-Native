# Phase XXII packed-route recovery

`packed-first-sweep-540c79b` preserves every byte of the first current GPU
sweep. All thirteen benchmarks and route guards ran; three performance guards
failed (channel mix, FP16-storage packed forward, FP32 backward fusion).
Numerical parity passed. The failed status remains a failure.

`decision-tables-before` preserves the mutable runtime tables before deliberate
re-recording through the maintained drivers. The old root-level benchmark
receipts are left unchanged and their historical contract tests read these
paired tables. Current qualification instead requires every canonical receipt
under `20261112-unified-release/phase19z/packed-cell` and the current runtime
tables, through `check_measured_producer_routes.py`.

Re-recording retains each producer's original batches, samples, warmups and
decision margin. A rebuild and an independent complete sweep are required;
recording a policy alone does not qualify it.

`packed-second-sweep-3f7ac75` retains the complete post-rebuild sweep. Ten
benchmark/guard pairs passed. Three timing guards failed: time-mix projection
at batch 4 (staged ahead 3.7%), channel mix at batch 8 (staged ahead 3.4%), and
FP16-storage rematerialization at batch 1 (packed ahead 3.9%). The original 2%
band remains unchanged, and the status is FAIL. The numerical requirements
passed; the existing time-mix geometry refusals remain explicit.

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

The current packed timing protocol uses 120 timed observations per candidate.
This is a complete number of the existing three-leg and four-leg rotations,
with more observations than the earlier 20/40-sample runs. Candidate graphs,
batch sweeps, warmups, eight-step chains, 50 key/state repetitions and the
original 1%/2% decision bands remain unchanged. The CPU guard now rejects a
receipt whose sample count differs from its maintained recipe. The earlier
short measurements stay archived with their actual FAIL statuses.

`packed-third-sweep-0fc7897-balanced120` preserves the first complete 120-sample
sweep, still FAIL (10/13). All numerical comparisons passed. The performance
guards rejected channel mix, dense low-rank and FP16-storage packed forward at
batch 32. Their policies will be recorded with the same 120-sample protocol,
then independently replayed after rebuilding affected executables.

The chain recording driver supports a candidate output table so all three
calibrations can run before changing the shipped tables. Each recording pass
still uses the maintained build and measurement path. Candidate recording is
not qualification. The pre-recording binary/table literal audit records which
executables embed the three mutable tables; unaffected numerical evidence can
be reused only if its binary and source bindings still verify.

`packed-fourth-sweep-259f63a-balanced120` preserves the second 120-observation
sweep. Every numerical pair passes, but three performance guards fail (channel
mix, dense low-rank and FP32 packed forward), so status remains FAIL. The final
binary lock also caught Cargo relinking the key/state consumer during the sweep.
It did not embed the three changed tables, but library metadata still invalidated
its build artifact. The producer now builds all nine packed release executables
before capturing its binary lock; the lock itself remains mandatory.

Independent 120-observation channel ratios at batch 16 were 1.01832, 1.05522 and
0.97458 (composed/fused). Dense low-rank batch 16 ratios were 0.98671, 0.98222 and
0.97272. The original 2% boundary is crossed by measurement variability. The
next protocol uses 1,200 observations (100 complete cycles of both rotations)
for all eleven timed producers. No numerical limit, decision band, batch,
warmup, candidate graph, chain length or key/state repetition count changes.
Leaf routes are recorded/applied first; forward, rematerialization and backward
chains are recorded/applied in order, with real rebuilds between stages. A full
independent thirteen-pair replay after the final rebuild is still required.

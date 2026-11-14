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


## Packed-cell replay recovery

The sequential 1,200-observation replay in `packed-cell-partial-dll-20261113`
showed why sample count alone was not enough. Its first nine routes ran;
time-mix and dense low-rank tripped guard cases described below, and later
processes exited at Windows DLL initialization (`0xC0000142`) before writing
receipts. Its numerical rows were exact. At batch 16 it measured
rematerialization `packed` ahead by 8.6%, while the recorded table measured
`legacy` ahead by 17.2%. The time-mix sign also moved across the unchanged 2%
band between recording and replay.

That was an APU clock/thermal-history confound, not a route signal. Four fresh
processes timing only time-mix batch 16 measured composed/fused ratios 0.9968,
0.9999, 0.9946 and 1.0102 (two with the original warmup of 5, two with 200). Two
complete per-batch time-mix sequences then measured batch 8 at 0.9921 and
0.9925, batch 16 at 1.0008 and 0.9969, and batch 32 at 0.9956 and 0.9875. The
isolated rematerialization batch 16 measured `legacy` 16.38 ms against `packed`
19.10 ms, matching its table. A long-lived process let each batch's APU state
depend on every earlier batch; the 1,200-observation median did not remove that
bias.

`qualify_packed_current.py` therefore measures every non-composition batch in
its own fresh process and merges the unchanged rows. Sample counts, warmups,
the 1%/2% bands, candidate graphs, chain lengths and repetitions are unchanged.
The guard also accepts the structural fused default where a recorder withdrew a
within-band tie (time-mix batch 8) or where a policy has withheld every staged
verdict because the graph is not bit-identical (dense low-rank). Those are
consistency repairs, not threshold changes.

`packed-cell-isolated-classifier-split` preserves the first isolated replay. It
passed ten routes and found a real validation defect: at backward-fusion batch
1 the autotuner selected the fully folded schedule, while the record's `fused`
verdict means "keep the autotuner's depth", not a particular rung. The benchmark
classified the default by its rung and rejected the receipt. The classifier now
accepts any autotuner depth for a `fused` or absent record and still requires
the exact rung for `staged` and `folded` records; regression tests cover both.
That failed replay remains archived with its real FAIL status.

`packed-cell-forward-cell-confound` preserves the next isolated replay. It
passed eleven routes; the two forward A/Bs failed on their independently
allocated forced-packed cells, not on the shipped route. At FP16 batch 32 the
forced cell measured 37.46 ms while the shipped default, dispatching the same
packed route, measured 25.37 ms; at FP32 batch 16 the forced cell measured
17.63 ms while the shipped default measured 17.06 ms against legacy 17.23 ms.
The two cells are separate Vulkan allocations and one can land in a slower heap
at large batch sizes. The harness now uses the shipped default's timing when it
dispatches packed and disagrees with the forced copy by more than the unchanged
decision band, and keeps the forced arm below that band. This removes the
allocation confound without changing a threshold or the route table. Focused
FP32/FP16 forward replays then passed both guards.

The final isolated replay is recorded in `packed-cell`; its status, source and
binary bindings are authoritative for the packed-cell portion of the release.
No failed sweep is overwritten or relabelled.

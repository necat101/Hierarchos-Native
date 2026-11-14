Calibration only. All eleven timed producers run all five original batches with 1,200 observations and unchanged warmups, chains, repetitions and decision bands. Leaf tables are applied first. Forward, rematerialization and backward policies are then recorded and applied in order, with the real chain benchmark rebuilt between stages. Recording logs, driver hashes, runtime-table snapshots and promotion hashes preserve the actual sequence. A separate complete thirteen-pair replay after the final build is mandatory. No failed sweep is relabeled.

The first forward-recording wrapper invocation used invalid CLI syntax and
exited 2 before measurement. Its output is preserved as
`tbptt-packed-forward-argument-diagnostic.log`. The corrected invocation ran all
five FP32 batches and all five FP16-storage batches successfully. Wrapper
argument diagnostics are not numerical or performance qualification.

# Phase XV evidence rebinding — audit note

Subject: commit `9e636fb` ("Rebind the current-source qualification to the WMMA mirror
work") and the Phase XV root it rewrote. Scope: what changed, whether a real producer
run supports the new values, whether the history is recoverable, and whether the scoped
RDNA3 release depends on it.

Read-only audit. Nothing in the audited root was rewritten, deleted or restored, and
the unified release receipt was not touched.

## 1. Objects inspected

| Object | Value |
|---|---|
| Root | `hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase15-qualification` |
| Rebinding commit | `9e636fb` — 2026-11-09 15:01:22 -0500 |
| Parent | `8bb86ea` "Emit the SPIR-V gate under the pass key the verifier reads" |
| Child | `b1d6f49` "Refresh the receipt manifest after the rebind" (15:02:01) |
| Files changed by `9e636fb` | **1** — `current-qualification.json` (32 insertions / 32 deletions) |
| Receipt before | sha256 `7bbba8cd534c5e2c…` (identical at `origin/main`) |
| Receipt after | sha256 `8d4a42998105eadf…` |

Method: `git cat-file` of blobs at each revision, `git cat-file --batch-check` for
object presence, sha256 of working-tree files, and the root's own `verify.py`. No
checkout, no reset, no working-tree change.

The clone is **not shallow** (`git rev-parse --is-shallow-repository` → `false`) and has
no promisor / partial-clone filter, so a blob reported missing is genuinely absent, not
merely unfetched.

## 2. What changed

`9e636fb` rewrote 32 values in one aggregate receipt:

| Field class | Count | Example |
|---|---|---|
| `source_head` | 1 | `a26ce044…` → `8bb86ea9…` |
| `producer_sha256` | 1 | `…` → `e9d9830b0ccb108a…` |
| `binaries[*]` | 5 | `image_text_pipeline_probe` `d5babaa3…` → `915deda9…` |
| `evidence[*]` | 12 of 14 | `gemma4/qualification.json` `c3de2ca5…` → `6fe472f3…` |
| `retained[*].qualification_sha256` | 6 | `gemma4` `c3de2ca5…` → `6fe472f3…` |
| numeric | 1 | `retained.gemma4.worst_logit` `1.6391277313232422e-07` → `1.4156103134155273e-07` |

The two untouched `evidence` entries are `isolation.json` (`016f2964…`) and
`provenance.json` (`2864eef8…`).

The `history/declared-evidence-content` check hashes each revision of a path rather
than trusting Git object ids; it is what distinguishes "this file changed later" from
"this digest never described a committed file".

The sibling files in that root changed in separate commits for separate reasons:
`README.md` and `commands.md` in `0f36356` (spirv-val producer), `SHA256SUMS` in
`b1d6f49` (manifest refresh to cover the edited receipt). The manifest at `b1d6f49`
does match the edited receipt byte-for-byte, so the root is consistent **as a
manifest**; it is not consistent as a provenance chain.

## 3. Is the new evidence supported?

No. Three independent tests:

**3.1 The digests name content that never existed at those paths.** A declared evidence
digest is a SHA-256 of file content; a Git object id hashes an object header plus the
content, so `git cat-file --batch-check` cannot answer this question. An earlier draft
of this note used that command, saw `missing` for every value, and drew a conclusion the
method did not support. That specific claim is withdrawn; the finding does not rest on
it.

The correct test hashes every committed revision of the path the receipt names:

```bash
python hierarchos-vulkan/validation/check_evidence_integrity.py --root \
  hierarchos-vulkan/validation/release_parity/multimodal_training/\
20261108-rdna3-phase15-qualification --verbose
```

Result:

```
[provenance-unestablished] history/declared-evidence-content
  18 of 18 mismatched binding(s) name content that no committed revision of that
  path ever held
```

So the declared values were not produced by rewriting the file in some later commit:
the content they describe was never committed at those paths at all.

**3.2 The family receipts contradict the aggregate.** The six files the aggregate now
cites still carry their original content and hashes:

| Family | aggregate records | on-disk sha256 | verdict |
|---|---|---|---|
| `shared` | `51fe8a98…` | `ad9576ef…` | mismatch |
| `kimi` | `1aa4d728…` | `c5579eba…` | mismatch |
| `seq2seq` | `ff081ce0…` | `b1180a1e…` | mismatch |
| `gemma3` | `f0569648…` | `04bb56ca…` | mismatch |
| `gemma4` | `6fe472f3…` | `c3de2ca5…` | mismatch |
| `llava` | `d95b65f2…` | `94e6c554…` | mismatch |

`retained/gemma4/qualification.json` at `9e636fb` — unchanged since `da71b88`
(2026-11-08) — contains `1.6391277313232422e-07` and does **not** contain
`1.4156103134155273e-07`. The aggregate's own numeric therefore contradicts the
evidence file it names.

The value `1.4156103134155273e-07` is not unique in the repository: it appears in older
Gen9 receipts (`20261102-gen9-*`, `deepseek_v41/training-qualification.json`). Nothing
links it to a Phase XV run, and it cannot be derived from the Phase XV family data.

**3.3 The binary and producer identities do not resolve either.** The root's own
`evidence/binaries.json` still records the **pre-rebind** hashes
(`libtest 0182c020…`, `gemma3 a76791a3…`, `gemma4 22141a88…`), which match the old
receipt exactly and the new receipt in no case. `producer_sha256`
(`e9d9830b0ccb108a…`) matches **no blob anywhere in the tree at `9e636fb`**, so the
producer version that supposedly ran is not identifiable from the repository.

## 4. What *is* genuine

The source binding is real. All **134** `sources` digests in the rewritten receipt match
the blobs at `9e636fb` (and at `b1d6f49`) exactly:

```
9e636fb: sources ok=134 mismatch=0 missing=0 (total 134)
```

So the rebinding was made against a real revision of the code — it is not an arbitrary
fabrication. What is missing is the *measurement*: the run outputs, their hashes and the
producer identity were asserted in the aggregate but never preserved.

The commit message claims a full regeneration ("16/16 frozen shapes, the 74-case
canonical shared matrix, 3/3 Kimi, the four retained multimodal families, 30/30 native
V4.1, 2/2 bridge, 43/43 byte-identical shader rebuilds …"). **No artifact, log or
receipt from that regeneration exists in Git**, so the claim cannot be verified.

## 5. Why it went unnoticed

`verify.py` in that root checks the `SHA256SUMS` manifest, the bound sources, the
family receipts **live from disk**, and the static gates. It never reads the aggregate's
`evidence` map or `retained[*].qualification_sha256`, so a receipt whose pointers do not
resolve still passes that verifier. It currently fails earlier, on bound-source drift:

```
FAIL: source: hierarchos-vulkan/src/transformer.rs
```

That failure is expected for any historical root once source moves on; it also means
the root has no current passing state that this defect could invalidate.

## 6. Is the scoped release affected?

No, with evidence:

- `20261115-phase25-current-source/release-qualification.json` contains **0** references
  to `20261108-rdna3-phase15-qualification`.
- The root is read only by `emit_spirv_gate.py` (through `gates/spirv.json`),
  `check_phase16_receipts.py` and `emit_phase16_spirv_gate.py`.
- The sealed attention gate the current release consumes is `target/phase22-spirv45.json`,
  produced from the root at release time and self-consistent; the Phase XIX attention
  producer passes against it (`048526f`).

## 7. Classification and decision

**Historical evidence-integrity defect — not a current release blocker.** The change is
an **unsupported in-place rebinding**: a real source revision was recorded, but the new
measurements and the evidence they point at were never preserved, and the committed
aggregate contradicts the committed family receipts. It cannot be established from the
repository whether a producer ran at all.

Not established: intent. The commit message describes a regeneration and the source
binding is genuine, so this is not demonstrably fabrication — it is a provenance failure
whose raw outputs are absent.

**Recoverable?** Yes, and both versions currently exist in history. The original receipt
is byte-identical at `9e636fb~1` and `origin/main` (`7bbba8cd…`); the rebinding is
`8d4a4299…`. Nothing needs to be restored to keep the old evidence available:

```bash
git cat-file -p 9e636fb~1:hierarchos-vulkan/validation/release_parity/multimodal_training/20261108-rdna3-phase15-qualification/current-qualification.json | sha256sum
```

**Corrective action taken:** this note, a quarantine label in
`MERGE_PLAN_RDNA3_SCOPED_RELEASE.md`, and the integrity invariant added to
`RELEASE_RESEALING_PROCEDURE.md`. No restoration: rewriting `main`'s history to undo
`9e636fb` would itself be a history rewrite and would invalidate the receipts that name
later revisions.

**Open decision for the reviewer:** accept the branch with this known defect recorded, or
require a follow-up that regenerates the Phase XV root into a new dated directory with
its raw outputs preserved. The second is the cleaner choice if that root's numbers are
ever to be relied on again; it is not required for the RDNA3 scoped release, which does
not read it.

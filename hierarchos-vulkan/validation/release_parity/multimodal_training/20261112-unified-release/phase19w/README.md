# Current Phase XIX-W numerical evidence

These are raw, newly executed qualification JSON files and logs from the
release executables rebuilt after the packed-route policy update at `6efc631`.
The JSON bytes were copied unchanged from their execution directories.

* Forward: 166 resident/streaming comparisons across 83 cases and 43 profiles.
* Generation: 72 requests, covering full-prefix/cache and resident/streaming.
* PEFT: 32 independent CPU/PEFT profiles, all passing the original strict gate.

The matrix accounts for 146 canonical registry entries and 83 aliases; those
counts do not mean 229 independently tested models. Untested capabilities stay
explicit in the matrix. The original CPU oracle environments remain distinct:
the forward/generation fixtures use their frozen AVX2 references, while PEFT
uses Torch 2.6 with its pinned Transformers checkout and AVX-512 arithmetic.

The matrix binds exact producer, registry, source, executable and evidence
hashes. Twenty-four compatibility controls passed, including the positive
receipt before rejection mutations. The preceding matrix is preserved without
byte changes under `20261113-phase22-recovery/phase19w-a055f3630b5b`.

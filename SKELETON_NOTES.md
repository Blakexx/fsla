# consumer_kernel.cuh skeleton -- study notes

Branch `learn/consumer-skeleton`, based on `a7cb0a7` (tip of `perf/lsu`).
`git diff a7cb0a7` touches exactly one file:
`csrc/rola/src/consumer_kernel.cuh` -- everything else in the tree is
byte-identical to that commit. Only the body of `consumer_forward_kernel<BC,
DV, D, DECAY, GL, GC>` was removed; its signature, `__launch_bounds__`, every
`#include`, `atom_row_base`, and every other helper/constant in the file are
untouched.

## Stage-order map

The gutted body is organized as SETUP (S1-S6, once per CTA: resolve the work
item, decode the owner's leaf-address rectangle, look up decay rates, load the
entry state exactly once, scan for member tokens and group them into ordered
chunks, gather their raw amplitude/value data) followed by the WALK (V1-V12,
looped once per chunk of up to `kGroupTokens` tokens: build this chunk's decay
survival factors, reduce the leaf-invariant row scales, build the read and
write routing panels, prep the scaled value operand, run the readout GEMM
against the state frozen at chunk entry, run the small Gram between this
chunk's own write and read tokens, fold this chunk's writes into the running
state, add the Gram-carried within-chunk causal correction to the readout,
emit the finished numerator/denominator, optionally track summary-pass
bookkeeping, then carry the folded state forward as the next chunk's entry)
and finally EXIT (E1-E3: drain any outstanding gathers, publish the
segment-stash summary if this was a summary pass, or write the final state
back to HBM once if this CTA covered the owner's last segment). The key
correctness invariant threading all of it: within one chunk the state is
treated as frozen at its chunk-entry value for that chunk's own readout GEMM;
only the Gram/intra term supplies within-chunk causality, and the fold
advances the state only after that chunk's readout has already used the old
value.

## Ground truth

- **fp64 oracle**: `rola/ops/naive.py::naive_rola` -- the textbook gated
  linear-attention recurrence this kernel is a routed, tiled, GPU-resident
  reduction of. It is the ONLY validation target; do not write or compare
  against a second freshly-written reference (a shared-router bug once passed
  exactly that way).
- **Conformance/oracle-diff gate** (Tier 1, `tests/tier1_semantics/`):
  ```
  pytest tests/tier1_semantics/test_consumer_vs_oracle.py -k <case>
  ```
  Run scoped to a specific test/parametrization, never the unscoped
  `tests/` battery. It hands the same routing tensors to `naive_rola` and to
  the compiled kernel and compares at the bf16 family bound
  (`BF16_RTOL = 1e-2`).
- **Dual-run protocol** (for changes to the oracle itself, not needed while
  just filling in the kernel body): `tools/oracle_dual_run.py`, e.g.
  ```
  python tools/oracle_dual_run.py --surface oracle --against HEAD~1
  ```

## Mirrored docs

`docs/internals/consumer_kernel.md` is the prose walkthrough this file's
header claims as authoritative, along with `docs/internals/consumer.md`
(host dispatch), `docs/internals/consumer_launch.md`, `docs/internals/
smem_layout.md`, `docs/internals/mma_tile.md`, `docs/internals/
readout_mma.md`, and `docs/internals/intra_panel.md`.

**Caveat**: `consumer_kernel.md`'s own "Orientation" section (its §1) still
describes an OLDER shape of this kernel -- SMEM-resident state as two bf16
planes plus a separate mass column, and a "window of `W` groups" walked per
visit with a block-triangular Gram. The kernel body actually removed here
(see the real diff) is the newer "P26 endgame executor" form described in
this file's own top-of-file comment (lines 1-72, which were NOT removed): the
state lives in REGISTERS as the fold's own accumulator fragments, and each
visit handles exactly ONE group of `kGroupTokens` tokens, not a window of
several. Trust the top-of-file comment and the real diff over the external
doc's §1 when they disagree; the rest of the doc's part-by-part sections
(entry state, producer/scheduler, staging, the four GEMMs, the epilogue) are
still a reasonable per-topic reference even where the exact register-vs-SMEM
residency has moved on.

## Seeing the real implementation of any stage

```
git diff a7cb0a7 -- csrc/rola/src/consumer_kernel.cuh
```

Everything after the `-` lines is the real body, in the same order the
stage-order map above follows it. To view it as a plain file instead of a
diff:
```
git show a7cb0a7:csrc/rola/src/consumer_kernel.cuh
```

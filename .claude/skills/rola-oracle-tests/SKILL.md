---
name: rola-oracle-tests
description: Adding or running an fp64-oracle cell for the carry kernel — Cell/alt_cell fixtures, GPU-locked pytest, standing failures, re-deriving expectations from the law
---
The fp64 oracle is the truth (KERNEL_STANDARDS §10, "Oracle + conformance diff are the crown
jewels"). Read `tests/k31/test_carry_vs_oracle.py`'s own module docstring — it states the gate is
the fp64 oracle, not the shipped kernel, because there is no bit-identity target across a redesign,
only the arithmetic.

## Adding a cell (`tests/k31/fixtures.py`)
Read `Cell`'s signature and `ref_levels()` directly for the current parameter set and what it draws.
The key design points to preserve when adding a new cell:
- `ref_levels()` must return the fp64 doubles of the ROUNDED (bf16-cast) operands, never the raw
  fp64 draws — the oracle must reference what the kernel actually received, not what was cast away.
- `k_tok` (nonzeros per row) is a property of the DRAW, never told to the kernel — that separation
  is the design's whole claim.
- `alt_cell`-style declarations draw a dense pass first (to consume the RNG stream identically),
  then redraw the real support off a FRESH same-seeded generator — mirror this exactly if you add a
  new alternating-declaration cell; getting the RNG-stream order wrong silently changes what a
  "seed" cell means.
- `cohort` (clustered rather than uniform-random sparsity) uses a different draw helper than the
  uniform case — check `fixtures.py` for which.
- Read `alt_cell`'s own ruling comment for which levels are read-sparse vs. write-sparse; don't
  assume a parity rule without checking it, since it's a design choice recorded there, not a law.

## Running under the GPU lock
GPU work is serialized under `flock /tmp/rola_gpu.lock` — every campaign, not just this suite (see
`development/operations/OPERATIONS.md`). Run a NAMED file, never the bare directory:
```bash
flock /tmp/rola_gpu.lock env TMP=/tmp TEMP=/tmp TMPDIR=/tmp CUDA_HOME=/usr/local/cuda \
  PYTHONPATH=. TORCH_EXTENSIONS_DIR=/tmp/<task>-ext <venv>/bin/python -m pytest \
  <exact test file(s)> -x -q
```
`tests/test-gate-mechanical-rules`: the global hook DENIES an unscoped `pytest tests/` — VRAM
freezes have come from exactly that. Briefs must name the exact file(s); never widen a gate command
past what the stage touched. Add a `--timeout` (pytest-timeout or a shell `timeout N`) around any
GPU run you don't already trust — a hang here blocks the whole flock queue for every other agent.

## Requesting an arm — `require_arm`, never a bare dispatch call
Read `require_arm`'s docstring/signature for its current parameters. It SKIPS (and names the exact
missing arm) when the binary doesn't carry it — a default build compiles only the SHIPPED rows
(`gen_shards.CARRY_SHIPPED_ARMS`); TEST rows need `ROLA_CARRY_ARMS=all` to exist at all. An
UNDECLARED arm (no row in the list, in ANY configuration) is NOT a skip — it falls through to the
dispatch's own "unbuilt carry arm" refusal, because turning a truly-absent arm into a skip would
report it as satisfied coverage.
Call `assert_fresh_binary()` (device-side stamp read off a SHIPPED arm) at the top of any new test
module touching carry — a stale binary cannot be caught by path or hash checks (recurring trap,
`extension-trap-device-side-check`).

## The standing-failure list — where it lives
There is no single standing-failure file; a standing (pre-existing, not-your-regression) failure is
recorded IN the FINDINGS entry that surfaced it, with the reproduction proof (see e.g. `## K36b` for
the shape of such an entry). Before treating any red test as "known", REPRODUCE it on a clean
checkout of the parent tip with the identical command — a claim of "standing failure" that skips
this step is not distinguishable from hiding a real regression.

## The stale-expectation antipattern
Never hand-encode an expected number and then patch it when a test goes red. RE-DERIVE the expected
value from the LAW (the fold/decay/routing math) each time a mechanism changes, and cross-check the
kernel's SEAM output against the CENSUS (`rola.ops.carry.arms()`) — a seam that silently drifted
from the census row it claims to serve is the shared-router-bug shape (`matching-the-naive
antipattern`): validating a variant only against itself, never against the one canonical ground
truth (the fp64 reference in `tests/k31/reference.py`).

---
name: rola-oracle-tests
description: Adding or running an fp64-oracle cell for the carry kernel — registry cells, GPU-locked pytest, standing failures, re-deriving expectations from the law
---
The fp64 oracle is the truth (KERNEL_STANDARDS §10, "Oracle + conformance diff are the crown
jewels"). Read `tests/oracle/test_carry_vs_oracle.py`'s own module docstring — it states the gate is
the fp64 oracle, not the shipped kernel, because there is no bit-identity target across a redesign,
only the arithmetic.

## Adding a cell (`benchmarks/cells/carry_cells.json`)
A carry cell is a record in the registry, never a draw written in a test: the file's `fields` block defines each field
(`widths`, `dv`, `tokens`, `warps_per_cta`, `draw`, `k_tok`, `cohort`, `support`, `backing`, `state`, `tier`), and
`benchmarks/cells/__init__.py`'s `realize` draws it. The design points to preserve:
- The oracle reads the ROUNDED operands: `realize` casts the draw to bf16 and `RealizedCell.doubles()` returns their fp64
  values, so the reference sees what the kernel received, not what the cast threw away.
- `k_tok` (nonzeros per row) is a property of the DRAW, never told to the kernel -- that separation is the design's
  whole claim.
- The sparse side alternates by level (`realize`'s own ruling comment says which levels are read-sparse and which
  write-sparse); `cohort` draws clustered support through `clustered`, the uniform case through `simplex`.
- A cell at an arm the binary does not build asks `require_arm` first (the carry oracle's `run` does): a declared test
  row skips with its name, an undeclared arm runs to the launch surface's refusal and stays red.

## Running under the GPU lock
GPU work is serialized by the shared `gpu_lock()` primitive (`rola_devtools.locks.gpu`, K46) — every
campaign, not just this suite (see `development/operations/OPERATIONS.md`). pytest takes it itself
via a session-scoped fixture in `tests/conftest.py`; invoke BARE, never wrap the command in an
external `flock` on the same lock path (self-deadlocks — K38). Run a NAMED file, never the bare
directory:
```bash
env TMP=/tmp TEMP=/tmp TMPDIR=/tmp CUDA_HOME=/usr/local/cuda \
  PYTHONPATH=. TORCH_EXTENSIONS_DIR=/tmp/<task>-ext <venv>/bin/python -m pytest \
  <exact test file(s)> -x -q
```
`tests/test-gate-mechanical-rules`: the global hook DENIES an unscoped `pytest tests/` — VRAM
freezes have come from exactly that. Briefs must name the exact file(s); never widen a gate command
past what the stage touched. Add a `--timeout` (pytest-timeout or a shell `timeout N`) around any
GPU run you don't already trust — a hang here blocks the whole lock queue for every other agent.

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
truth (the fp64 reference in `tests/oracle/reference.py`).

## The compute-sanitizer named gate (K39)
`tools/sanitize_oracle.py` runs `racecheck`/`synccheck`/`initcheck` (memcheck already ran, ad hoc,
on paging — `docs/testing.md`'s "sanitizer lane") against one named cell per family (carry, intra,
decode), each its own `compute-sanitizer`-wrapped pytest process. Invoke `sanitize_oracle.py` BARE
(K46): it takes `gpu_lock()` itself, once, for the whole run, and the reentrant marker lets the
wrapped pytest subprocess's own conftest fixture see the lock already held rather than deadlock on
it. Run it — it is a SCHEDULED lane, not a PR gate (the tools cost one to two orders of magnitude) —
whenever a change touches shared-memory publish/consume protocols or barrier placement (the S2d
ring's bug class), not on every commit.

## The hypothesis `cells` strategy (K39)
`tests/oracle/test_carry_hypothesis_cells.py` draws (widths, k_tok, seed, cohort, d_v) over the
DECLARED arm/`d_v` space via a `st.composite` strategy, opt-in under the `hypothesis_cells` marker
(deselected by default alongside `bench` — `pyproject.toml`'s `addopts`). A SUPPLEMENT to the fixed
named cells above, never a replacement — run it with `-m hypothesis_cells` when you want shrinking
to find a minimal failing cell automatically, not as part of the default battery.

- THE RUN RECIPE IS: plain `pytest <file>` from the worktree venv — the conftest takes the GPU lock itself
  (K46); wrapping it in `flock` SELF-DEADLOCKS (G2c lost 120 s to it). Any `flock … pytest` line in an
  older section of this skill is stale.

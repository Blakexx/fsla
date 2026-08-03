# RoLA — Routed Linear Attention

RoLA is a linear-attention layer whose recurrent state is a **set of addressable
slots**, and whose per-token routing decides which slots each token reads from and
writes to. The state is a fixed size chosen by the topology, not a cache that grows
with the sequence — so the memory footprint at token *T* is the same as at token 1.
This repository ships the CUDA consumer kernel that executes that routed
recurrence, the planner that chooses its runtime geometry, the layer, and the
evidence for all three.

Paper: *(link on publication)* · Citation: [below](#citation)

---

## What this repository ships that a kernel repository usually does not

**A ratified supported matrix, not a supported-arch aspiration.** Every
architecture this package will run on has a committed manifest of measured
per-instantiation register and spill counts, pinned to the exact assembler that
produced them. The build refuses to compile an architecture without one; the binary
refuses to launch on a device without one. There is **no PTX JIT fallback** — an
unmeasured architecture fails loudly rather than silently running codegen nobody
looked at. → [`tools/manifests/`](tools/manifests/), [`docs/bringup.md`](docs/bringup.md)

**A two-tier test taxonomy that separates two different questions.** *Tier 1
(inter-implementation)* gates the kernel against an fp64 oracle across the full
semantic matrix, at tolerances that are derived and documented rather than tuned
until green. *Tier 2 (intra-implementation)* gates the kernel against **itself**
across every runtime geometry the planner may choose — and every geometry pair
carries a declared contract class: engineered identities are tested as **byte
equality**, and legitimate reassociations are tested against the oracle plus a
**derived** error bound. A pair changing class is itself a test failure.
→ [`docs/testing.md`](docs/testing.md),
[`tests/tier2_geometry/matrix.py`](tests/tier2_geometry/matrix.py)

**Committed measured cells, with provenance.** The benchmark results in this
README are files in this repository, each carrying the machine, driver, toolchain,
and the SHA of the ratification manifest the binary was built from. The planner's
cost constants cite the cell they came from, and a gate checks that the planner's
chosen geometry is the argmin of the measured grid.
→ [`benchmarks/cells/`](benchmarks/cells/), [`docs/build.md`](docs/build.md)

---

## Installation

### Requirements

| | |
|---|---|
| Python | 3.10 – 3.13 |
| PyTorch | ≥ 2.6, built against CUDA 12.4 |
| CUDA toolkit | **12.4** — the toolchain the manifests are ratified under |
| GPU | a **ratified** compute capability (see the table below) |
| Build | `ninja`; source installs need `--no-build-isolation` |

`--no-build-isolation` is not optional and not a workaround: pip's build isolation
would fetch a CPU-only torch from PyPI and build the extension against the wrong
ABI. Flash-attention and mamba-ssm require the same thing for the same reason.

### Install modes

| Mode | Command | Notes |
|---|---|---|
| Wheel | `pip install rola` | Pre-built for the ratified matrix |
| Source | `pip install rola --no-build-isolation` | Builds the fatbin ahead of time; no JIT |
| Development | `pip install -e . --no-build-isolation` | Runs the R1/R3 gates and the post-build ratification |
| Extras | `pip install "rola[transformers]"` / `"rola[fla]"` | Optional interop only; the kernel needs neither |

The build is **ahead-of-time and closed-world**. It emits `code=sm_XX` only, never
`code=compute_XX`, so the fatbin carries no PTX for a driver to JIT; the shipped
`.so` is asserted to contain zero PTX images. Setting `ROLA_CUDA_ARCHS` to an
architecture with no ratified manifest fails the build before it compiles anything.

## Supported configurations

This table is **generated** by `tools/supported.py` from the kernel's arch table and
the committed manifests, and CI fails if it drifts. A hand-maintained support table
is a claim; a generated one is a report.

<!-- BEGIN GENERATED: tools/supported.py -->
| Compute capability | GPUs | Ratified under | Instantiations | Status |
|---|---|---|---|---|
| `sm_80` | A100, A30 | ptxas 12.4.99 | 512 | ratified |
| `sm_86` | A10, A40, RTX 30xx | ptxas 12.4.99 | 608 | ratified |
| `sm_87` | Jetson Orin | — | — | **not ratified — will refuse to load** |
| `sm_89` | L40S, RTX 40xx | — | — | **not ratified — will refuse to load** |
<!-- END GENERATED: tools/supported.py -->

The unratified rows are not an apology; they are the product statement. The first
claim above is worth nothing if this table quietly implied coverage that was never
measured. Bringing up a new architecture is a measurement, and
[`docs/bringup.md`](docs/bringup.md) is the procedure.

### Routing activations

Also generated, from the one registry every validator reads
(`rola/routing/activations.py`). `kerneled` is what decides dispatch: an
unratified activation routes to the differentiable reference path carrying an
explicit `not_kerneled` flag, and the kernel entry refuses it outright — a labelled
research surface, never a silent fallback.

<!-- BEGIN GENERATED: rola.routing.activations.capability_table -->
| activation | normalized | exact_zeros | support_differentiable | kerneled |
| --- | --- | --- | --- | --- |
| entmax(alpha=1.5) | yes | yes | no | yes |
| entmax(alpha=2) | yes | yes | no | yes |
| softmax | yes | no | yes | yes |
<!-- END GENERATED: rola.routing.activations.capability_table -->

## Usage

### The functional op

```python
import torch
from rola import LevelConfig, Topology, rola_func, select_plan

topology = Topology(levels=(LevelConfig(8, routing), LevelConfig(8, routing)), d_v=64)
plan = select_plan(topology, normalization="global")
y, state = rola_func(v, read_levels, write_levels, g_read, g_write, plan)
```

`rola_func` is the **single autograd boundary**. A call that requires gradients runs
the whole operation — forward and backward, from one graph — on the differentiable
reference, so a value and its gradient never come from two implementations that
merely agree to a tolerance.

### The layer

```python
from rola import RoLA

layer = RoLA(hidden_size=512, num_heads=8, head_v_dim=64,
             levels_per_head=2, states_per_level=[8, 8],
             state_norm="global", layer_idx=0).cuda()
y, _, state = layer(x, use_cache=True)
```

### The state

`rola.RoLAState` is a **fixed-size recurrent state, not a KV cache**: the per-layer
slot state, the short-convolution ring buffers, and a token tally. Every
`transformers` cache abstraction is built for the growing case, so RoLA owns its
own object and adapts at one point (`rola.state.as_layer_state`). `transformers`
interop lives behind the `[transformers]` extra.

### The planner

`select_plan` chooses the runtime geometry from a cost model whose constants are
traceable to committed cells. A `Plan` can also be built by hand and passed straight
to `rola_func` — the kernel's correctness does not depend on which geometry the
planner picked, and that is what Tier 2 exists to prove.

### The oracle

`rola.naive_rola` is the fp64 definition, and it **ships in the wheel**. A user who
suspects a numerical problem can run the definition against the kernel on their own
shapes, on their own hardware, with no repository checkout. That is what makes the
tolerance claims auditable rather than asserted.

## Performance

Benchmark cells are committed data, not scripts that might reproduce it:
[`benchmarks/cells/topology_manifest.json`](benchmarks/cells/topology_manifest.json)
carries each cell's realized statistics, and
[`benchmarks/cells/latency_baseline.json`](benchmarks/cells/latency_baseline.json)
carries per-arm latencies with normalized provenance (GPU, driver, toolchain,
manifest SHA — no host identifiers). The methodology — interleaved paired arms,
median of per-pair ratios, IQR reported alongside — is documented in
[`benchmarks/bench/pairing.py`](benchmarks/bench/pairing.py).

## Evidence

- [`docs/testing.md`](docs/testing.md) — the two tiers, the tolerance discipline, the
  geometry matrix and its three meta-tests.
- [`docs/ratification.md`](docs/ratification.md) — what a manifest is, what the
  closed-world rules refuse, and why the gate self-tests.
- [`docs/build.md`](docs/build.md) — the ahead-of-time build and the
  self-describing binary.
- [`docs/bringup.md`](docs/bringup.md) — bringing up an unratified architecture.
- [`docs/provenance.md`](docs/provenance.md) — where this code came from.
- [`docs/open-work.md`](docs/open-work.md) — what is knowingly unfinished.

## Tests

```bash
pytest tests/unit tests/tier1_semantics -m "not cuda"   # CPU: no device, no build
pytest tests/tier1_semantics -m cuda                    # TIER 1: kernel vs fp64 oracle
pytest tests/tier2_geometry                             # TIER 2: kernel vs itself
pytest tests/layers                                     # layer <-> op
python tools/ratify.py --arch 80 --arch 86              # ratification, no GPU needed
python tools/ratify.py --self-test                      # prove the gate can fail
pytest benchmarks/test_latency_regression.py -m bench   # latency vs the committed baseline
```

Never `pytest tests/` unscoped. The `cuda` marker exists so that `-m "not cuda"`
deselects at *collection* time, before torch initializes a CUDA context.

## Bringing up new hardware

Probe → manifest → gate, and the manifest change is always a reviewed diff, never a
CI auto-commit. See [`docs/bringup.md`](docs/bringup.md).

## Related repositories

| Repository | What lives there |
|---|---|
| `flash-linear-attention-rola` | The FLA fork: HF model classes, training entry points, and the full development history |
| `rola-zoology` | Synthetic capability evaluations (MQAR and friends) |
| `rola-bench` | Language-modelling benchmark suites and result data |
| `rola-paper` | The paper source |

## Changelog

Semantic changes are listed per minor version. **A loosened tolerance is a semantic
change** and requires a changelog entry, the derivation that justifies the new
number, and reviewer sign-off; tightening one requires none of that.

`0.1.0` — first extracted release. See [`docs/open-work.md`](docs/open-work.md) for
what is deliberately not in it.

## Citation

```bibtex
@misc{rola,
  title  = {Routed Linear Attention},
  author = {Bottum, Blake},
  year   = {2026},
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

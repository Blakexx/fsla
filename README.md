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
and the SHA of the ratification manifest the binary was built from. Regression
thresholds are derived from each cell's own measured spread, never an invented
percentage, and the planner's shared-memory ceiling is gated against the *built
binary's own* dispatch-table carve rather than a modelled one.
→ [`benchmarks/cells/`](benchmarks/cells/), [`docs/build.md`](docs/build.md)

**A documentation tree the code is checked against.** The engine's design reasons
live in `docs/internals/`, one document per source file, with one-line hazard
stubs left in the code at the sites they explain; the decisions that outlive any
particular kernel live in `docs/architecture/`. A test fails if a source file has
no mirror document or a stub cites an anchor that no longer exists, so the two
cannot drift apart.
→ [`docs/architecture/`](docs/architecture/README.md),
[`docs/internals/`](docs/internals/README.md)

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

The consumer's 512 kernel instantiations are compiled as **eight generated
translation units** ([`csrc/rola/src/instantiations/`](csrc/rola/src/instantiations/)),
partitioned on precision x normalization x tile width, so the matrix parallelizes
across `MAX_JOBS` instead of serializing inside one compiler process. They are
checked in — what is reviewed is what compiles — and `python tools/gen_shards.py
--check` is the gate that keeps them the files their generator produces. `MAX_JOBS`
defaults to a value derived from **available RAM**, which is the resource that
actually binds ([`docs/build.md`](docs/build.md)).

## Supported configurations

This table is **generated** by `tools/supported.py` from the kernel's arch table and
the committed manifests, and CI fails if it drifts. A hand-maintained support table
is a claim; a generated one is a report.

<!-- BEGIN GENERATED: tools/supported.py -->
| Compute capability | GPUs | Ratified under | Instantiations | Status |
|---|---|---|---|---|
| `sm_80` | A100, A30 | ptxas 12.4.99 | 144 | ratified |
| `sm_86` | A10, A40, RTX 30xx | ptxas 12.4.99 | 144 | ratified |
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
| opaque | yes | yes | no | yes |
| softmax | yes | no | yes | yes |
<!-- END GENERATED: rola.routing.activations.capability_table -->

## Usage

The API is the reduction theorem: linear attention's contract, plus ONE new concept —
the feature map. `docs/api.md` is the contract in full.

### The op

```python
import rola

producer = rola.SimplexProducer((8, 8), rola.sparse_routing(1.5),
                                hidden_size=512, num_heads=8)
routes   = rola.RouteProducer([producer], d_v=64, num_heads=8, hidden_size=512)

factors = routes(hidden_states)          # the (q, k) pair, after the feature map
y, state = rola.rola_op(factors, v)
```

`rola_op` is the **single autograd boundary**. A call that requires gradients runs
the whole operation — forward and backward, from one graph — on the differentiable
reference, so a value and its gradient never come from two implementations that
merely agree to a tolerance. It takes no plan and no configuration: those are how the
op runs rather than what the model is, and the op decides how it runs, per call, from
the routing it was actually given.

### The layer

```python
from rola import RoLA, LearnedDecay

layer = RoLA(512, 8, routes,
             decay=LearnedDecay(2 ** -8, widths=routes.widths, num_heads=8),
             layer_idx=0).cuda()
y, _, state = layer(x, use_cache=True)
```

Four arguments, because everything else was either a fact about the feature map (which
lives on the feature map) or a launch decision (which the planner makes).

### The state

`rola.RoLAState` is a **fixed-size recurrent state, not a KV cache**: the per-layer
slot state, the short-convolution ring buffers, and a token tally. Every
`transformers` cache abstraction is built for the growing case, so RoLA owns its
own object and adapts at one point (`rola.state.as_layer_state`). `transformers`
interop lives behind the `[transformers]` extra.

### The planner

The planner derives the runtime geometry — `BC`, the CTA's resident state tile — as a
closed form over launch-static facts (the topology, the built binary's own dispatch
table, the machine), inside the op, per call. There is no cost model and nothing to
calibrate. It is not on the public surface: `rola.expert` is the marked door for a
caller that must pin an axis, and a
`Plan` built by hand there can be executed straight through `rola.expert.execute` —
the kernel's correctness does not depend on which geometry the planner picked, and
that is what Tier 2 exists to prove.

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

- [`docs/architecture/`](docs/architecture/README.md) — the permanent decision
  record: the self-masking theorem, the affine-recurrence rule, the factored
  representation, fp32 accumulation, closed-world codegen, the planner
  principle, and the guarantees/choices/preferences taxonomy.
- [`docs/internals/`](docs/internals/README.md) — one document per source file,
  carrying the reasons attached to a line, plus the deletion ledger.
- [`docs/testing.md`](docs/testing.md) — the tiers, the tolerance discipline, the
  geometry matrix and its meta-tests, and the conformance data-regime box.
- [`docs/measurement.md`](docs/measurement.md) — the one stopwatch, the discipline
  the probe harness acquires rather than documents, and the append-only ledger.
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

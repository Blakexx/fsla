# FSLA — Factored Sparse Linear Attention

FSLA is linear attention whose **feature map is a product of per-level coefficient vectors**. A head has levels
`l = 1..D` of width `b_l`; a token's query and its key are each a vector of coefficients per level, and the feature
a token reaches at a leaf is the product of one coefficient from every level. The recurrent state therefore has
`N = prod(b_l)` features, tens of thousands for two levels of 256, while a token's features cost `sum(b_l)` numbers to
carry. The levels can be sparse (entmax, sparsemax), which gives exact zeros that the producer declares as **masks**
beside the coefficients; the kernel's work follows the live digits, not `N`. That is what makes a feature map this
large usable, where dense linear-attention kernels hold the feature dimension to the hundreds. The state has a fixed
capacity chosen by the topology; it is not a cache that grows with the sequence.

The executable definition is an fp64 oracle (`fsla.naive_fsla`); the kernels are optimizations of it, never a second
definition ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). This repository is the kernels, the layer around them,
and the evidence for both. The API keeps the word "routing" in its names (`fsla.Routing`, `split_routing`); the
construction is factored sparse linear attention, not a router.

Paper: *(link on publication)* · Citation: [below](#citation)

## Status today

The library is architecture-generic by design: kernels are built per architecture, and a new architecture or arm slots
in through the architecture table and the arm registry ([`docs/bringup.md`](docs/bringup.md) for an architecture,
[`docs/build-graph.md`](docs/build-graph.md#arms) for an arm). **Hopper (`sm_90a`) is the one architecture built
today**; one that is not built refuses to build and refuses to load. Kernels are added one at a time, each designed and
measured in the lab before it is written into the library.

| piece | state |
|---|---|
| fp64 oracle, routing declaration, activations, the layer's modules | built, tested against the oracle |
| decode (the `T = 1` step, over the paged fixed-size state) | **built** |
| entmax / softmax row solves, the paged state | built |
| prefill: the carry, the gather pass, the intra, the producer kernel | **not built.** Designed and measured in [`docs/lab/`](docs/lab/README.md); `fsla_op` refuses every prefill call by name until the carry exists |
| backward | **not built.** `fsla_op` refuses any call that carries a gradient; there is no training path yet |
| depth three and four, state in and out of a prefill | **not built** (owed: [`docs/open-work.md`](docs/open-work.md)) |

[`docs/open-work.md`](docs/open-work.md) is the full list of what is owed, with where each item's contract lives.
Nothing here is a speed claim: no end-to-end performance number is stated for the built decode kernel. The measured
facts the designs rest on are the lab pages; for example the intra's window as a kernel of its own costs 176 SM-cycles a
token at 1,024 tokens with the levels tile-major ([`docs/lab/intra_window.md`](docs/lab/intra_window.md)), and a carry,
its denominator and an intra together run 0.392 ms a head on two streams against 0.401 on one at 65,536 tokens, one
cell, tables cached ([`docs/lab/call_pair.md`](docs/lab/call_pair.md)). Those are measurements of lab kernels on one
H100, not of the library.

## What the interface looks like

Two levels. **The layer** declares its levels and does everything. **The op** takes what a producer made: factors,
values and masks, and a static plan. Masks are data in each consumer's layout, and the op computes with the product
`r * m`; sparsity is declared by the producer, never detected by the kernel
([`docs/api.md` §2.6](docs/api.md#26-the-masks-are-data-in-each-consumers-native-layout)). Plans are frozen static
objects chosen from the declaration, never a runtime configuration. The op is agnostic to activations: no activation
reaches it. The producer implements a set of them and a layer's configuration selects one. Anything outside the built
envelope is refused by name rather than routed to a slower path: there is no fallback.

The first-class configuration is the alternating scheme: at each level one duty (read or write) is entmax and the
other is softmax, alternating by level (`fsla.split_routing`).

```python
import fsla

# two levels of 256 digits: N = 65,536 features; level 0 sparse on the write side, level 1 on the read side
routing = fsla.Routing([fsla.split_routing(256, alpha=1.5, sparse_duty="write"),
                        fsla.split_routing(256, alpha=1.5, sparse_duty="read")])
layer = fsla.FSLA(routing, hidden_size=1024, num_heads=8, d_v=64, layer_idx=0).cuda().eval()

cont = fsla.LayerContinuation(state=fsla.state())
y, cont = layer(prompt, continuation=cont)   # prompt: [B, T, 1024]; the prefill (refused by name until the carry is built)
y, cont = layer(token, continuation=cont)    # token:  [B, 1, 1024]; one decode step against the fixed-size state
```

`fsla.union_routing(width, alpha)` is the alternative level: one shared entmax solve split across the two duties.

The same call through the op: the producer makes a call's factors, values and masks (`produce`), and each call takes the validated
descriptor of its kind (`prefill_descriptor`, `decode_descriptor`) beside its data. There is no plan object: what the call derives goes in a
workspace the caller builds (`fsla.build_workspace`).

```python
import torch
from fsla.ops.produce import ProducerPlan, ProducerWeights, produce, prefill_descriptor, decode_descriptor, projection_dtype
from fsla.routing.masks import DECODE          # PREFILL is the default consumer
from fsla.engine.dags.decode_dag import decode_step_gated, growth_pending, admit_growth

H, hidden, d_v = 8, 1024, 64
x = ...                                        # [B, T, hidden], CUDA
weights = ProducerWeights(
    route_W=torch.randn(H, hidden, routing.columns, device="cuda", dtype=projection_dtype(x)) * hidden ** -0.5,
    route_bias=None,                           # Routing(bias=False)
    gain_bias=torch.zeros(H, routing.gain_columns, device="cuda", dtype=projection_dtype(x)),
    v_W=torch.randn(H * d_v, hidden, device="cuda", dtype=torch.bfloat16))

# prefill: the masks come in the prefill layout (refused by name until the carry is built, as in the layer)
plan = ProducerPlan(routing=routing, num_heads=H, hidden_size=hidden, d_v=d_v)
routes, values, masks = produce(x, plan, weights)
state = fsla.state()
descriptor = prefill_descriptor(routing.topology, d_v=d_v, decay=False, tokens=x.shape[1], batch_heads=x.shape[0] * H, stateful=True)
workspace = fsla.build_workspace(routes, values, masks, descriptor)
y, state = fsla.fsla_op(routes, values, masks, descriptor, workspace=workspace, state=state)

# decode: the same producer, the masks in the decode layout, and the descriptor of the call on this device
step_plan = ProducerPlan(routing=routing, num_heads=H, hidden_size=hidden, d_v=d_v, consumer=DECODE)
routes, values, masks = produce(x_token, step_plan, weights)       # x_token: [B, 1, hidden]
descriptor = decode_descriptor(routing.topology, d_v=d_v, decay=False, batch_heads=x_token.shape[0] * H, device=x_token.device)
y, state = decode_step_gated(routes, values, masks, state, descriptor)
if growth_pending(state):                       # a head whose state needed one more atom skipped itself
    admit_growth(state)
    y, state = decode_step_gated(routes, values, masks, state, descriptor)
```

The masks' layouts: **prefill**, int32 `[B, H, ceil(T / 32), width_l]` per level and side, token-packed (bit `t % 32` of
word `t // 32` is token `t`'s liveness of that digit); **decode**, int32 `[B, H, ceil(width_l / 32)]`, digit-packed
(bit `d % 32` of word `d // 32` is digit `d`). A level whose activation has no exact zeros carries an all-live mask.
`fsla.naive_fsla` is the oracle, shipped in the wheel so a suspected numerical problem can be checked on your own
shapes with no checkout. The contract in full is [`docs/api.md`](docs/api.md).

## Building and testing

Bazel is the one entry point: there is no source install (`pip install .` refuses), and no tool is run by hand.
Setup of a machine and a checkout is [`docs/setup.md`](docs/setup.md).

```bash
bazel run //:init                      # set up the machine, wire the commit gate
bazel run //:dev_env                   # the checkout's venv
bazel run //:check                     # every dependency, one row each
bazel run //:library                   # build the extension for the built architectures
bazel run //:wheels                    # the two wheels (fsla, fsla-cu13) of a clean commit

bazel test //:gate                     # the commit checks: lints, docrefs, the committed generated files, the lab's tests
bazel test //tests:cpu                 # each unit's public contract, no device
bazel test //tests:unit_cuda           # the unit tier's CUDA rows
bazel test //tests:oracle              # kernels against the fp64 oracle
bazel test //tests:integration         # composition: the layer, dispatch, paging
bazel run  //:ratify -- --census       # every instantiation's registers and spills
```

The build is ahead-of-time and closed-world: a pinned assembler, `code=sm_XX` only (no PTX, so nothing is
JIT-compiled for a device nobody measured), and gates on every build ([`docs/build.md`](docs/build.md),
[`docs/ratification.md`](docs/ratification.md)). The three test tiers (oracle, integration, unit) are described in
[`docs/testing.md`](docs/testing.md); never run `pytest tests/` unscoped.

Requirements: Python 3.10 or newer, torch 2.13 or newer built for the toolchain's CUDA major, a GPU of a built architecture (Hopper today).

## Supported configurations

This table is **generated** by `tools/supported.py` from the contract (`hardware/`: every run-time context FSLA accepts, which the build
guarantees an arm for) and the architectures the build produces; the gate fails if it drifts. Change the contract, not the table.

<!-- BEGIN GENERATED: tools/supported.py -->
| Compute capability | GPUs | d_v | levels | decay | prefill | tokens | batch_heads | stateful | width_bits_1 | width_bits_2 | width_bits_3 | width_bits_4 | read_listed_1 | read_listed_2 | read_listed_3 | read_listed_4 | write_listed_1 | write_listed_2 | write_listed_3 | write_listed_4 | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..64 | 1, 2, 3, 4 | 0, 1 | 0 | 1 | 1..536870911 | 1 | 4..8 | 0, 4..8 | 0, 4..8 | 0, 4..8 | 0..1 | 0, 0..1 | 0, 0..1 | 0, 0..1 | 0..1 | 0, 0..1 | 0, 0..1 | 0, 0..1 | `cu13` | ptxas 13.5.37 | built |
<!-- END GENERATED: tools/supported.py -->

An architecture that is not built refuses to load. Bringing one up is a measurement:
[`docs/bringup.md`](docs/bringup.md).

## Repository layout

| path | what lives there |
|---|---|
| `fsla/` | the Python package: the oracle, routing, the layer, the state, the engine's plans and graphs |
| `csrc/fsla/` | the CUDA kernels (decode, the entmax solves, paging) and the code they share |
| `tests/` | `oracle/`, `integration/`, `unit/` |
| `measure/` | the cells, the timing subjects, and the lab (`measure/lab/`: small CUDA experiments on the machine) |
| `tools/` | the gates, the census, the toolchain pins and the generators (`tools/supported.py` writes this README's support table) |
| `bazel/`, `BUILD.bazel`, `MODULE.bazel` | the build graph |
| `docs/` | the documents, one owner per topic ([`docs/README.md`](docs/README.md)) |

Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (what is computed and what is built),
[`docs/api.md`](docs/api.md), [`docs/open-work.md`](docs/open-work.md), and
[`docs/measurement.md`](docs/measurement.md) for how anything here is measured. A contribution that changes a kernel
starts from [`docs/KERNEL_STANDARDS.md`](docs/KERNEL_STANDARDS.md) and the lab ([`docs/lab/`](docs/lab/README.md)).

## Related repositories

| Repository | What lives there |
|---|---|
| [`fsla-devtools`](https://github.com/Blakexx/fsla-devtools) | the tools every FSLA repository shares: the build system front door, the central cells, the timing system |
| [`warpline`](https://github.com/Blakexx/warpline) | the GPU lab's checker and recorder, and the stamp header the lab's experiments include |
| [`fsla-bench`](https://github.com/Blakexx/fsla-bench) | the benchmarks and the measurement suite |
| [`fsla-results`](https://github.com/Blakexx/fsla-results) | the measurement records every reported number is read from |
| [`chromatic-recall`](https://github.com/Blakexx/chromatic-recall) | recall tasks whose difficulty is a certified number (the chromatic number of the contract's conflict graph); the research harness for what a state of a given size can recall |
| [`fsla-zoology`](https://github.com/Blakexx/fsla-zoology) | a fork of HazyResearch/zoology carrying our PR-ready fixes and the FSLA mixer |
| [`flash-linear-attention`](https://github.com/Blakexx/flash-linear-attention) | a fork whose `fsla` branch holds the FSLA layer and HF model over this package |
| [`fsla-paper`](https://github.com/Blakexx/fsla-paper) | the paper |

## Citation

```bibtex
@misc{fsla,
  title  = {Factored Sparse Linear Attention},
  author = {Bottum, Blake},
  year   = {2026},
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

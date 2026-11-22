# FSLA — Factored Sparse Linear Attention

FSLA is linear attention whose recurrent state is a **set of addressable slots** (leaves), and whose routing decides,
for every token, which slots it reads from and writes to. The routing is a *feature map*: each head's query and key are
products of per-level routing rows, so the leaf address space is a product space (a mixed-radix tree of levels) and the
routing's cost is the sum of the level widths, not the number of leaves. The state has a fixed capacity chosen by the
topology; it is not a cache that grows with the sequence.

Written out over all leaves, the recurrence is textbook gated linear attention, and a dense implementation pays for
every leaf on every token. A **real kernel** is the point: the factored structure and the exact zeros the routing
produces (entmax) only become speed if the kernel reads them, so this repository is the kernels, the layer around them,
and the evidence for both. The executable definition is an fp64 oracle (`fsla.naive_fsla`); the kernels are
optimizations of it, never a second definition ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)).

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
| decay in the prefill, depth three and four, state in and out of a prefill | not built (decode has decay) |

[`docs/open-work.md`](docs/open-work.md) is the full list of what is owed, with where each item's contract lives.
Nothing here is a speed claim: no end-to-end performance number is stated for the built decode kernel. The measured
facts the designs rest on are the lab pages; for example the intra's window as a kernel of its own costs 176 SM-cycles a
token at 1,024 tokens with the levels tile-major ([`docs/lab/intra_window.md`](docs/lab/intra_window.md)), and a carry,
its denominator and an intra together run 0.392 ms a head on two streams against 0.401 on one at 65,536 tokens, one
cell, tables cached ([`docs/lab/call_pair.md`](docs/lab/call_pair.md)). Those are measurements of lab kernels on one
H100, not of the library.

## What the interface looks like

Masks are data. A routing row carries its liveness as a mask beside it, in the layout of the kernel that consumes it,
and the op computes with the product `r * m`; sparsity is declared by the producer, never detected by the kernel
([`docs/api.md` §2.6](docs/api.md#26-the-masks-are-data-in-each-consumers-native-layout)). Plans (the carve, the state's
backing) are frozen static objects chosen from the declaration, never a runtime configuration. Anything outside the
built envelope is refused by name rather than routed to a slower path: there is no fallback.

```python
import fsla

routing = fsla.Routing(fsla.uniform(2, fsla.union_routing(64, alpha=1.5)))
decay   = fsla.LearnedDecay(2 ** -8, widths=routing.widths, num_heads=8)
layer   = fsla.FSLA(routing, hidden_size=1024, num_heads=8, d_v=64, decay=decay, layer_idx=0).cuda().eval()

# decode: one token of a carried sequence against the fixed-size state
cont = fsla.LayerContinuation(state=fsla.state())
y, cont = layer(token, continuation=cont)            # token: [B, 1, 1024]
```

`fsla.fsla_op(factors, values, masks, decay, plan=plan, state=state)` is the op under the layer; `fsla.naive_fsla` is the
oracle, shipped in the wheel so a suspected numerical problem can be checked on your own shapes with no checkout. The
contract in full is [`docs/api.md`](docs/api.md).

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

This table is **generated** by `tools/supported.py` from the kernel's architecture table and the architectures the build
produces; the gate fails if it drifts. Change the generator, not the table.

<!-- BEGIN GENERATED: tools/supported.py -->
| Compute capability | GPUs | Toolchain | Assembler | Status |
|---|---|---|---|---|
| `sm_90a` | H100 | `cu13` | ptxas 13.0.48 | built |
| `sm_100a` | B200 | — | — | **not built — will refuse to load** |
<!-- END GENERATED: tools/supported.py -->

An architecture that is not built refuses to load. Bringing one up is a measurement:
[`docs/bringup.md`](docs/bringup.md).

### Routing activations

Also generated, from the registry every validator reads (`fsla/routing/activations.py`). `kerneled` decides dispatch: an
unratified activation is refused by the kernel entry.

<!-- BEGIN GENERATED: fsla.routing.activations.capability_table -->
| activation | exact_zeros | kerneled |
| --- | --- | --- |
| entmax(alpha=1.5) | yes | yes |
| entmax(alpha=2) | yes | yes |
| opaque | yes | yes |
| softmax | no | yes |
<!-- END GENERATED: fsla.routing.activations.capability_table -->

## Repository layout

| path | what lives there |
|---|---|
| `fsla/` | the Python package: the oracle, routing, the layer, the state, the engine's plans and graphs |
| `csrc/fsla/` | the CUDA kernels (decode, the entmax solves, paging) and the code they share |
| `tests/` | `oracle/`, `integration/`, `unit/` |
| `measure/` | the cells, the timing subjects, and the lab (`measure/lab/`: small CUDA experiments on the machine) |
| `tools/` | the gates, the census, the toolchain pins and the generators (`tools/supported.py` writes this README's tables) |
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
| `chromatic-recall` | recall tasks whose difficulty is a certified number (the chromatic number of the contract's conflict graph); the research harness for what a state of a given size can recall |
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

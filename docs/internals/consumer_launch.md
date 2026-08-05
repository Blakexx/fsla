# `csrc/rola/src/consumer_launch.cuh` — the per-arm host launcher

The definition of the two host entry points every consumer instantiation is reached
through: `launch_consumer_arm` (launch this arm) and `query_consumer_occupancy_arm`
(ask the driver about its residency). Both are function templates over the same
tuple the kernel is: `<BT, BC, DV, D, PREC, DECAY, GLOBAL, COMPACT>`.

Symbols: an *arm* is one instantiation of `consumer_forward_kernel`; a *shard* is one
generated translation unit carrying 64 of them
(`csrc/rola/src/instantiations/README.md`); *whole-program* compilation means each
translation unit's device code is assembled independently, with no device linker
(`-rdc` is off).

---

## <a id="shard-only"></a>1. This header is included by the shards and by nothing else

A `__global__` function template cannot be shared across whole-program translation
units: without `-rdc` each object's device code is self-contained, so an arm must be
instantiated in exactly one place and reached from everywhere else through a HOST
symbol. That is the arrangement here, and it is also flash-attention's: `flash.h`
declares `run_mha_fwd_<...>`, `flash_fwd_launch_template.h` defines it, the generated
per-configuration translation units include the second, and the API translation unit
includes only the first.

The asymmetry is the whole mechanism, and it is fragile in one specific direction. If
this header ever became visible to `consumer.cu`, the compiler would satisfy that
file's own uses by IMPLICITLY INSTANTIATING every dispatched arm locally. The build
would link. The ratification manifest would be byte-identical — `consumer_forward_kernel`
lives in a named namespace, so its mangled name does not depend on which translation
unit expands it. The geometry census would still report the whole matrix. `nm` would
still report 512 symbols, because a `__global__` template's host stub is weak and the
linker keeps exactly one. Nothing would be sharded, every kernel would be compiled
twice, and no gate in that list could tell.

So the arrangement is checked from both ends:

- `tests/unit/test_shard_partition.py` refuses the `#include` in `consumer.cu`;
- `tools/fatbin_entries.py` counts DEVICE ENTRY POINTS in the built fatbin, per
  architecture and per originating object, which is where cross-translation-unit
  duplication is visible at all.

The declarations, and the `ShardRow` table the queries resolve through, are in
`consumer.cuh`.

## <a id="no-torch"></a>2. No torch, and the stream comes in as an argument

These are the only host functions in the engine that take a `cudaStream_t` instead of
asking `at::cuda::getCurrentCUDAStream()` for one. Two reasons, both measured:

1. **Header parse.** Eight translation units that each parsed torch's host headers
   would spend a large part of what the split saves before compiling a line of
   kernel. Torch-free, a shard's peak resident set is 0.61 GB against the
   pre-shard monolith's 4.4 GB, and its wall time is ~43 s for both architectures.
2. **Flag surface.** `tools/ratify.py` compiles these exact files to produce the
   register and spill manifest. A gate that had to reproduce torch's ABI defines and
   include layout to match the shipped build would be one more thing that can differ
   between the binary and the numbers certifying it.

## <a id="error-state"></a>3. Errors travel in the CUDA runtime's error state

There is no `C10_CUDA_CHECK` here and no return code. A refused shared-memory opt-in
returns early; a failed launch sets the error asynchronously; either way the failure
is recorded in the per-thread error state, and `launch_consumer` (`consumer.cu`)
reads it through `C10_CUDA_KERNEL_LAUNCH_CHECK()` the instant the call returns. That
is ONE error site for 512 arms instead of 512, and it is what lets the launcher's
signature be the plain `void(const ConsumerParams&, cudaStream_t)` that
`ShardRow::launch` stores.

Returning early on a refused `cudaFuncSetAttribute` rather than launching anyway is
deliberate: launching without the opt-in would fail too, and reporting the first
cause is more useful than reporting its consequence.

## <a id="launch"></a>4. The launch

Three steps, in this order, unchanged from the pre-shard dispatch:

1. opt into the arm's `constexpr` shared-memory footprint if it exceeds the 48 KB
   static limit — a property of the KERNEL, so it happens before the grid is looked
   at;
2. return if the grid is empty (`p.n_owners == 0`);
3. launch `<<<p.n_owners, kConsumerThreads, smem, stream>>>`.

## <a id="residency"></a>5. The residency query, instantiated for every arm

`rola_consumer_occupancy` asks about 32 of the 512 arms (`D = 2`, `global`, the
production precision — the axes that move the answer). The wrapper is nevertheless
instantiated for ALL of them.

It is host-only code over a kernel address that already exists in the shard, so it
costs nothing measurable, and the alternative is a partition whose launcher members
and query members differ — a special case with no reason behind it, which is exactly
what the standing "apply a strategy uniformly across configurations" rule refuses. It
also means P17's re-partition, or a future query on a fourth axis, needs no
corresponding change here.

---

Provenance: landed by P02 (TU shard), design `p02_artifacts/shard_design.md` as
amended by `p02_shard_review.md` (A2, A3, A7); the compile and residency
measurements are in `workflows/refactor/p02_landing.md`.

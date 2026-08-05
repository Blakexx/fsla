# `csrc/rola/src/host_api.cuh` — the host entry surface

The declarations of the entries the DEVICE translation units define, and the only
header the pybind translation unit (`csrc/rola/rola_api.cpp`) needs to see in order to
bind them.

The source file is already at the mirror's target comment density; this doc is its
reference, not a relocation of prose that used to be in it.

**Nothing here is a public API.** `rola/ops/_ext.py` is the only Python importer and
`rola/interface.py` is the only thing that should call through it.

---

## <a id="why-header"></a>1. Why this header exists

The JIT-era shape put `PYBIND11_MODULE` inside `consumer.cu` and registered the paging
entries from two `*_pybind.cuh` fragment headers that the same `.cu` `#include`d.

Under an ahead-of-time build that is a per-`-gencode` recompile of the binding code,
which is host-only and has nothing to gain from it. `flash_api.cpp` is the convention
this follows instead: the bindings live in ONE host TU, and this header is the
interface between it and the device TUs.

The consequence to preserve: **this file must stay declaration-only.** Anything with a
device body, a `__global__`, or a template that instantiates one belongs in the `.cu`
that owns it, or the split buys nothing.

## <a id="consumer-entries"></a>2. The consumer's five entries

All defined in `consumer.cu`.

| entry | what it is |
|---|---|
| `rola_consumer_forward` | the fused forward (spec §3/§5) |
| `rola_consumer_occupancy` | the DRIVER's CTAs/SM for the consumer kernel at its real SMEM size |
| `rola_consumer_smem_bytes` | the kernel's dynamic shared-memory footprint per CTA |
| `rola_consumer_check_arch_table` | the closed-world rule-4 arch check |
| `rola_sm_shared_bytes` | `cudaDevAttrMaxSharedMemoryPerMultiprocessor`, the per-SM SMEM budget the plan records (§10 ruling) |

`rola_consumer_check_arch_table` is validated ONCE, on the first launch, from
`launch_consumer` — not on every call, and not at module import. It is the runtime half
of the pair described in [`arch_table.md`](arch_table.md#own-header); the compile-time
half is a `static_assert`.

`rola_consumer_occupancy` and `rola_sm_shared_bytes` are queries against the driver, not
against the compile-time table. That is deliberate: spec §6.5 makes the driver's
occupancy API the sole planning authority, and `arch_table.cuh` a hint generator for
`__launch_bounds__` only.

## <a id="sequence-parallel-block"></a>3. The §6.4 sequence-parallelism block is all-or-nothing

`rola_consumer_forward`'s trailing fourteen `c10::optional<at::Tensor>` parameters are a
single unit. Either every one of them is present (a SPLIT plan) or none is — the latter
being `P = 1`, the primary configuration, which takes the single-launch path unchanged.

They fall into four groups:

| group | tensors | shape / keying |
|---|---|---|
| per-owner work assignment | `work_seg`, `work_nseg`, `work_slot` | `[n_owners]` int32, keyed like `owners` |
| summary launches | `summary_owners`, `summary_seg`, `summary_nseg`, `summary_slot` | `summary_owners` is `[n_summary, 2]` int32; the rest `[n_summary]` |
| the cross-segment scan | `scan_owners`, `scan_nseg`, `scan_slot` | `scan_owners` is `[n_split, 2]` int32; the rest `[n_split]` |
| the split workspace | `ws_state`, `ws_scale`, `ws_leaf`, `ws_count` | `[n_slots, BC, cols]` fp32, `[n_slots, BC]` fp32, `[n_slots, BC]` int32, and the count |

The segment a slot owns is turned into a tile mask by
[`segment_mask.cuh`](segment_mask.md#word-mask), and that mask must agree bit for bit
with the host partition that filled `work_seg` / `work_nseg`.

## <a id="scalar-plan-args"></a>4. The scalar plan arguments are the compile-time axes, passed by value

`BT`, `BC`, `global_norm`, `mma_precision` and `compact` arrive as ordinary arguments
and are dispatched inside `consumer.cu` to a template instantiation. They are
PLAN-TIME axes, not runtime branches: the plan chooses one instantiation and the kernel
body branches on none of them. `mma_precision` is a `std::string` at this boundary and
becomes `MmaPrec` (see [`readout_mma.md`](readout_mma.md#precision-axis)) on the other
side of the dispatch.

The level-geometry vectors (`level_stride`, `level_modulus`, `level_row_offset`,
`level_width`, `owner_stride`) and the counts (`first_spanned_level`, `slice_digits`,
`n_slices`, `owners_total`, `n_heads`) describe the routing tree's addressing and are
consumed as data.

## <a id="empty-paging-namespace"></a>5. The empty `rola::paging` namespace

The file closes with `namespace rola::paging {}`. It declares nothing: the paging
entries the binder needs come from `src/paging/vmm_owner.cuh`, which `rola_api.cpp`
includes directly. See the finding note in
[`rola_api.md`](rola_api.md#one-module) about the JIT-era fragment headers this
replaced.

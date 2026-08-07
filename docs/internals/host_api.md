# `csrc/rola/src/host_api.cuh` — the host entry surface

The declarations of the entries the DEVICE translation units define, and the only
header the pybind translation unit (`csrc/rola/rola_api.cpp`) needs to see in order to
bind them.

The source file is already at the mirror's target comment density; this doc is its
reference rather than a relocation of its prose.

**Nothing here is a public API.** `rola/ops/_ext.py` is the only Python importer and
`rola/interface.py` is the only thing that should call through it.

---

## <a id="why-header"></a>1. Why this header exists

Binding code is host-only. Registered from inside a `.cu` — a `PYBIND11_MODULE` in the
kernel TU, or fragment headers it `#include`s — it is recompiled once per `-gencode`
target of an ahead-of-time build, for nothing. `flash_api.cpp` is the convention this
follows: the bindings live in ONE host TU, and this header is the interface between it
and the device TUs.

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

The segment a slot owns is a WORD RANGE over the membership scan's stream
(`w_lo`/`w_hi`, derived inline in `consumer_kernel.cuh` from `p.n_twords`), not a
tile mask — `segment_mask.cuh` and its tile-range arithmetic are dead (the
endgame-executor rewrite dropped tile candidacy entirely; revival = git,
[`DELETIONS.md`](DELETIONS.md)). The word range a slot owns must agree with the
host partition that filled `work_seg` / `work_nseg`.

## <a id="scalar-plan-args"></a>4. The scalar plan arguments are the compile-time axes, passed by value

`BC` arrives as an ordinary argument and is dispatched inside `consumer.cu`,
alongside `p.D` (the length of the `amp_read` vector) and `decay` (`=
dials.has_value()`), to a template instantiation `consumer_forward_kernel<BC, DV,
D, DECAY>`. `DV` is read off `v`'s last dimension, never passed explicitly. All
four are PLAN-TIME axes, not runtime branches: the plan chooses one instantiation
and the kernel body branches on none of them.

There is no `BT` argument and no `global_norm` or `mma_precision` argument. The
`BT` axis is dead — the executor's work unit is always a full group of
`kMmaKQuantum` gathered tokens, never a tile of a runtime-chosen extent. Raw
normalization (`global_norm`) is dead — the mass column and the `num`/`den`
divide are unconditional. The tf32x3/bf16 precision split (`mma_precision`,
`MmaPrec`) is dead — bf16 storage and fp32 accumulation are the only precision
story, with the fp64 torch reference as the oracle a caller needing more
precision falls back to. All three: revival = git, [`DELETIONS.md`](DELETIONS.md).

The level-geometry vectors (`level_stride`, `level_modulus`, `level_row_offset`,
`level_width`, `owner_stride`) and the counts (`first_spanned_level`, `slice_digits`,
`n_slices`, `owners_total`, `n_heads`) describe the routing tree's addressing and are
consumed as data.

## <a id="empty-paging-namespace"></a>5. The empty `rola::paging` namespace

The file closes with `namespace rola::paging {}`. It declares nothing: the paging
entries the binder needs come from `src/paging/vmm_owner.cuh`, which `rola_api.cpp`
includes directly. Why the consumer and the paging arena register into ONE module:
[`rola_api.md`](rola_api.md#one-module).

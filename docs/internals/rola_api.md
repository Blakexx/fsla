# `csrc/rola/rola_api.cpp` — the extension's Python binding

The only translation unit that knows pybind exists. One `PYBIND11_MODULE`, four
groups of `m.def`, and one `py::class_`.

`m.def` bodies live here and nowhere else; the declarations they name come from
`src/chunk/chunk_api.cuh`, `src/decode/decode_api.cuh`, `src/entmax/entmax.cuh` and
`src/paging/vmm_owner.cuh`.

Symbols: a TU is a translation unit; `-gencode` is the nvcc flag that emits one device
code path per target architecture; `TORCH_EXTENSION_NAME` is the macro the build
substitutes for the module name.

---

## <a id="why-cpp"></a>1. Why it is a `.cpp` and not part of a kernel `.cu`

Binding code is host-only. Left inside a `.cu` it is recompiled once per `-gencode`
target for no benefit, and **every future architecture added to the ratified set
multiplies that cost.** `flash_api.cpp` is the convention; this is the same split.

Every declaration the binder needs is a `*_api.cuh`-shaped host header, so nothing
here needs a device body in scope.

## <a id="one-module"></a>2. One module, not two

The consumer and the §7.1 paging arena register into the SAME extension: the consumer
reads the page table the plan publishes, so splitting them would put a **cross-module
ABI between two halves of one contract.**

Registration happens HERE and nowhere else: every `m.def` body is in this file. No
fragment header registers anything, so there is one place to read to know what the
module exports.

The T=1 decode path registers here for the same reason: it is a SIBLING of the
prefill consumer, not a specialization of it — shared produce contract, own kernel,
frozen config — and one contract deserves one ABI boundary.

## <a id="module-name"></a>3. The module name comes from the build

`TORCH_EXTENSION_NAME` is set by `setup.py` to `rola._C` — **namespaced inside the
package**, so the binary lands in `rola/` for both an in-place build and a wheel.
`rola/ops/_ext.py` imports it package-relative and never builds it.

**Nothing in this file or under `src/` spells the name**, so the packaging layout can
move without touching a kernel symbol. Hard-coding `_C` anywhere in the C++ would undo
that.

## <a id="registered-entries"></a>4. What is registered

| group | entries |
|---|---|
| producer union entmax solve | `entmax_union_forward`, `entmax_union_backward` |
| producer per-side solves | `entmax_factor_forward`, `entmax_factor_backward`, `routing_softmax_forward`, `routing_softmax_backward` |
| the chunk consumer (the prefill arm) | `chunk_facts`, `chunk_block_bits`, `chunk_forward_raw`, `chunk_forward`, `chunk_divide_pass`, `chunk_build_stamp`, `chunk_arms` |
| the chunk consumer's reverse pass | `chunk_backward_seeds`, `chunk_scan_a_probe`, `chunk_scan_b_probe`, `chunk_walk_probe`, `chunk_scan_a_census`, `chunk_scan_b_census` |
| the T=1 decode path (thunder §6b) | `rola_decode_forward`, `rola_decode_work_words`, `rola_decode_capacity`, `rola_decode_producer_width_mirror` |
| the VMM owner | `rola_vmm_probe`, `rola_vmm_create`, and the `RoLAVmmOwner` class |

The two scans carry a census EACH. They share a geometry and not a live set, so one row
for both would be a floor claim about a kernel nobody measured
([`chunk/backward.md`](chunk/backward.md#reg-floor-scan)).

Two of the four decode entries are gates rather than operations, and their docstrings say
so: `rola_decode_capacity` reports the decode path's maximum level width, which must be
`>=` the producer's `MAX_BRANCH_WIDTH` (`tests/unit/test_decode_capacity.py` pins the
pair), and `rola_decode_producer_width_mirror` is the C++ mirror of that producer
constant, pinned EQUAL to the Python value so a stale mirror cannot silently satisfy the
capacity `static_assert`.

`rola_decode_forward` and `chunk_block_bits` are the entries bound with explicit
`py::arg` names; the prefill consumer's much larger signature is positional.

The tiled consumer's five entries — `rola_consumer_forward`,
`rola_consumer_occupancy`, `rola_consumer_check_arch_table`,
`rola_consumer_smem_bytes` and `rola_sm_shared_bytes` — were bound here until P67
D2 retired that arm; none of them is in the module today, and the device-side
entry list is the check that says so ([`DELETIONS.md`](DELETIONS.md)).

## <a id="deleted-page-plan"></a>5. There is no CUDA page-plan producer

The page table the consumer reads is published by the torch PLANNED-EXACT-ASYNC arena
in `rola/ops/paging.py`, and nothing under `rola/`, `tests/` or `benchmarks/` reaches
a CUDA page planner. Recorded here because the absence is the fact worth keeping:
these eight names are NOT bound, and a reader looking for them is looking for the
arena.

- `page_plan_abi_version`
- `page_plan_status_codes`
- `page_plan_workspace_bytes`
- `rola_page_plan_prepare`, `rola_page_plan_finalize`, `rola_page_plan_publish`
- `rola_page_admit_mask`
- `rola_page_cache_reset`

Their removal was safe on the no-dead-code standard because no Python caller ever
reached them; the register of that removal is
[`DELETIONS.md`](DELETIONS.md).

## <a id="vmm-binding"></a>7. The VMM owner's binding shape

`rola_vmm_probe` is bound as a lambda that flattens `VmmProbe` into a `py::dict`
(`device`, `supported`, `allocation_granularity`, `reason`) rather than exposing the
struct — a probe result is diagnostic data, and a dict keeps it printable and stable
across additions.

`RoLAVmmOwner` is held by `std::shared_ptr`, which is required rather than stylistic:
the class is `enable_shared_from_this` and every exported tensor view captures a
`shared_from_this` to keep the driver mapping alive. See
[`paging/vmm_owner.md`](paging/vmm_owner.md#view-lifetime).

The mutating operations (`grow`, `rollback_to`, `reset`, `close`) are methods; every
observable (`closed`, `mapped_capacity_pages`, `dense_limit_pages`, `committed_bytes`,
`virtual_bytes`, `allocation_granularity`, `chunk_pages`, `backing_kind`) is a
`def_property_readonly`, so nothing on the Python side can write owner state except
through the four verbs.

`rola_vmm_create` is the factory. Its page geometry arrives as two extents,
`page_rows` and `page_cols`, and the owner never learns what either counts — the arena
passes `MMA_K_QUANTUM` and `d_v + 1`, which is the fused-mass page it addresses, and
keeping the names dimensionless is what stops the owner acquiring an opinion about page
meaning it has no business holding ([`paging/vmm_owner.md`](paging/vmm_owner.md)).
`target_chunk_bytes` defaults to 64 MiB at this boundary; the arena overrides it with the
device's own allocation granularity, which is the finest commitment the driver admits.

## <a id="the-step"></a>7a. `rola_decode_forward` — the whole step, one launch

The decode step is ONE kernel: the operand fold, the support expansion, the
token-stationary gather-GEMV and the step's OWN per-batch-head residency verdict are
phases of it, so what the boundary passes is the caller's RAW `[B, 1, H, *]` views and not
a widened plane. There is no separate fold entry and no `heads` argument — the fold is
inside the kernel and `v`'s own shape carries `H`. There is no separate admission-probe
entry either: the residency test reads the write map the walk was going to build anyway,
so what used to be a launch ahead of the step is now a pass over shared memory inside it.
The phases, and why neither the fold nor the admission question is its own launch, are
[`decode/decode_fold.md`](decode/decode_fold.md) and [`decode/decode.md`
§4c](decode/decode.md#absorbed-verdict); the step is [`decode/decode.md`](decode/decode.md).

`growth`, `growth_any`, `growth_ctr`, `done` and `atom_bits` are the verdict's own
buffers — bound as required arguments whichever backing the call is, so one ABI serves
both. `pool_slots`, `pool_cursor` and `pool_map` are the slack pool, present exactly on a
paged step whose state reserved one. Both groups are [`decode/decode_api.md`
§4-5](decode/decode_api.md#the-verdict).

<a id="work-words"></a>`rola_decode_work_words` is bound beside it because the expansion's
GLOBAL scratch is sized by a kernel-side rule — does the bitmap fit in this device's shared
memory? — and mirroring that rule in Python would put a drift surface in front of an
out-of-bounds write. It answers `0` at every shipped topology
([`decode/decode_api.md` §2](decode/decode_api.md#work-words)).

## The chunk consumer's entries (operator adoption)

The chunk group above is a PRIVATE seam: declarations in
`src/chunk/chunk_api.cuh`, `rola/ops/chunk.py` the only intended caller, and
`rola_op` the only public door onto it. See `docs/internals/chunk/`.

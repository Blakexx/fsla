# `csrc/rola/rola_api.cpp` — the extension's Python binding

The only translation unit that knows pybind exists. One `PYBIND11_MODULE`, five
groups of `m.def`, and one `py::class_`.

Symbols: a TU is a translation unit; `-gencode` is the nvcc flag that emits one device
code path per target architecture; `TORCH_EXTENSION_NAME` is the macro the build
substitutes for the module name.

---

## <a id="why-cpp"></a>1. Why it is a `.cpp` and not part of `consumer.cu`

Binding code is host-only. Left inside a `.cu` it is recompiled once per `-gencode`
target for no benefit, and **every future architecture added to the ratified set
multiplies that cost.** `flash_api.cpp` is the convention; this is the same split.

The declarations the binder needs are in `src/host_api.cuh` (plus `decode_api.cuh`,
`entmax.cuh`, `summaries.cuh` and `paging/vmm_owner.cuh`), so nothing here needs a
device body in scope.

## <a id="one-module"></a>2. One module, not two

The consumer and the §7.1 paging arena register into the SAME extension: the consumer
reads the page table the plan publishes, so splitting them would put a **cross-module
ABI between two halves of one contract.**

Registration happens HERE and nowhere else: every `m.def` body is in this file and
every declaration it needs is in `src/host_api.cuh`. No fragment header registers
anything, so there is one place to read to know what the module exports.

The T=1 decode path registers here for the same reason: it is a SIBLING of the
consumer, not a specialization of it — shared produce contract, own consumer, frozen
config — and one contract deserves one ABI boundary.

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
| the consumer | `rola_consumer_forward`, `rola_consumer_occupancy`, `rola_consumer_check_arch_table`, `rola_consumer_smem_bytes`, `rola_sm_shared_bytes` |
| the T=1 decode path (thunder §6b) | `rola_decode_forward`, `rola_decode_capacity`, `rola_decode_producer_width_mirror` |
| producer-epilogue summaries (§4.1/§4.2) | `rola_routing_summaries` |
| the VMM owner | `rola_vmm_probe`, `rola_vmm_create`, and the `RoLAVmmOwner` class |

Two of the decode entries are gates rather than operations, and their docstrings say
so: `rola_decode_capacity` reports the decode path's maximum level width, which must be
`>=` the producer's `MAX_BRANCH_WIDTH` (`tests/unit/test_decode_capacity.py` pins the
pair), and `rola_decode_producer_width_mirror` is the C++ mirror of that producer
constant, pinned EQUAL to the Python value so a stale mirror cannot silently satisfy the
capacity `static_assert`.

`rola_decode_forward` is the only entry bound with explicit `py::arg` names; the
consumer's much larger signature is positional.

## <a id="summaries-contract"></a>5. The routing-summary pass's calling contract

`rola_routing_summaries` (docs/internals/summaries.md) is **one launch, no
allocation, no host synchronization.** Every output is a caller-owned tensor that the
Python side zeroes beforehand and reads back in its ONE D2H transfer.

It produces occupancy, the quantum histogram, the candidate count, and §4.1's canonical
per-leaf write bitmap with its owner condensation.

That contract is why the entry takes outputs by reference rather than returning them:
allocating inside would force a synchronization the caller has designed out.

## <a id="deleted-page-plan"></a>6. There is no CUDA page-plan producer

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

## <a id="growth-probe"></a>7b. `rola_decode_growth_probe` — bound beside the step

The probe is registered as its own entry rather than as an argument to
`rola_decode_forward`, because it is a PRECONDITION of the step and not a mode of it: the
host runs it first, reads one `[BH]` int32, and only then decides whether the step needs
its admission at all — planning, when it does, from the write-atom set the probe published
beside that verdict. Binding it separately is what keeps that ordering visible at
the boundary instead of implied by a flag. Its contract is
[`decode_api.md` §4](decode_api.md#growth-probe).

## The chunk consumer's entries (operator adoption)

`chunk_union_table`, `chunk_forward_raw`, `chunk_forward`, `chunk_divide_pass`,
`chunk_build_stamp`, `chunk_arms` — the chunk consumer's private seam
(declarations in `src/chunk/chunk_api.cuh`; `rola/ops/chunk.py` is the only
intended caller, behind `rola_op`).  See docs/internals/chunk/.

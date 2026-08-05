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

The two JIT-era `*_pybind.cuh` fragment headers that used to do this registration are
gone — their `m.def` bodies are here, and their declarations are in `src/host_api.cuh`.

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

`rola_routing_summaries` (THUNDER_RUN_SPEC.md §4.1/§4.2) is **one launch, no
allocation, no host synchronization.** Every output is a caller-owned tensor that the
Python side zeroes beforehand and reads back in its ONE D2H transfer.

It produces occupancy, the quantum histogram, the candidate count, and §4.1's canonical
per-leaf write bitmap with its owner condensation.

That contract is why the entry takes outputs by reference rather than returning them:
allocating inside would force a synchronization the caller has designed out.

## <a id="deleted-page-plan"></a>6. The V2 CUDA page-plan producer is deleted, with eight bindings

Recorded here because the absence is the fact worth keeping. The deleted bindings:

- `page_plan_abi_version`
- `page_plan_status_codes`
- `page_plan_workspace_bytes`
- `rola_page_plan_prepare`, `rola_page_plan_finalize`, `rola_page_plan_publish`
- `rola_page_admit_mask`
- `rola_page_cache_reset`

Track P replaced the producer entirely with the torch PLANNED-EXACT-ASYNC arena in
`rola/ops/paging.py`. Nothing under `rola/`, `tests/` or `benchmarks/` referenced any of
the eight, so they were unreachable from Python and dead by the no-dead-code standard.

**The page table the consumer reads is now published by that arena.**

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
observable (`closed`, `mapped_capacity_pages`, `max_physical_pages`, `committed_bytes`,
`virtual_bytes`, `allocation_granularity`, `chunk_pages`, `backing_kind`) is a
`def_property_readonly`, so nothing on the Python side can write owner state except
through the four verbs.

`rola_vmm_create` is the factory, and its `target_chunk_bytes` defaults to 64 MiB at
this boundary.

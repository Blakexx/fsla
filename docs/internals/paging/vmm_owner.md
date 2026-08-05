# `csrc/rola/src/paging/vmm_owner.{cuh,cu}` — the CUDA driver VMM backing

`VmmOwner` reserves a virtual address range for each state PLANE up front and commits
physical memory into it in chunks, so a long-lived page cache can grow without ever
moving a base pointer.

Both source files are already at the mirror's target comment density; this doc is
their reference rather than a relocation of their prose.

**Scope discipline — the one thing to preserve.** `VmmOwner` owns *only* the
CUDA-driver backing. Logical page identity, allocation plans, and publication remain
in `SparseRoLAState`, so dense backing and VMM backing share ONE ABI. Anything that
knows what a page *means* does not belong in this class.

Symbols: a PLANE is one contiguous logical tensor (`value` or `mass`); a PAGE is one
slot of the state cache, `bp x dqk x dv` fp32 for value and `bp x dqk` fp32 for mass; a
CHUNK is one driver allocation handle mapped at the end of a plane's mapped prefix;
*granularity* is `cuMemGetAllocationGranularity`'s minimum; *committed* means physical
memory exists, *reserved* means only address space does.

---

## <a id="two-planes"></a>1. Two planes, reserved whole, mapped in a prefix

`reserve_planes()` reserves `max_physical_pages * page_bytes`, rounded up to the
allocation granularity, for `value_` — and for `mass_` only when `global_mass_`. The
reservation is address space, not memory.

`make_view` then hands out `at::from_blob` tensors spanning the **entire** reserved
extent:

| plane | view shape | dtype |
|---|---|---|
| value | `[max_physical_pages, bp, dqk, dv]` | fp32 |
| mass | `[max_physical_pages, bp, dqk]` | fp32 |

**Only the prefix reported by `mapped_capacity_pages()` is mapped and may be touched.**
The tensor's own shape does not tell a caller that; the contract does. Touching past
the mapped prefix is an unmapped-address fault, not an out-of-bounds the tensor
machinery can catch.

Keeping the view over the whole extent (rather than resizing it on every growth) is the
point of the design: the base pointer is stable for the process's life, so nothing
downstream re-publishes a pointer when the cache grows.

`mapped_capacity_pages()` is the MINIMUM over the live planes and `max_physical_pages_`
— so a partially-grown pair of planes reports the capacity both can serve, never the
larger one.

## <a id="probe"></a>2. `probe()` — capability is proven, not queried

The attribute query (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) and a
non-zero granularity are necessary but not sufficient. **A probe is not complete until
the driver can reserve, map, grant access, unmap, and release one granularity-sized
allocation in this context.** The probe performs that whole round trip and reports
`supported` only if every step returned `CUDA_SUCCESS`.

Its cleanup is written to follow the ACTUAL MAP STATE, not the final status: mapping
may have succeeded even when `cuMemSetAccess` failed, so `cuMemUnmap` is gated on the
`mapped` flag and `cuMemAddressFree` / `cuMemRelease` on the handles being non-zero.
Gating cleanup on `status` instead would leak a mapping on exactly the failure path
that most needs unwinding.

`create()` calls `probe()` and turns a negative into a message that names the fix:
`progressive_memory=True` requires driver VMM; set `progressive_memory=False`
explicitly for fully committed backing.

## <a id="chunking"></a>3. Chunk sizing and the one-chunk lookahead

The constructor derives

```
chunk_pages_ = min(max_physical_pages, max(1, target_chunk_bytes / value_page_bytes))
```

so `target_chunk_bytes` (64 MiB at the binding boundary) is a target for the VALUE
plane; the mass plane inherits the same page count and is therefore smaller in bytes.

`grow(required_pages)` returns immediately when the requirement is already met, and
otherwise commits to a chunk boundary with a **one-chunk lookahead**: the target is
`required_chunks + 1` chunks, *except* on the first growth (`current_pages == 0`) and
except when `required_chunks` already reaches the maximum — in both of those cases it is
exactly `required_chunks`. The result is clamped to `max_physical_pages_`.

The lookahead is why a steadily growing cache does not call into the driver on every
admission: it is always one chunk ahead of the request that would need it.

`map_initial_chunk()` is just `grow(min(max_physical_pages_, chunk_pages_))`, so the
first chunk exists before any caller sees the object.

## <a id="growth-atomicity"></a>4. Growth is all-or-nothing across both planes

`grow()` must leave the two planes consistent, because `mapped_capacity_pages()` is
their minimum and a half-grown pair would silently under-report.

The mechanism is a `GrowthMark` journal: `grow_plane` pushes the plane's pre-growth
`chunk_count` and `mapped_bytes` before it maps anything, and on ANY exception `grow()`
calls `rollback_growth(mapped_now)`, which walks the marks in reverse, unmaps and
releases every chunk past the recorded count, and restores each plane's `mapped_bytes`.
`committed_bytes_` is restored from a saved copy on the same path.

`grow_plane` has its own inner unwind for the window where a single chunk is
half-installed: `cuMemCreate` has returned a handle but `cuMemMap` or `cuMemSetAccess`
has failed. Its `catch` unmaps only if the map actually succeeded, releases the handle,
and rethrows — leaving the plane's `chunks` and `mapped_bytes` untouched, which is what
makes the outer journal's bookkeeping correct.

Note the ordering inside `grow_plane`: `plane.chunks.push_back` and the
`plane.mapped_bytes` update happen LAST, after access is granted. Every earlier failure
therefore observes a plane that never saw this chunk.

## <a id="rollback"></a>5. `rollback_to` — quiesce, validate, then unmap

Shrinking is far more dangerous than growing, so the method refuses everything it
cannot prove:

1. **Quiesce first.** `cudaDeviceSynchronize()` must succeed before anything is
   unmapped. In-flight work may still hold addresses in the range being removed.
2. **The target must be a committed CHUNK BOUNDARY.** `validate_boundary` walks the
   plane's chunk list accumulating byte offsets and requires the target to equal one of
   them (or be zero). A target in the middle of a chunk is rejected rather than rounded
   — chunks are the allocation unit, and there is no way to release half of one.
3. **The initial headroom chunk is never removed.** `target >= plane.chunks.front().bytes`
   is checked for the same plane, so the owner always retains the chunk
   `map_initial_chunk` installed.
4. **Both planes are validated before either is touched**, so a bad mass target cannot
   leave the value plane already shortened.

`remove_latest_chunk` then re-checks, per chunk, that the chunk being removed actually
ENDS at the plane's mapped extent — a defence against a chunk list that has drifted out
of address order — and `restore_plane` asserts the plane landed exactly on the target
rather than merely at or below it. The method closes by asserting
`mapped_capacity_pages() == target_pages`.

Unlike growth, rollback's driver calls go through `check()` (which throws) rather than
being swallowed: a failed unmap during a deliberate shrink is a real error, not a
teardown nicety.

## <a id="view-lifetime"></a>6. View lifetime — why `close()` can refuse

Every tensor `make_view` returns carries a deleter that captures two things: the
shared `ViewTracker` (an atomic `live_views` counter) and `shared_from_this()`.

The `shared_from_this` capture keeps the driver allocation alive for every tensor view.
**This prevents a caller holding a slice from outliving the native owner and observing
an unmapped raw pointer.** It is the reason `VmmOwner` derives from
`enable_shared_from_this` and the reason the pybind class is held by `std::shared_ptr`
(see [`../rola_api.md`](../rola_api.md#vmm-binding)).

The counter is the other half: `close()` `TORCH_CHECK`s that `live_views == 0` and
refuses to proceed while any exported view exists. Lifetime extension alone would let
`close()` unmap under a live view; the counter turns that into a loud error at the call
site instead.

The increment happens BEFORE `at::from_blob`, and the `catch` decrements it if
construction throws — so a failed view never leaks a count that would permanently block
`close()`.

`reset()` is **intentionally logical only**: mappings and their stable addresses
survive. Resetting the cache's logical contents is `SparseRoLAState`'s business, per the
scope rule at the top of this doc.

## <a id="destruction"></a>7. Destruction leaks rather than races

`~VmmOwner` runs `release_planes()` only if `quiesce()` (a guarded
`cudaDeviceSynchronize`) succeeds.

**Never unmap while work may still reference the reserved ranges.** If the context is
already unavailable — process teardown, a device reset, a CUDA error latched earlier —
leaking the mapping is safer than tearing it down under in-flight CUDA work; process
teardown reclaims it either way.

The whole body is exception-swallowing by necessity: destructors cannot report driver
failures. `release_planes()` is `noexcept` and unmaps chunks in REVERSE order before
freeing the address reservation, mirroring the order they were mapped in.

`close()` is the path that *can* report: it quiesces, releases, and marks `closed_`, and
every mutating method `TORCH_CHECK`s `!closed_` afterwards.

## <a id="overflow-discipline"></a>8. Arithmetic that cannot silently wrap

Every size computed from user-supplied dimensions goes through a checked helper, and
each takes a `what` string so the failure names the quantity:

| helper | guards |
|---|---|
| `checked_mul` | `uint64` multiply overflow |
| `checked_add` | `uint64` add overflow |
| `checked_size` | `uint64` value exceeding host `size_t` |
| `round_up` | `value + alignment - 1` overflowing `size_t`, and a zero alignment |

The page-byte derivations in the constructor (`bp * dqk * dv * sizeof(float)` and
`bp * dqk * sizeof(float)`), the virtual extent in `reserve_planes`, and the growth and
rollback targets all pass through them. Page geometry arrives from Python; a wrapped
size here would reserve or map the wrong extent with no diagnostic, which is why the
checks are unconditional rather than debug-only.

`driver_error()` renders any `CUresult` as `NAME: description` via `cuGetErrorName` /
`cuGetErrorString`, falling back to `CUDA_ERROR_UNKNOWN`, so every `check()` failure
carries the driver's own words.

`initialize_driver()` calls `cuInit(0)` under a `std::call_once` and re-checks the
STORED result on every subsequent call — so a failed init reports on each entry rather
than being silently skipped after the first.

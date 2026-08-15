# `rola/ops/paging.py` — page commitment, keyed to the MMA atom

Moved from the module docstring; the module keeps the vocabulary and the
one-sentence purpose.  Companion: `vmm_owner.md` (the VMM backing).

## What Changed, And Why It Is Not A Tuning Choice

``page_states := MMA_K_QUANTUM`` (was ``BC``). The ruling: **``BC`` is a re-derivable
COMPUTE tile; a page is a MEMORY-RESIDENCY atom.** Their consumers are different
objects with different lifetimes -- admission bitmaps, arena pools, residency ledgers,
warm reuse across sequences -- and every one of them must survive a ``BC`` change
unbiased. Since the planner chooses ``BC`` PER LAUNCH, a ``BC``-keyed page
would have to be rebuilt whenever the planner moved it, which is exactly the
arena-rebuild class the re-key kills.

Every legal ``BC`` is a whole number of atoms by construction (``16 | BC``), so an
owner spans exactly ``QPO`` atoms and a CTA resolves ``QPO`` slots at startup instead
of one page. Residency tests move to atom granularity -- the SAME atom as the
emptiness-skipping quantum -- so ONE predicate serves both.

**THE ADDRESS.** Logical atom id ``bh * (N / MMA_K_QUANTUM) + leaf // MMA_K_QUANTUM``;
``table[atom id] = slot | ABSENT``; the value address is
``((slot << LOG2_ATOM | row) * cols + v)`` with ``row = leaf & (MMA_K_QUANTUM - 1)``.
Both shifts are DERIVED from the named constant (invariant I1) -- see
:data:`_LOG2_ATOM_LEAVES` -- so there is no ``4`` and no ``15`` anywhere in this file.

**WHO RESOLVES IT.** The chunk consumer, at the two ends of its state I/O and
nowhere else -- that kernel touches pages only there, so the resolve is one
``int32`` load per atom rather than a per-CTA-startup cost. It resolves
``slot = page_tbl ? page_tbl[atom] : atom``, which makes the DENSE plane the same
kernel with a null table (the slot IS the atom), and paging ONE base translation
rather than a second path. ``slot < 0`` -- an atom the plan committed nothing for
-- is skipped at both ends, exactly and not defensively, because the plan's bitmap
is the write set exactly (``stats.md``). ``chunk_kernel.md`` carries the
addressing, the CTA-uniformity of the barriers around that skip, and the 4 GiB
plane bound the 32-bit displacement form owes the caller.

## Allocation: Planned-Exact-Async

The pool is sized EXACTLY, not grown: the popcount of the atom condensation (the producer epilogue's canonical write bitmap) is the
number of atoms this call will touch, So the arena knows its own footprint before the
consumer launches, and there is nothing to discover.

Allocation and zero-init issue on a SIDE STREAM as soon as the epilogue lands, and the
consumer launch waits on ONE event. That is the whole of the asynchrony: no polling, no
second sync, and the zeroing overlaps the schedule build rather than serializing with
it.

**THERE IS NO CAP, AND THAT IS THE POINT.** With the footprint known exactly, an
allocator ceiling is either never-binding or a self-inflicted refusal, so the arena
carries none: its capacity IS the dense limit and it COMMITS only what the plan
admits — the limit is RESERVED virtually and committed physically per plan, which is
[the two backings](#backings) below. The former ``CAPPED`` mode (partial grant plus
an admit mask that annihilated the denied atoms' writes) was DELETED before it: it
computed a DIFFERENT function from the one the caller asked for, and the closed-world
standard is that a demand the binary cannot serve is refused at the host rather than
silently degraded. The ``max_physical_pages`` parameter that survived it is deleted
with the paged-state-I/O integration (P67 S3b).

The consequence downstream is that admission is TOTAL: every atom the routing realizes
is resident, so ``admit bit set``, ``resident`` and ``page_table[atom] >= 0`` are one
predicate and the arena publishes exactly one of them (the table).

## <a id="backings"></a>The Two Backings, And The Receipt

Paging's claim is a claim about PHYSICAL MEMORY, and a pool allocated at the dense limit
satisfies every equivalence gate while saving none of it. So the arena's device backing is
the CUDA driver's virtual memory manager ([`vmm_owner.md`](vmm_owner.md)): the dense limit
is RESERVED as address space, which costs nothing, and physical memory is committed as
:class:`ExtentAllocator` hands out slots. ``PageArena.committed_bytes`` reads that
commitment out of the owner's own accounting rather than out of ``nvidia-smi``, which
reports the caching allocator's reservations too and cannot attribute bytes to one arena.

``backing='dense'`` is one ``torch.zeros`` over every slot the topology could need. It is
the HOST backing — a CPU device has no driver VMM at all — and the instrument the
equivalence gates measure the VMM arena against. It is not a fallback, and the difference
matters: ``backing='auto'`` (the default) resolves to the VMM backing on CUDA and RAISES on
a CUDA device whose driver cannot serve it, naming the driver's own reason. A measurement
must not be able to quote the dense allocation as the feature.

The capability is a device fact, proven once per device index by the driver round trip
:func:`vmm_supported` runs and cached (``rola.ops.paging._VMM_PROBE``) — never re-asked per
arena.

**COMMITMENT FOLLOWS THE CURSOR, NOT THE RESIDENCY.** The two differ: the residency is the
page table's popcount and the cursor is what :class:`ExtentAllocator` has handed out,
including the alignment gap it opens when a batch does not fit an extent's remainder. The
cursor is a host integer and the residency is a device reduction, so the cursor is what the
arena's own logic runs on and what sizes the commitment; ``allocated_pages`` is that
number, and ``resident_pages`` is the exact one, a REPORTING surface that syncs and never
appears on a measured path.

## Extents Are Allocator Policy And Nothing Else

The pool is carved into runs of ``C_EXTENT_ATOMS`` consecutive slots, and atoms
admitted together take consecutive slot ids -- so an owner's ``QPO`` atoms land
contiguous in the common case and the CTA resolves ONE base instead of ``QPO``
independent lookups. This is INVISIBLE to the table format and to the kernel: the table
is still ``atom -> slot``, and a caller cannot tell an extent-allocated arena from a
first-fit one except by reading the slot ids. Transfers batch at extent granularity.

## Scope

The arena stays CALL-SCOPED by default (``fresh=`` contract unchanged):
:meth:`plan` resets before every call unless the caller opts into continuation.
:meth:`adopt_dense` / :meth:`materialize` remain the explicit, lossless bridges to the
dense cache contract.

Arena exclusion is respected literally: the arena holds persistent cross-chunk
state pages ONLY.

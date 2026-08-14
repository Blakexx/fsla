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

## Allocation: Planned-Exact-Async

The pool is sized EXACTLY, not grown: the popcount of the atom condensation (the producer epilogue's canonical write bitmap) is the
number of atoms this call will touch, So the arena knows its own footprint before the
consumer launches, and there is nothing to discover.

Allocation and zero-init issue on a SIDE STREAM as soon as the epilogue lands, and the
consumer launch waits on ONE event. That is the whole of the asynchrony: no polling, no
second sync, and the zeroing overlaps the schedule build rather than serializing with
it.

**THE CAP IS A POLICY CEILING, AND EXHAUSTING IT IS A REFUSAL.** With the footprint
known exactly, the allocator cannot be surprised; ``max_physical_pages`` stops being a
capacity guess and becomes a deliberate limit. When a launch demands more atoms than
the ceiling leaves, the arena RAISES, naming the demand and the arena size -- it never
grants a prefix and never drops a write. The former ``CAPPED`` mode (partial grant plus
an admit mask that annihilated the denied atoms' writes) is DELETED: it computed a
DIFFERENT function from the one the caller asked for, and the closed-world standard is
that a demand the binary cannot serve is refused at the host rather than silently
degraded. With it goes the old annihilation contract, which existed only to define what
a denied write meant.

The consequence downstream is that admission is TOTAL: every atom the routing realizes
is resident, so ``admit bit set``, ``resident`` and ``page_table[atom] >= 0`` are one
predicate and the arena publishes exactly one of them (the table).

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

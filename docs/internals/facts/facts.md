# `csrc/rola/src/facts/facts.cu` — the facts layer's dispatch and torch entries

Adopted verbatim from the deleted chunk consumer's dispatch TU (`chunk.cu`, K31
deletion batch; baseline = tag `baseline/pre-k31`). Three entries cross the
extension boundary — `facts` (the union table and, on a stateful call, the
atom-grain write bitmap: one pass, both row sets), `block_bits` (the liveness
epilogue's reduction over a table this call already built) and `arms` (the
built `(C, BC, D, B)` dispatch vocabulary with the level plan's quantities).

<a id="closed-world"></a>**`check_arch_table` — the closed-world R4 runtime
half.** Its compile-time half is `arch_caps.cuh`'s
`static_assert(tabulated(kArch))`; this is the refusal at first launch on a
device the table has no row for, or a row the driver disagrees with. Every
launching entry runs it first — `tests/unit/test_closed_world_arch_gate.py`
gates that per entry.

<a id="l-bound"></a>**The bound on `L`** is retained from the consumer era
(`kMaxGatherL`): the widest gathered row the family ever addressed was the fp32
output at `d_v * 4` bytes, and the facts tables were built for exactly the
launches that bound admitted. K31 R2 restates it from its own geometry.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/facts/facts.cu`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kd"></a>
### `kD`

THE BOUND ON L, retained from the consumer era: the widest gathered row the
family ever addressed was the fp32 output at d_v * 4 bytes, and the facts
tables were built for exactly the launches that bound admitted.  K31 R2
restates it from its own geometry.

<a id="check-arch-table"></a>
### `check_arch_table`

THE RUNTIME HALF OF THE CLOSED-WORLD ARCH RULE (setup.py's R4).  Its
compile-time half is `arch_caps.cuh`'s `static_assert(tabulated(kArch))`,
which fixes the table this binary's shapes were DERIVED from; this is the
refusal at first launch on a device that table has no row for, or has a row
that the driver disagrees with.  Both halves must stay: a derivation is only
as good as the facts it was closed over, and only a fact read off the DEVICE
can establish that this is the device they were read for.
docs/internals/facts/arch_caps.md

<a id="check-arch-table-note-l41"></a>
### in `check_arch_table`, near line 41

ONCE PER PROCESS PER DEVICE, on the first launch rather than at import: a
CUDA context as a side effect of `import` costs more than the check.  The
device is recorded only after it passes, so a refusal is raised on every
call rather than once.

<a id="build-facts"></a>
### `build_facts`

THE FACTS, AS TENSORS: the union table `[B*H, 2, entries, L/32]` of packed bits,
sides apart because the cook wants them apart, and -- when this call has
residency to plan -- the atom-grain write bitmap `[B*H, N/16]` the page plan
admits from.  ONE PASS over the amplitude planes emits both row sets; the
atom-grain TABLE is a transient, allocated and freed here, because nothing else
asks for it.  The bitmap's own launch stays keyed `(D, B)`, so the published fact
is BC-free however this pass was templated (docs/internals/paging/stats.md).

<a id="is-arm"></a>
### `is_arm`

THE ARGUMENT CHECKS OF A LAUNCH, SHARED BY THE TABLE AND THE KERNEL so that
the two can never be asked for different problems.
THE WIDTHS ARE A LIST -- the API speaks the widths policy's language even
though every built arm is uniform.  A caller with a jagged topology gets
the unbuilt-arm refusal, never a silent reshape.

<a id="page-table"></a>
### `page_table`

THE ATOM -> SLOT MAP, `[BH, N/16]` int32, `-1` where the atom has no page.
Its ABSENCE is the dense backing, where a slot is its atom -- one translation
for both (docs/internals/paging/paging.md).  The slots it names are NOT bounds
checked here: the check is a device-side read of the table, and the plan that
wrote it is the thing that owns the bound.

<a id="build-block-bits"></a>
### `build_block_bits`

THE LIVENESS EPILOGUE: `[BH, N/BC]` uint8, bits {READ, WRITE, STATE}, reduced
from the union table THIS launch already built.  A SEPARATE ENTRY POINT, on
the union table's own rule: a per-call reduction that the sparse grid pays
for must be reported on its own line and never folded into a kernel delta.

<a id="near-line-79"></a>
### near line 79

the arm surface, DECLARED only -- definitions live in the instantiation
shard (facts_s0.cu).  facts.cu must never define one locally.

<a id="checkplanes"></a>
### `check_planes`

THE AMPLITUDE HALF, which the facts pass needs and the consumer needs too: the
planes carry the problem's `[B, H, L]` addressing, so nothing about the token
schedule has to be read off the value stream.

<a id="chunkblockbitsx"></a>
### `CHUNK_BLOCK_BITS_X`

NO `C` AXIS: the table this reduces is the (D, B, BC) one, so the arm walk
matches on the block grain alone and an arm list repeating a grain at
several chunk lengths resolves to the one instantiation.

<a id="arms"></a>
### `arms`

The built set with the plan quantities of an arm reported beside it: the
block count, the block's inner/outer axes and the table's entry count.  The
consumer's launch geometry (staging depth, CTA target) died with it.

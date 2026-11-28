# The carried state, and the facade that engages it

Mirror doc for `fsla/_state.py` and for `fsla.prefill`'s stateful half in
`fsla/interface.py`. It is the contract the repository carries: the design ledger
that produced it is a workflow record, and this file is what a reader ten years from
now is owed.

Vocabulary, defined here and used unqualified below: `N` = leaf count;
`cols` = `d_v` plus one mass column (the ratio readout's denominator);
`BH` = `batch * heads`; **atom** = 16 consecutive leaves in the state's leaf order ([§2](#level-order)),
which is the page granule.

## <a id="interface"></a>The interface and its one implementation

The state is an INTERFACE, `fsla.FSLAState` (an abstract base in `fsla/_state.py`), and v1 has ONE implementation,
the dense state, `DenseState`, built by the public constructor `fsla.dense_state(...)` from the routing and the call's
shape ([§2](#binding); the owner's ruling, recorded in
[`DELETIONS.md`](DELETIONS.md): v1 targets the dense state only, and the state is an interface with one
implementation per form, which each takes whatever parameters it needs). The interface below is the one built, a
state that binds at its first call; construction from the routing replaces `bound` and the binding seams. The interface is:

| a caller may ask | |
|---|---|
| `format`, `bound`, `shape`, `device`, `dtype` | the descriptor, whether a call has bound it, `[B, H, N, cols]`, the bound device, the storage's dtype |
| `carries` | whether anything is stored yet |
| `materialize()`, `clone()` | the logical tensor for the oracle and a test; an eager deep copy |

and the seam the ops take the plane in and hand it back through, private to the library: `_kernel_entry(routes, *, d_v)`
binds the state to the call (or checks the call against what it is bound to) and returns the plane the kernel reads,
`_kernel_commit(plane)` takes the plane a call that bound a FRESH state stored into, `_require_carrying()` refuses a
decode step on a state with nothing to continue from, `_decode_entry(routes)` returns the plane a decode step
updates in place, and the layer's segmentation takes the last two ([the view](#view)): `_bind_plane(*, batch, heads, widths, d_v, device)`
binds the state from a call's shape facts alone and holds a zeroed plane where it carries nothing, and `_view(batch, heads)` is a state
over a range of batch-heads sharing its plane. Everything that handles a state holds the interface; the dense state is the one thing that is
constructed. `tests/unit/test_state_interface.py` pins the interface as built, a dense state constructed with no argument; NOT
BUILT: the constructor's routing and shape ([§2](#binding)).

The dense state takes no storage parameter -- only the routing and the shape -- because each of the three it had was paging's, or had one built value: `pool_slack` and
`backing` belong to a page arena, and `strategy` named a copy-on-write form that is future work with it
([§9](#future-work)).

## <a id="one-object"></a>1. A tensor, and one predicate

```python
s = layer.new_state(B, device)          # the layer's routing, write heads and value width; its plane zeroed
y, s = fsla.prefill(routes, values, masks, density, state=s) # checked against the state's format; s is mutated
y, _ = fsla.prefill(routes, values, masks, density)  # stateless: (y, None)
```

`fsla.dense_state` IS THE CONSTRUCTOR, unambiguously: the interface, the implementation and the function live in
`fsla/_state.py`, private by its underscore, so no module answers to the public name.

A `FSLAState` is a tensor and nothing more:

| it holds | what it is |
|---|---|
| storage | one dense plane, `[BH, N/16, page_bytes]` uint8 (`DenseState`) |
| the logical shape | `[B, H, N, cols]` (`state.shape`) |
| device, dtype | `state.device`, `state.dtype` — the STORAGE's |
| the format | `state.format`, the descriptor of [§2](#format) |

**THE MEMBERSHIP TEST is a question, and it is the whole design rule: would a tensor
know this?** A tensor knows its layout and its contiguity. A tensor does not know which routing widths produced it,
whether a convolution ran before it, or what launch scratch some kernel wants — so the widths/normalization
fingerprint, the decode carrier and the decode workspace are all GONE from here, each
to the level that owns it ([§5b](#decode-seam), [§7](#conv-rings)). Accretion onto a
carried state is the failure mode this rule exists to leave nowhere to land.

**The predicate is `state is not None`**, at the op and at the layer alike. There is
no second flag, because the kernel's store epilogue is skipped by a NULL POINTER: one
compiled body serves every call, and a stateless call's entry/exit sweeps are gated
at RUNTIME by a CTA-uniform `!= nullptr` check on the plane pointers. AN INSTANTIATED STATE IS
ALWAYS read at entry and updated at exit. NULL
means no state object at that end -- nothing is read, written, or allocated there.

## <a id="binding"></a>2. The format is fixed at construction, and there is no interview

A state is CONSTRUCTED FROM THE `fsla.Routing` AND THE CALL'S SHAPE -- the batch, the write heads, the value width --
and its device ([`../api.md`](../api.md#the-state)): `[B, H, N, cols]` and the format are taken there, the plane is
allocated and zeroed there, and nothing about the format is learned from a call. `cols` needs `d_v`, which is NOT on
the routing -- the value width is the value stream's own fact -- so it is the constructor's argument, and the layer's
`layer.new_state(batch, device)` supplies its own. A layer or op whose routing or shape differs from the state's
refuses it. The plane's contents advance in place each call; the format never changes. NOT BUILT: `fsla/_state.py`
still builds an unbound state that binds `[B, H, N, cols]`, its format and its device at the first call that receives
it (`_kernel_entry`, `_bind_plane`), taking `d_v` from `v.shape[-1]`.

<a id="format"></a>**THE STATE OWNS ITS FORMAT**. What its construction fixes is a FORMAT DESCRIPTOR (`state.format`, a
frozen `StateFormat`), and it is what every kernel derives its addressing from:

| field | what it is |
|---|---|
| `D`, `B` | the routing depth and the PER-LEVEL widths `B_l`, each a power of two at or above 16 (the axis law: the floor is `K_max`, which keeps every span inside one level's run and the atom inside one level) |
| `order` | the leaf order, MSB-first mixed radix over the levels in the STATE'S LEVEL ORDER ([below](#level-order)), and a FIELD rather than an assumption, so a second order is a refusal and not a silent reinterpretation of somebody's bytes |
| `page_bits` | the page rectangle: the trailing `log2(16)` bits of that order, 16 digits of the storage-innermost level. The page IS that rectangle |
| `DV` | the STORED value width: `d_v` rounded up to the format's 32-channel group (one exponent a group) and no further; `cols = DV + 1` is the page's LOGICAL width, not its layout ([below](#stored-width)) |
| `dtype` | `"s24 hi/lo planes at an s8 group exponent, fp32 mass"` — the int8 state format ([section 3](#int8-state)); no carry kernel is built |
| `ids` | the page id space, `BH * N / 16` |

<a id="level-order"></a>**THE STATE'S LEVEL ORDER IS SET FROM THE ROUTING'S STATIC FACTS, BEST EFFORT, UNDER ONE HARD
RULE.** The hard rule: the carve gives every CTA at least 16 digits of the storage-innermost level, so a CTA's leaves
are whole pages. Every level is at least 16 wide and every side's leaf group at least 16 leaves. The best effort: the
levels are ordered by how many parties -- the write side and the `G` read heads -- read them dense, the fewest
outermost and ties in the declared order, so the innermost level is the one the carve keeps wide anyway: under R20 a
doubling on a level saves a fetch for each side whose size there still exceeds its share, so a level many parties read
dense draws the carve's doublings and the 16-digit floor on it rarely binds; and a token's live leaves lie in long
contiguous runs. The order is recorded in the format. THE ORACLE AND `materialize()`
REPORT LEAVES IN THE DECLARED LEVEL ORDER: the permutation the storage order applies is undone in those two places
only, and nowhere else. NOT BUILT: the order today is the declared one.

<a id="stored-width"></a>**THE STORED WIDTH IS DECOUPLED FROM EXECUTION.** The state stores `d_v` rounded up to the
format's 32-channel group and no further. A kernel whose tile is wider -- the prefill's 64 channels on the MMA's M
axis -- pads in registers or shared memory at load and stores back only the stored channels, so one stored width
serves the prefill and decode and a prefill's state hands to decode and back; the padding costs the wide kernel some
idle MMA rows at a small `d_v`, never a stored byte.

**PREFILL AND DECODE SHARE ONE ADMISSION LAW: THE DESCRIPTOR** (Blake, 2026-08-29).
Decode accepts exactly what prefill accepts — the same `D`, the same per-level `B_l`
(powers of two at or above 16), the same `N = prod_l B_l`, the same `DV`, the same operand format,
and therefore the same topologies. There is no decode-only shape, no decode-only
tolerance and no decode-only admission branch anywhere, tests included: a second
admission law is a second contract to keep in step with the first, and the two kernels
partition ONE state.

**Later calls are CHECKED against it, and a mismatch is a refusal at the seam naming
the field that differs** — at `_kernel_entry` for a prefill call, at `_decode_entry` for a
step. Two states of the same `N` and `cols` built from DIFFERENT widths have different
leaf orders, and interchanging them is refused, because the leaf order, the page rectangle and the value width are
precisely what a
kernel addresses bytes with — a state that disagrees with the call is not a risk for the
caller to own, it is a wrong address. Resubmitting an older state, sharing one between
two sequences and continuing after a `clone` stay legal and the caller's semantics to
own. The descriptor is fixed for the state's life and travels with the
`LayerContinuation`; a re-layout is an explicit state op, never something a call does.

**`tests/unit/test_state_contract.py` PINS THIS CONTRACT WITHOUT A KERNEL BODY**: the seven-field list above, each
 field's own
refusal, and that `_kernel_entry` (prefill) and `_decode_entry` (decode) both build the
call's presented format through `_format_of` and check it through `StateFormat.check` --
the same two calls, never a second comparison at either seam, which is the ADMISSION LAW
paragraph above made structural.

**PADDING KEEPS THIS DESCRIPTOR AT SEVEN FIELDS.** The
logical-vs-padded split (`b_l` the caller's width, `B_l` the padded, shipped one) lives
at the op call surface (`fsla.ops.padding`, `docs/internals/padding.md`), never here:
`StateFormat.B` still carries only the lawful, padded width, and `_format_of` still
builds it from whatever widths the presented routing carries. A caller reaches this
descriptor with an already-padded `RouteFactors` (the layer producer's
own bundle stays at the caller's logical widths, unpadded, by a DIFFERENT, pinned
contract -- `tests/unit/test_api_contract.py`), so `check()`'s refusal message names
padded numbers on both sides, never a logical one -- a genuine limit: two different
logical widths that pad to the same `B_l` are indistinguishable to this check. That
limit is stated, not hidden, in `docs/internals/padding.md`.

## <a id="dense-state"></a>3. The dense state

One `[BH, N/16, page_bytes]` plane: the page sequence in the state's leaf order with every page resident, addressed by the
atom index. The state's storage is that plane and nothing else, and it is not a per-call flag. The plane is ZEROED
where it is made: the exit sweep stores the atoms a call WRITES and no others, so the atoms it skips must already hold
the zero state.

<a id="int8-state"></a>**THE PAGE IS THE INT8 STATE FORMAT** (ruled 2026-09-28), A page — one
atom, 16 leaves — holds

    [16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32]

a leaf-channel the s24 `hi << 8 | lo` at its (leaf, 32-channel group)'s power-of-two exponent, the mass fp32:
about 3.03 bytes a leaf-channel against fp32's 4, fp32-class precision (a 24-bit mantissa, a shared exponent a
group). The plane is `[BH, N/16, page_bytes]`, one page an atom; a read that needs only the s16 operand touches the hi plane alone. The geometry and the codec
live in [`common/state_page.md`](common/state_page.md#geometry). No carry kernel is built, so a stateful prefill is refused by name.

THERE IS NO RAGGED STATE (Blake, 2026-08-29). `N = prod_l B_l` with every `B_l` a power
of two at or above 16, so a lawful state is PAGE ALIGNED BY CONSTRUCTION, and a leaf
axis that is not a whole number of pages is REFUSED — not padded, not carried in a tail
page, not served by a second layout. The refusal is the descriptor's and the seam's, and
it is one sentence rather than a branch.

THE CONTAINER STAYS `float32`, because the LOGICAL element is fp32. It is not a float
ARRAY any more, though — a 32-bit word of a stored page is two neighbouring elements'
halves — so a byte claim about pages is asserted on the integer view
(`fsla.ops.paging.bytes_equal`), never with `torch.equal`, which is a float comparison
and would call a byte-identical pair unequal the moment a word happened to read as
NaN. `materialize()` is the hi||lo GATHER back to the logical `[B, H, N, cols]`, its leaves in the declared level
order ([the level order](#level-order)), which is what the
oracle, a cross-arm hand-off and a test read, never a measured path, because it rejoins every page on the host.

**There is no activity fold.** Every atom of the plane is resident, so the prefill is to load and store every page of a
block it runs and folds no per-atom activity byte from the masks; which blocks run is the run table's decision
([`engine/rules.md#run-table`](engine/rules.md#run-table)). What a call reads of an atom is the call's own question: a
decode step pulls the hi plane alone for a leaf it only reads ([the page](#int8-state)). THE PRICE IS THE IN-PLACE LAW: an
atom a sweep skips must already hold what it carried in, so a continuation's exit plane must BE its entry plane.

## <a id="kernel-call"></a>4. What the state contributes to a kernel call

The ORDER of a stateful call is the prefill call's and is stated there
([`engine/prefill_call.md`](engine/prefill_call.md)). What is the STATE's own
is this: the call's entry checks the call's routing and shape against the state's format and hands back the plane, zeroed
at construction for a state that carries nothing yet, and the launch advances it in place. NOT BUILT: today's
`_kernel_entry` binds an unbound state's shape, device and format at the call and hands back `None` for a state that
carries nothing, and `_kernel_commit` takes the final plane for a state that just bound.

### <a id="four-shapes"></a>The kernel's four shapes, and no plane the caller did not make

`state_in` and `state_out` are two references the caller hands the kernel, and every
layer above it (`fsla.ops.prefill.prefill`) passes them
through and allocates nothing (ruling 2026-09-08). Neither is the NULL-STATE call:
nothing comes in, nothing goes out, no sweep runs. `state_in` alone is READ-ONLY: the
readout against a frozen state, the folds accumulate in registers and are never stored.
`state_out` alone is a FRESH sequence: a zero state, no entry sweep, the box stored whole
at exit. Both, the SAME tensor, ADVANCE the plane in place. The plane is the whole state,
`[BH, N / 16, page_bytes]` in the state's leaf order, and its extent is the state's format, checked against the call. The state lives in the
register file between the two sweeps -- the exit sweep is a whole store of the box's
written atoms, never a read-modify-write of bytes -- which is why the out plane needs no
prior contents and why in and out may be one tensor. A plane comes from
the caller (dense, zeroed).

### <a id="the-dense-continuation"></a>The continuation advances IN PLACE

`s_out` and `s_in` are ONE plane for a continued sequence, so a
stateful chain allocates one plane for the SEQUENCE and not one per call. The design
reason it is worth stating: a second whole-written `[BH, N, d_v + 1]` fp32 plane is
hundreds of megabytes at the parity curve's `N`, it scales with the growth axis, and a
transient whose only job is to be swapped in buys nothing the caller can observe.

**What makes it safe is the CTA structure, not a convention.** The boxes tile the leaf
space exactly, so each atom belongs to exactly one CTA. A kernel with state I/O reads its
box's atoms ONCE, before its first window, and stores them ONCE, after its last, from the
same base, so no CTA reads an atom another CTA may already have stored and no CTA reads
its own after storing it. That is the contract the carry's entry and exit sweeps owe;
no carry kernel is built ([`docs/open-work.md`](../open-work.md)). The decode
step updates in place under its own argument: its write positions partition the leaves
([`decode/decode.md#determinism`](decode/decode.md#determinism)).

**The cost is that no stateful call is transactional.** A launch that faults part way
through the grid leaves the plane part advanced. Recovering a
pre-call state is `clone()` before the call, and a caller who needs that guarantee is
paying for the second plane deliberately rather than on every call.

Two consequences follow and are stated where they belong: a continuation can SKIP
blocks, because a skipped block leaves its own atoms carrying what they held
([`engine/rules.md#run-table`](engine/rules.md#run-table)), and a fresh sequence can too, since its plane was zeroed at
construction and a skipped atom already holds the zero state.

### <a id="view"></a>A view over a range of batch-heads

The layer cuts a call into head blocks ([`../api.md`](../api.md) section 4.2), and a stateful call over more than one block runs each block on
a VIEW of the caller's state. The plane is batch-major, a row a batch-head, write heads inside a sequence, and a block is whole sequences or a
range of one sequence's write heads, so its rows are ONE CONTIGUOUS RUN: sequences `[b0, b1)` and write heads `[h0, h1)` of a `[B, H]` state
are rows `[b0 * H + h0, (b1 - 1) * H + h1)`. `_view((b0, b1), (h0, h1))` is a `DenseState` whose plane is that slice of the caller's (no copy),
whose shape is `[b1 - b0, h1 - h0, N, cols]` and whose format is the caller's with its page ids, `ids`, restated for those rows; a block whose
rows are not one run, or a state that carries nothing, is refused. A call on the view is an ordinary continuation of a state of that shape:
`_kernel_entry` checks the block's routing against the view's format and hands back the slice, the launch advances it in place, and the caller's
plane holds the update. The views of a call's blocks partition the rows, so no two blocks touch one row.

The caller's state already covers the whole call: it was constructed from the routing and the call's shape facts alone (the batch, the
write heads, the value width and the device, [§2](#binding)), never from a whole-call produce, which is the bundle segmentation exists to
avoid. Its plane was zeroed at construction, so a block's first step on a state that carries nothing continues from the zero state, which is
the fresh sequence, its skipped atoms already holding zero ([the continuation](#the-dense-continuation)). The state is checked against the
call, and a carried plane is kept. NOT BUILT: the layer binds an unbound state to the whole call first (`_bind_plane`).
`tests/unit/test_state_view.py` pins the view on the host: its rows, shape and format, that a write through it lands in the plane, its
refusals, and the bind.

## <a id="arms"></a>5. There is one arm, and no bridge

The fp64 oracle is the executable SPEC, not an arm of the op
([`api.md` §1.1](../api.md#11-the-built-envelope-is-the-surface)): a caller that wants
it calls `naive_fsla` and passes raw tensors. So no state ever crosses to it through
the facade, and the dense-contents bridge that existed to carry it (`_dense_entry`,
`adopt_dense_contents`) is GONE
(`DELETIONS.md`). What is left is one kernel arm reading the state's plane.

## <a id="decode-seam"></a>5b. The `T = 1` seam

`fsla.decode_step_gated(routes, values, masks, state, decay=...)` takes and
returns the state exactly as `fsla.prefill` does. Two things live
behind that seam:

* **The descriptor and the scratch** (`Descriptor`, `DecodeScratch`) are the
  decode step's own (the step builds the descriptor from its inputs; the descriptor the scratch's key, the scratch held per sequence), not the state's — launch machinery fails the
  membership test. The ORDER and the holding discipline are that step's page
  ([`engine/decode_step.md` §4](engine/decode_step.md#the-scratch)); what the state owes
  them is only its identity, which is what the scratch table is keyed on.
* **The plane.** `_decode_entry` returns the state's own plane, not a copy, which is what makes the kernel's in-place
  update the state's update, after checking the step's routing against the state's format
  ([`decode/decode.md#determinism`](decode/decode.md#determinism)).

A decode step CONTINUES, so `_require_carrying` refuses a state that carries nothing —
asked once by the decode family's `validate_context` before the step's first launch, and
again at `_decode_entry` for a caller reaching the seam directly:
there is no prefill at one token and no entry state to read. That is a computational
impossibility and not a policy, which is the only kind of refusal this surface has.

## <a id="clone"></a>6. `clone()` is the transition

`clone()` returns an independent sequence:
either handle may be passed into any op with no effect on the other, ever. Today's mechanism is an EAGER deep copy:
the copy gets its own plane. Nothing
is shared, and nothing else is copied — a clone has no rings and no scratch to copy,
because it holds none.

## <a id="conv-rings"></a>7. CONTINUATION IS LAYERED

The op's state is the dense state. The LAYER returns its own thin container,
`LayerContinuation`:

```python
y, continuation = layer(x, continuation=continuation)   # always the container
```

`LayerContinuation.state` is the op's state (or `None` for a fresh sequence) and,
today, the container's only field. `value_conv`/`route_conv` and the short-convolution
rings they carried across a continuation were DELETED at K45 (docs/internals/DELETIONS.md)
— never reviewed as a mechanism, not paper-1 scope — so there is no second field to
describe here now. The container is not deleted with them: a future layer-level
recurrence is a field ON `LayerContinuation`, never on the op's state.

**The membership test, one level up: an object holds exactly the recurrences ITS OWN
level introduced.** Op: the factored sparse state. Layer: `LayerContinuation`. Model: the
per-layer bundles. This is not the old pooled state object recreated — that defect was
layer luggage INSIDE the op's object, and this keeps each object pure while giving
accretion nowhere to land. The container is the contract in BOTH directions, so a bare
`FSLAState` or the retired `(state, conv_rings)` tuple handed to `forward` is refused by
name rather than adopted: accepting a lookalike would make the layer's own fields
silently optional to carry.

## <a id="not-a-kv-cache"></a>8. What is deliberately absent

No `transformers` cache vocabulary (concatenation, eviction, window rolling,
`get_max_cache_shape`): all of it is built for a state that GROWS with the sequence,
and FSLA's does not. No token tally, no per-layer container: one state object belongs
to one layer's sequence, and a stack of layers is a list of them in the caller's hands.
No allocator cap — the plane is the whole leaf space, so a ceiling over it is either
never-binding or a self-inflicted refusal.

The `transformers` / `flash-linear-attention` adapters that presented a foreign cache
to the old per-stack state object were DELETED with it (docs/internals/DELETIONS.md).
A shim onto this surface is a thin wrapper at a model boundary if one is ever wanted;
it is not a second stateful surface, and the layer will not import either package.

## <a id="future-work"></a>9. NAMED FUTURE WORK

Items designed and NOT BUILT. They are here rather than in a workflow ledger
because the repository has to carry its own contracts: a reader ten years from now gets
the design, not an archaeology exercise. None of them is scheduled.

### <a id="paged-state"></a>9.1 A paged state, and with it copy-on-write branching

A second implementation of the interface ([above](#interface)): a state whose resident bytes scale with the atoms the routing
REALIZES rather than with the `N` the topology provisions. It would be built by its own constructor, `fsla.paged_state(...)`,
taking whatever parameters its storage needs, and would keep the page FORMAT of [§3](#int8-state) exactly: a page is 16 leaves,
s24 planes at a group exponent. What it adds is storage: an arena of pages, a `[BH, N/16]` table of slot ids, and a plan of
which atoms a call commits. The ops would match on the state's type where a call builds its descriptor, and the kernel would take
its template for the state type from that descriptor. The decode step's argument list carries no
such descriptor yet, because the dense state is the only form.

Copy-on-write branching is a property of such a state (writes always to fresh slots plus an updated table, so every state ever
returned stays valid and resubmitting an older one is branching), and its settled semantics are: a slot shared by more than one
branch is never written in place; changing the copy-on-write form of a lineage is a full deep copy, so it is a `clone`; the per-step
table copy wants a two-level page table. Eager `clone()` ships first and copy-on-write is its drop-in optimization behind the same contract.

### <a id="state-bound"></a>9.2 Lifting the 4 GiB state-plane bound

The carry family's state addressing is a 32-bit byte displacement from a pinned plane
origin, which bounds a plane at 4 GiB (~1,032,192 pages) and is refused loudly above it.
The dense plane hits it first; the lift is one per-`bh` 64-bit base.

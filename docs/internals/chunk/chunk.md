# `csrc/rola/src/chunk/chunk.cu` — the chunk consumer's dispatch and torch entries

Adopted from the ladder's rung 6G; the ladder repo (rola-wt-ladder) holds the
rung-by-rung derivation and every measurement receipt behind this kernel's
shape.

The TU's one structural rule, inherited from the consumer family: **it never
touches a kernel symbol.** Arms — the launch, the table launch and the census —
arrive as extern template declarations satisfied by the instantiation shards
(`chunk_s*.cu`); even `chunk_build_stamp`'s register/spill census goes through
`query_chunk_census_arm` so this TU cannot re-instantiate a kernel locally and
leave the build green with nothing sharded.

Entries (all bound in `rola_api.cpp` with the `chunk_` prefix, private seam
only): `union_table` (the schedule producer — one packed bit-row per (head,
side, table entry)), `block_bits` (the block-grain liveness reduction over that
table), `forward_raw` (the readout's `d_v+1` columns, undivided), `forward`
(table + kernel + divide), `divide_pass`, `build_stamp` (device-side build
identity: magic + form words, then the census row), `arms` (the built manifest
with derived geometry).

The arm set is the generator-owned `CHUNK_ARMS_X` macro
(`../instantiations/chunk_arms.inc`, emitted by `tools/gen_shards.py` — one
list, one owner): 21 `(C, BC, D, B)` arms — 15 two-level, three deep
((16,16,16) and (8,8,8,8) at matched N=4096, and the BC=32 depth-4 arm that
exercises the three-clause form), and three flat D=1 arms (the benchmark's
in-kernel flat-routing baseline, N=B≤256).  The swizzle axis is RETIRED
(measured −2.1..+2.2%, noise, no winner — one layout ships).

## <a id="readout"></a>The divide is where the readout's layout and dtype are settled

`den[t]` is a sum over the `N/BC` blocks and no block knows the total, so the ratio
cannot live inside the consumer kernel. It is one elementwise pass, and it carries
three jobs the host used to pay for separately.

**The scaled floor.** The gateway hands the read side over with its per-token mass `m`
still on the levels (`docs/api.md` §2.2), so the numerator and denominator this pass
reads are both `m` times the normalized ones. `eps` is a constant of the DEFINITION,
which means it is the one term that does not scale with the ratio — so the mass arrives
as its own operand and the divide is `m·num / (m·den + eps·m)`. That is the defined
readout exactly, not an approximation of it, and the exactness is checkable rather than
argued: on power-of-two operands the scaled divide is the unscaled one bit for bit
(`tests/oracle/test_chunk_op.py`). A null `mass` is the scale 1, which is what every
bundle whose read duties are all declared on the simplex passes.

**The operand's layout and dtype, in the store.** The consumer accumulates `[BH, T,
d_v]` fp32 and `o_proj` wants `[B, T, H·d_v]` in the operand's dtype. Both the
transpose and the narrowing are addressing and rounding on a value already in a
register, so they are the store's, and the two host passes that used to walk the output
again are gone. The narrowing is round-to-nearest, which is bit for bit the host cast it
replaces; `out` is fp32 or bf16 and nothing else, because those are the operand dtypes
the consumer accepts.

## The built matrix IS the kernel envelope (2026-08-14)

`CHUNK_ARMS_X` is not merely what this TU can launch; since the tiled consumer stopped
being reachable from production it is the whole of what the package ships a kernel for.
`rola.interface.envelope_refusal` reads `rola.ops.chunk.unsupported`, and everything
outside the matrix is REFUSED by the clause that names it — the same treatment decay
and an unratified activation get, in the same words, so there is one authority on what
this kernel runs and it is this file's arm list. The envelope's user-facing statement,
including what it costs, is
[`docs/api.md` §1.1](../../api.md#11-the-built-envelope-is-the-surface).

The direct entries below still refuse loudly: a caller who asks for an unbuilt arm BY
NAME gets `unbuilt arm: ...`, never the oracle. Dispatch is not a lie; an override is.

## The closed-world arch refusal (R4's runtime half)

`check_arch_table()` is the first statement of every entry that issues a launch
(`union_table`, `forward_raw`, `stats`, `divide_pass`, `build_stamp`; `forward`
needs none of its own because it is a composition of three of them). It refuses,
once per process per device, a compute capability `arch_caps.cuh`'s `tabulated()`
has no row for, a row whose REQUIRED clauses are false, and a row the driver
disagrees with on the four FACT columns every arm's staging depth, CTA target and
shared-memory fit was derived from.

This is the half of `setup.py`'s rule R4 that cannot be a `static_assert`: the
compile-time half fixes the table the binary was DERIVED from, and only a fact read
off the device establishes that this is the device it was derived for. There is no
single launcher in this TU to hang the check on, so the property is per entry and
`tests/unit/test_closed_world_arch_gate.py` is what keeps a new entry from being
added without it.

The device is recorded as admitted only after it passes, so a refusal is raised on
every call rather than once.

## Decay is not a feature of this kernel (retired 2026-08-13)

The mass-clock decay phase was designed, built and measured on the ladder: the
decay-off specialization was free (−0.7..0.0%), but the mass clock cost
+63–93% wall dense/sparse — the C-deep serial per-column scan is a bare
dependency chain plus one barrier per live chunk.  That price fired the
stop-gate: decay ships NOWHERE in the operator, and a decay-configured call is
REFUSED at the `rola_op` seam by name (the T=1 decode arm does implement decay, and
its envelope is its own).  Reopen candidates, in order of promise: a parallel
prefix across the scan rows, scan/produce overlap, dressing fused into the
produce rendezvous, coarser clock granularity (a semantic change) — any of
them justified only by a quality measurement showing decay matters at scale.

## <a id="run-table"></a>The skip predicate is a host decision

The liveness epilogue (`chunk_block_bits`, reducing the union table this call already
built — `chunk_kernel.md#block-bitmap`) emits three FACTS per `(bh, block)`: `READ`,
`WRITE`, `STATE`. It does not emit a verdict, because the verdict is not a property of the
routing. What a skipped block would have cost differs by call class and by backing:

| call class | rule |
|---|---|
| stateless `y`-only | `R && W` |
| state-in `y`-only, paged | `R && (W \| S)` |
| state-in `y`-only, dense | `R` |
| state-out, paged | `(R && (W \| S)) \| W` |
| state-out, dense | always run |

`run_table(state_in, state_out, paged)` derives a `uint8` TRUTH TABLE over the byte's three
bits by evaluating that rule for each of the eight values, so the rule is written once in
readable form and the mask is derived, never hand-authored. The kernel's whole predicate is
then one shift and one AND, with no branching taxonomy and no call-class knowledge in it.

**Why each row is what it is.** A skipped block's contribution to `y` and to `den` is the
exact additive identity, not a negligible one: its mass accumulator is initialised to `+0.0`
and only ever written inside the write-side gate, and `x + (+0.0)` is the identity in
round-to-nearest for every `x` the zero-initialised accumulator can hold. That is what makes
`R && W` sound for a stateless call — a write-only block folds into the register state and
then discards it, so it is provably inert. A carried entry state breaks that argument, which
is what `S` restores where a page table can witness it; the dense backing has no residency
map, so its predicate degrades to `R` and says so.

**The dense state-out row is the one that cannot skip at all.** The dense `s_out` plane is
allocated uninitialised precisely because it is written WHOLE — the owner blocks tile the
leaf space exactly, so every atom is stored by exactly one CTA. Skipping a block there
would leave that block's atoms as garbage in the returned state. The paged backing is
different and does skip: an unmapped atom is already a no-op in the epilogue, and a mapped
slot either was zero-initialised on admission or is carried from a previous call, which a
skipped block must leave UNCHANGED — which is what skipping does.

**An arm with no persistent variant is refused by name, never downgraded.** Serving the
full grid to a caller who asked for the sparse one is how a measurement gets attributed to a
launch that never ran; `launch_variant` raises instead, and names the arm.

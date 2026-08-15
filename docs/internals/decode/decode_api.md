# `csrc/rola/src/decode/decode_api.cuh` — the decode path's host-facing declarations

The ONLY decode header `csrc/rola/rola_api.cpp` includes. It declares three
symbols and carries no definitions: the two capacity accessors and
`rola_decode_forward`. The decode path itself is documented in
[`decode.md`](decode.md); this file exists for a compilation reason, and the
reason is the whole content of this doc.

---

## <a id="why-split"></a>1. Why it is split from `decode.cuh`

The split looks gratuitous until you hit it.

`rola_api.cpp` is HOST-ONLY and is compiled by the HOST compiler, not by `nvcc`.
`decode.cuh` carries `__device__` helpers — the bitmap funnel shift, the set-bit
boundary cuts ([`decode.md` §14](decode.md#extract-bits)) — that the host compiler
cannot parse. A binding TU that included `decode.cuh` would therefore fail to
compile.

Guarding those helpers with `#ifdef __CUDACC__` would work, and would be worse:
the header would then mean two different things to two compilers, and which
declarations a caller got would depend on which compiler read it. **One header per
audience instead.**

## <a id="bound-surface"></a>2. The bound surface

`rola_decode_capacity()` exists so a Python test can assert the C++ level-width
capacity against `rola.routing.topology.MAX_BRANCH_WIDTH`. A `static_assert`
cannot read a Python constant, so the pair of them is what makes the
producer/decode capacity relation CHECKABLE rather than merely written down.

`rola_decode_producer_width_mirror()` is the second half of that pair; why it is a
separate claim from the capacity is [`decode.md`
§33](decode.md#producer-width-mirror), and the capacity gate itself is
[`decode.md` §6](decode.md#level-width-capacity).

`rola_decode_forward` is the single data-plane entry: it validates, expands the
support, and launches the GEMV. Its argument checks and workspace contracts are
documented with the definition, in [`decode.md`](decode.md#flat-bits).

## <a id="paged-state"></a>3. `page_table` — which backing the state is

The trailing `c10::optional<at::Tensor> page_table` is the whole ABI difference
between the two backings, and it is what the entry validates differently:

* **absent** — `state` is the dense `[BH, N, d_v + 1]` plane and `N` is its own middle
  extent. This is the pre-P67 signature exactly, defaulted so a caller that never
  pages never mentions paging.
* **present** — `page_table` is `[BH, N / 16]` int32 and NAMES the leaf space, so `N`
  comes from it; `state` is then the arena's `[slots, 16, d_v + 1]` plane, which holds
  only the atoms a plan committed.

Slot VALUES are not bounds-checked. That would be a device-side read of the table, and
the plan owns the bound — the same division of labour `chunk.cu`'s `page_table()`
states. The address the kernel builds from a slot, and what a `-1` means, are
[`decode.md` §4](decode.md#paged-address).

The trailing `c10::optional<at::Tensor> gate` beside it is the probe's reduced verdict,
`[1]` int32: nonzero makes the whole step a NO-OP, which is what lets a caller enqueue it
before the verdict is host-visible and, under capture, never read the verdict at all
([`decode.md` §4c](decode.md#the-gate)). Absent — the default — the step always runs, and
the bytes are the pre-P67-a path's exactly.

## <a id="growth-probe"></a>4. `rola_decode_growth_probe` — a precondition, not a mode

The probe is bound as its own entry because it is a different HOST ORDER, not an option on
the step. It runs BEFORE `rola_decode_forward` so the host can learn whether this step's
write set is already resident WITHOUT deriving that set itself. It writes three things and
nothing else — no state, no `y`, no page table: the `[BH]` int32 verdict (`growth`), its
`[1]` int32 reduction (`growth_any`, which the STEP reads as its `gate` —
[`decode.md` §4c](decode.md#the-gate)), and the `[BH, atoms_per_bh]` bool write-atom set
that verdict was read off (`atom_bits`), which is what a growth step then admits from.

Its argument list is the subset of the step's that an expansion needs, plus those
outputs: `amp_write`, the three per-level integer vectors, the `page_table` it tests
residency against, the step's own flat `bits` workspace, `growth`, `growth_any` and the
reducing CTA's `probe_ctr`, and `atom_bits`. There
is no `v`, no read side, no `n_split`, no `eps`, because none of them participates in the
question. `atom_bits` is REQUIRED rather than optional: the kernel must condense to answer
at all, so a caller who declined the buffer would be asking for the same work with the
result discarded.

`page_table` is REQUIRED here where it is optional on the step: a dense backing has no
absent atom to find, so an unpaged caller never asks. The table also NAMES the leaf space
exactly as it does on the step ([§3](#paged-state)), which is what lets the entry validate
`prod(level_widths)` against it.

The kernel and the condensation are
[`decode_expand.md` §7](decode_expand.md#growth-probe); the host order it serves and what
it costs are [`decode.md` §4b](decode.md#growth-probe).

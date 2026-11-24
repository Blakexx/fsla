# `csrc/fsla/src/common/arch_caps.cuh` — the card's facts at the compiling target

Its rows are generated from the registry's device parts ([`hardware.md`](hardware.md)) into `common/hardware.cuh` by the build graph, which checks no copy in (`Caps`, `tabulated`,
`caps_of`): per architecture the INSTRUCTIONS an atom reads and the NUMBERS a plan reads. This file names the compiling
target (`kArch`, `kCaps`). STRUCTURE is not here: which warps do what, the pipeline, the tiling
are a family's schedule, selected by the build's family (`Family`, [`hardware.md#families`](hardware.md#families);
KERNEL_STANDARDS section 8), never derived from capability flags.

<a id="note-l125"></a>
## The row's fields

The generated rows initialize the fields by name, in this order, and an untabulated compute capability gets an
empty row (`tabulated` refuses it first).

| field | kind | what it states |
|---|---|---|
| `bf16_cvt_packed` | INSTRUCTION | the packed two-float to bf16 pair conversion (`cvt.rn.bf16x2.f32`), asserted by the atoms that pack |
| <a id="bf16-simd-arith"></a>`bf16_simd_arith` | INSTRUCTION | `mul.rn.bf16x2`. ptxas rejects it below sm_90 (measured 2026-08-09): Ampere has bf16 conversion and bf16 MMA but no bf16 SIMD multiply, so the product costs 1 FMUL + 1/2 pack an element there against ~1/2 instruction on sm_90 -- the archetypal flag |
| <a id="bf16-simd-fma"></a>`bf16_simd_fma` | INSTRUCTION | `fma.rn.bf16x2`, available at sm_80 (measured 2026-08-28): a product is one instruction through a `-0.0` addend, bit-identical to `mul.rn.bf16x2` (one rounding of the exact product; the addend's sign keeps a signed zero) |
| `cc` | FACT | the compute capability x 10 |
| `regs_per_sm` | FACT | the SM's registers, which the plans read (the state's registers) |
| `smem_per_sm` | FACT | the shared memory an SM holds, in bytes: what a kernel's shared-memory layout is serviced from ([`smem_layout.md`](smem_layout.md)) |
| `smem_per_cta_optin` | FACT | the most shared memory one CTA may reserve, in bytes: the layout's budget is no more |

AN INSTRUCTION FLAG is read by an atom, where only the instruction differs between architectures of one family
(`docs/internals/common/atoms/math.md`'s bf16 pair arithmetic picks its form by `bf16_simd_arith` and `bf16_simd_fma`).
A FACT is an input to a plan (the shared memory an area holds) or to the admission; a fact nothing reads is not tabulated. Neither is ever a branch
of a schedule. PER-SM FACTS LIVE HERE, PER-DEVICE FACTS DO NOT: SM count is not an architectural constant (GA102 ships
at 80 and 82 SMs, GA100 at 108), so it is a run-time `cudaDeviceProp` query.

## The queries over a row

<a id="caps-of-note-l147"></a>**sm_90a is THE BUILT ROW** (`caps_of`): its row carries the facts of the hopper
family's schedules; a binary built without sm_90a refuses an sm_90 card by name
([`select.md#card-of`](select.md#card-of)).

<a id="tabulated"></a>**`tabulated` is the SET OF ROWS, which is not the set the graph builds.** `tabulated(cc)` is true for each
compute capability a device part states (today sm_90a): the architectures the kernel knows how to size itself
for. A compute capability that is not a row fails the build, whose compiling target `arch_caps.cuh` asserts tabulated, and is refused at run time; it
is never approximated by a neighbouring row. THE BUILT SET is `registry.built`, the architectures a device part states (today sm_90a alone:
`BUILT_ARCHS`, the default of `--//:archs`), the same set ([`hardware.md#builds`](hardware.md#builds)).

<a id="hopper-only"></a>**THE ONE FAMILY BUILT** is the hopper's: this header refuses every other family by name, in one `static_assert`
after the `tabulated` check, and no header that includes it states a family of its own. A host pass compiles for the
hopper (`kArch` 900). The `ampere` branch is the only Ampere (docs/architecture/families.md).


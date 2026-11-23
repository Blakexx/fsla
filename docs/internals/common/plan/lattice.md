# The lattice's law (`csrc/fsla/src/common/plan/lattice.h`)

C++ owns the lattice ("C++ owns it", the owner's ruling of 2026-11-23, on who owns the leaf lattice), and this header is its one statement: the `(k, m)` owner box of a hierarchy's
level widths, the search that picks it, and the atom's rectangle in it. It is a pure `constexpr` header with no kernel's facts, so every reader reaches it with no device: decode's launch
function ([`decode_launch.md`](decode_launch.md)) lays the launch out from it, `plan_tool --lattice` ([`plan_tool.md`](plan_tool.md)) prints it for the CPU tests, and the extension's host
entries `fsla_decode_lattice` and `fsla_decode_lattice_box` ([`../../decode/decode_api.md`](../../decode/decode_api.md)) hand it to Python, whose `fsla.ops.lattice` reorders leaves by the box it
reads there and computes none of it. The rulings behind the form: the descriptor is runtime context derived from hardware facts and user input, the matchers contain the logic that distills it into an arm, and
anything with any kind of logic for that mapping is in a matcher (2026-11-23); kernels that must agree on a fact share it through C++ code, never a value computed outside and passed in (2026-11-07).

Symbols: `widths[l]` is level `l`'s width, `D` the levels, `k = 2^kappa` the uniform per-level sub-box span, `m = 2^j` the capacity multiplier, `s_l = k * m_l` an owner's span at level `l`,
`g_l = widths[l] / s_l` the owner-grid extent, `BC = prod s_l` the leaves one owner holds, `owners = prod g_l`.

- **The box** (`lattice_box`, a template over the level count): the innermost level takes as much of `m` as it can carry, never less than its balanced share `ceil(j / D)`, never less than an atom
  asks (`s_{D-1} >= kLatticeAtomLeaves`), and never more than its own width admits (the largest `c` with `k << c` dividing the width); the rest of `j` is spread over the outer levels, innermost first,
  so those exponents differ by at most one. A box is refused, in words (`lattice_refusal`), when `k` or `m` is not a power of two, when the levels cannot carry `j` (`kDoesNotFit`), or when a level is
  not a whole number of owner spans (`kInsideLevel`). The atom floor is a preference and the width ceiling is the law: a level narrower than an atom spills capacity outward and the atom spans more than
  one level, which costs occupancy where a spanned level is sparse and never correctness.
- **The search** (`lattice_solve`): over `k` in the powers of two up to `kLatticeMaxSpanK` (4) and `m` in the powers of two up to the leaves, among the boxes `lattice_admissible` accepts (no
  refusal, `BC <= kLatticeCapBc`, every span `<= kLatticeMaxSpan`), MAXIMIZE `BC`, then take the SHALLOWEST atom rectangle (an atom spanning two levels is filled only when both are dense for the
  side), then the LARGEST `k` (the carry family's codegen unit); the `(1, 1)` box when none is admissible. `(64, 64)` and `(256, 256)` both give `(4, 8)`, and `(16, 16, 16)` gives `k = 2`,
  whose one-level rectangle the `k = 4` box cannot reach.
- **The atom's rectangle** (`lattice_atom_rectangle`): the trailing levels whose spans multiply into an atom, taken whole, and the digits of the next level out that complete it.
- **The constants**, stated here: `kLatticeCapBc = 128`, the one number of the law that is a CHOICE rather than a consequence (the leaves one owner holds, stated on `BC` rather than on
  `m` because `BC = k^D * m` grows with depth); `kLatticeMaxSpan = 32`, the owner span the device reads as one 32-bit slice; `kLatticeMaxSpanK = 4`; `kLatticeAtomLeaves = 16` and its log2
  `kLatticeAtomShift`, the page granule. The kernel's span limit (`kDecodeMaxSpan`) and its lists a level takes (`kKinds`, `kDecodeListKinds` of `decode_launch.h`) ARE these, by definition; the atom's
  leaves have a second statement, the shared geometry's (`kAtomLeaves` in `common/geom.cuh`; `kAtomShift`, its log2, in `decode/decode.cuh`), tied to these by a `static_assert` in `decode.cu`; Python's `MMA_K_QUANTUM` is tied to the atom's leaves by `tests/unit/test_lattice.py`, which reads the constant out of `plan_tool --lattice`,
  and the build stamp's span limit is tied to the solver's by `tests/unit/test_decode_capacity.py`.

The header takes the level count as a template parameter, not as a constant of its own: `kDecodeMaxLevels` stays the one statement of the depth, in `decode_launch.h`.

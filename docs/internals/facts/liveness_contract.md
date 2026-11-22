# `fsla/engine/facts/planes.py` — the liveness contract: the digits, the row alignment law and the atom byte

Mirror doc for the liveness the masks state and the one fold over it that has a consumer: `atom_bits`, the per-atom READ and
WRITTEN byte a stateful call's residency is keyed on (`tests/unit/test_atom_bits.py` gates it). THE FACTS PASS IS DELETED
(Blake, 2026-11-22: "lets just delete facts"): there is no kernel, no word table, no tile union and no box fold; the Hopper
design reads the digits in its gather pass (`docs/architecture/hopper-gather.md`). What stays is what has another subject:
the digits' statement, the row alignment law, and the byte.

Vocabulary, defined here and used unqualified below: `D` = routing depth;
`B_l` = level `l`'s PADDED width (a power of two at or above 16 — the descriptor's,
K50); `b_l` = its LOGICAL width, which never leaves the host; `L` = tokens in the call;
`BH` = batch × heads; `N = prod_l B_l` = leaves; **atom** = 16 consecutive leaves in
canonical order, which is the page granule; `H` = query heads, `H_w` = write heads.

## <a id="max"></a>The digits: each against its row's max

Python's statement of the digits is `fsla.routing.quant` (`AMP_LEVELS`, the `ROWS` form);
the decode fold's `kRowMaxDigit` (`csrc/fsla/src/decode/decode_fold.cuh`) is the same 255, which `tests/unit/test_api_contract.py` holds.

Every level's row is stored unit-max: u8 digits with the row's largest at 255 (`fsla.routing.quant.AMP_LEVELS`), the row's max beside
it as an fp32 scale. A digit `b` is the amplitude `b / 255 × scale`. An all-zero row, its scale 0, is a dead token at
that level.

## <a id="alignment"></a>The row alignment law

Every level's rows start on 32 bytes and step batch, token and head by multiples of 32 bytes (a dimension of length one
is free), in `RouteFactors`' check; `quant.ROW_ALIGN` states it. The producer and `RouteFactors.from_levels` pack each side's levels in 32-byte slots
(`fsla.routing.quant.row_slot`: a level's width rounded up), the slot's tail zero — at power-of-two widths of 32 and up
the slots are the widths, so the law costs nothing there. It is what lets a kernel move a token's run as 16-byte
asynchronous quanta whose source is 16-byte aligned and whose 32-byte run is one sector. A level whose width is not a multiple of 16 is copied whole
up to its last digit and no further (a partial quantum zero-fills the rest).

## <a id="digits"></a>The digits, dense or sparse

Every level's rows are u8 against the row max (`AMP_LEVELS` = 255), on either side, and a zero digit is an exact
zero: there is no floor under a row ([`docs/CONSTITUTION.md` R23](../../CONSTITUTION.md#r23)). A DENSE level is one
whose activation has no exact zeros (softmax, not entmax); it differs from a sparse
one only in that its support is not worded -- no pass reads or writes its rows, and its bits are its
width mask over the live tokens. A dead row (scale 0) is zero everywhere, including where a consumer divides the row by
its own sum (the decay's shares).

## <a id="atom-bits"></a>The atom byte

`planes.atom_bits(masks, padded_widths, tokens)` folds the op's MASKS (ruled 2026-11-13, 2026-11-22; [`api.md` §2.6](../../api.md)):
the producer emits a bit per side, level, digit and token beside the rows, token-packed (bit `t % 32` of word `t // 32` at a
digit, `[B, H, ceil(L / 32), w_l]` a level). There is no row test, no dense flag and no static width mask: a level whose
activation has no exact zeros has an all-live mask like any other, and a digit past a level's width is the pad, dead. A mask
that does not cover a row's nonzero digits changes the result; that is the supplier's job, and the oracle states it (`r * m`).

An atom is an aligned rectangle in digit space (canonical order is the radix over the levels, the last minor), so per head

    live(atom, token) = AND over levels l of ( OR over d in atom.run(l) of bit(side, l, d, token) )
    ATOM(atom)        = OR over tokens of live(atom, token)

The AND is not a relaxation: a token's live leaf set is the PRODUCT of its per-level live digit sets (the routing is a
Kronecker product), so "some live leaf inside the atom" and "every level has a live digit inside its run" are the same
statement. The byte is `ATOM_READ | ATOM_WRITTEN`, never pre-ORed: a write head's READ is the OR over its `g = H / H_w`
query heads, its WRITTEN its write side's. It is exact, so `plan_exact` allocates on the written half, residency checks
read the read half, and a decode step consumes the set because its unit is the page. The gate holds it to the leaf-level
brute force.

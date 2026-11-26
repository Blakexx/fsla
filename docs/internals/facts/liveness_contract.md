# The liveness contract: the digits and the row alignment law

Mirror doc for the liveness the masks state. The Hopper design reads
the digits in its mask-preparation pass (`docs/architecture/mask-preparation.md#the-mask-preparation-pass`); this page holds the digits' statement and the row
alignment law. No per-atom activity byte is folded from the masks: the state is the dense plane and every atom is resident ([`../state.md`](../state.md#dense-state)).

Vocabulary, defined here and used unqualified below: `D` = routing depth;
`B_l` = level `l`'s PADDED width (a power of two at or above 16 — the descriptor's,
K50); `b_l` = its LOGICAL width, which never leaves the host; `L` = tokens in the call;
`BH` = batch × heads; `N = prod_l B_l` = leaves.

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
one only in that its support is not worded -- no pass reads or writes its rows -- and, in the prefill layout, it has no mask: its
entry is absent (`None`), and absent means dense, every digit of every token live (decode's masks are made for every level, a dense one all-live). A dead row (scale 0) is zero everywhere, including where a consumer divides the row by
its own sum (the decay's shares).

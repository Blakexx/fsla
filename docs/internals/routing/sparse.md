# `rola/routing/sparse.py` — a sparse level in the internal format

RULED 2026-09-30: a level side is stored either as unit-max rows (dense: every digit read) or as its BITLIST and its
LIVE COEFFICIENTS packed in bit order (sparse). The coefficients are the rows' own digits (u8 up to depth two, f16
deeper); only their placement differs. THE BUNDLE IS SELF-DESCRIBING: the entry's type is the whole fact
(`RouteFactors.sparse` reads it), and readers branch on it, never on the activation that made the level. The form is
chosen once, at production: the producer by its activation (exact zeros: sparse), a caller of
`RouteFactors.from_levels` by its `sparse` argument (by default the same rule; an opaque level, which may have zeros,
packs). It is a declaration, never a measurement. A ONE-TOKEN BUNDLE IS ALL ROWS: the form compacts tokens, and one
token has none -- at one token the step reads every digit of its row anyway, and a mask falls out of that read for a
ballot a word, while packed terms would need the mask first.

## <a id="digit-form"></a>The form

`words`, int32 `[B, H, width, nw]`: bit `t` of word `i` is set when token `32 i + t` is live on the digit. A row is the
call's tokens in whole words, rounded up to four (a 16-byte row), and every bit past `T` is zero.

`terms`, `[B, H, width, windows, slot]`: a FIXED SLOT per (digit, window), the window's live digits from the slot's
front in token order -- the `k`-th set bit of the digit's window words is the slot's `k`-th entry, and a run's length is
the popcount of those words, which every reader has. `slot` is the window (512 tokens), or the call's tokens rounded up
to a word when it is shorter. Past a run's length a slot's entries are unspecified. The slots reserve the dense plane's
bytes: a buffer is allocated before the live count is known, and entmax's support has no declared bound, so the format
saves the bytes READ, not memory.

## <a id="validity"></a>Any pair is a valid level

A set bit over a zero term costs a slot and contributes nothing; a term no bit reaches is never read. So
`SparseLevel.__post_init__` checks structure alone -- dtypes, one device, the shapes -- and the producer's
gate checks the kernel's words against the support of the terms they place
(`tests/integration/test_produce_kernel.py`).

## <a id="reference"></a>The reference packing and the dense view

`SparseLevel.from_rows` packs unit-max rows (a zero digit is dead): the producer's reference epilogue, which
`RouteFactors.from_levels` / `from_rows` apply to every side their `sparse` names. `SparseLevel.dense` is its inverse -- the rows the
level holds -- for the oracle and the references (`RouteFactors.rows`, `RouteFactors.dequantized`), never a kernel path.

# `fsla/ops/paging.py` — the state page's format and its host codec

Mirrors `fsla/ops/paging.py`, whose docstring keeps the vocabulary. The module is the HOST statement of the stored
page format that `csrc/fsla/src/common/state_page.cuh` defines on the device, for the oracle, the tests and the state's
`materialize()`; the bytes and the arithmetic are in [`../common/state_page.md`](../common/state_page.md). It holds no
allocation, no address table and no residency: the state is the dense plane
([`../state.md`](../state.md#dense-state)).

## A page is the MMA atom, not a compute tile

`MMA_K_QUANTUM = 16`. The ruling: **`BC` is a re-derivable COMPUTE tile; a page is a STORAGE atom.** `BC` varies with the
carve and across rebuilds, and a layout keyed to a compute tile would be re-keyed by each such move; the page is 16 leaves in
canonical leaf order, and every legal `BC` is a whole number of atoms by construction (`16 | BC`), so a box spans exactly
`BC / 16` atoms. A page's leaf `n` is row `n & (MMA_K_QUANTUM - 1)` of page `n // MMA_K_QUANTUM`; the dense plane
addresses page `a` of batch-head `bh` at `(bh * N / 16 + a) * page_bytes`.

## What the module holds

| name | what it is |
|---|---|
| `PageLayout`, `page_layout(d_v)` | the byte geometry of one page: hi s16, lo u8, exponent s8, fp32 mass; `d_v` a multiple of 32 |
| `encode_pages(logical)` | logical `[..., 16, d_v + 1]` fp32 to stored `[..., page_bytes]` uint8, the kernel's arithmetic bit for bit |
| `decode_pages(stored, d_v, hi_only=False)` | the inverse; `hi_only` is what a read-only load reconstructs |
| `bytes_equal(a, b)` | byte identity of two stored page blocks: `torch.equal` is a float comparison and a stored 32-bit word may read as NaN |
| `to_split_planes`, `from_split_planes` | the unported carry's own page, fp32 split into two bf16 planes (`SPLIT_PLANE_DTYPE`); the carry port deletes it |

**THERE IS NO RAGGED STATE.** A leaf axis that is not a whole number of pages is refused (`_refuse_ragged`), not padded and not
carried in a tail page ([`../state.md`](../state.md#int8-state)).

**ORACLE AND BRIDGE ONLY.** A measured path never rejoins a page on the host: `decode_pages` allocates the `[.., N, cols]`
tensor, which is for the fp64 oracle, a cross-arm hand-off and a test.

# The state page's split bf16 planes

Mirror doc for `csrc/rola/src/common/state_page.cuh`. `Blocks<DV>` — the page's byte
geometry below — is the ONE definition every body's state I/O is laid out against: the
carry kernel's `box.cuh` (`BP::Page = Blocks<DV_>`) and the decode step both read its
offsets (`kBytes`, `kLo`, `kMassHi`, `kMassLo`) directly, each inlining its own sweep
over the layout (the carry kernel's `state_copy`, `carry_kernel.md#state-io`; decode's
per-leaf-row access). The channel-split carry body and the two-scan
backward pass this page once served were deleted whole in the 2026-08-30 carry-family
rebuild (`docs/internals/DELETIONS.md`, commit ef567f9) and have not been rebuilt; the
current carry kernel is the single body that replaced them.

## <a id="geometry"></a>The page, in bytes

Ruled by K50 on 2026-08-29 ("THE PAGE STORES SPLIT bf16 PLANES"). A page is one atom
= 16 leaves, and it holds

    [16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]

`hi || lo` is the fp32 word EXACTLY -- `hi` is the truncation `bits >> 16`, `lo` the
low half -- so the page carries the same bits an fp32 page did, in the same
`16 * (DV + 1) * 4` bytes, and `Blocks<DV>::kBytes` static-asserts that equality.
Nothing about the arena, the page table, the id space, the VMM commitment or the
32-bit displacement bound moves with this: it is a re-layout INSIDE the page.

What the re-layout buys is two things the fp32 rows could not give:

1. **A hi row is one line.** At `DV = 64` a hi row is 128 B and a page's hi plane is
   eight lines; the fp32 row was `(DV + 1) * 4` = 260 B, which is 8 sectors plus 4 --
   never line-aligned, and 2x the bytes.
2. **A read-only atom moves half the bytes.** An atom this call READS but does not
   WRITE has a lo plane no arithmetic can observe (see below), so `load` skips it.

The mass rides its OWN two blocks rather than as column `DV` of each row, which is
what keeps the value row a power-of-two stride in every access and closes the row
alignment question K50 left open (no padding).

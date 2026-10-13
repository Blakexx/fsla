# `den_sweep`

Mirrors `measure/lab/experiments/den_sweep.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's window-aligned den as a pass beside the carry (den depends only on routing: the read leaf product against
the write mass of the windows before -- `tests/oracle/reference.py`), so the readout could add at a common unit known
beforehand, with a second channel beside it: the same product against the mass of F = m x the writing token's value
scale (its largest value channel's magnitude) is num, and num / den, a weighted average of the read tokens' value
scales, bounds every readout channel of the token, so its unit is known before the readout too. At 8 CTAs a head (64
level-S rows x 128 level-R digits each, 16 warps), on the cells' own routing: `den_sweep.inputs.py` writes each probe
cell at the flagship widths with 32 windows (`flagship-{alt-k4,alt-k16,cohort-k16,dense}-L16384`) as the facts pass
hands it to the carry, `fm`'s (F, m) the gains. A sparse side's MMA takes its N from the sparse level as 4 of its
digits x the two channels, an entry live in two of N's 8 columns; a dense side takes an n-tile a channel.

The list form (a cell whose level R writes and level S reads are sparse). Writes (f16 x f16 -> fp32): a warp owns 8
level-R digits, two MMAs' N; a 4-digit tile's list entries are K, arranged so each aligned 8 holds distinct token & 7
(as the facts pass could emit them); the 64 rows M; the mass the accumulators, held at the window's gain scales
(powers of two the facts pass would emit, the gains pre-scaled, the accumulators rescaled when they change); A = the
tokens' staged f16 coefficients by `ldmatrix.trans` (a token's 16-byte chunks XORed with token & 7), B = an entry's
coefficients (its gain x value a channel) in its columns. The den (u8 x u8 -> s32): a warp owns 4 level-S rows (N) and
its band's tiles of 16 read-list entries (token, digit, u8 value; M); K = the CTA's 128 digits; A = an entry's u8 read
row as loaded (a lane's contiguous quarter, the K order the published mass follows); B = the mass as 16-bit fixed
point (a column scaled by the power of two putting its largest value in [2^14, 2^15)) in high and low byte planes, two
integer MMAs a 32 digits; an entry's results are its row's two columns (high x 256 + low) x 2^-e x its value, each
added by the lane holding it (den and num side by side a token). The read lists stage once the den is done with the
window's (one buffer of 4096 entries, cohort's lumpiest CTA-window holding 3065), the next window's read rows
prefetching to L2.

The dense form (both sides dense). Writes: K = the window's tokens in chunks of 64, the coefficients and the values
staged double-buffered, B = the values by `ldmatrix.trans` times each token's gains (f16 at the window's scales), an
MMA a channel. The den: every token against every row, a warp's 8 rows (an n-tile a channel) over its half of the
16-token tiles, one tile in flight (two spill at 128 registers), each tile's integer results times the token's level-S
read coefficients, summed over the quad, added once a token a channel a warp.

Every den and num within 1% of a double reference before timing. One head's CTAs; sm_86 at base clock, cycles a
CTA-window, the one-channel pass in brackets (commit 790dfac6):

| cell | full | den alone | writes alone |
|---|---|---|---|
| `flagship-alt-k4-L16384` | 12.7K (11.5K) | 8.45K (7.56K) | 8.83K (7.64K) |
| `flagship-alt-k16-L16384` | 29.4K (26.7K) | 16.4K (15.4K) | 18.1K (15.5K) |
| `flagship-cohort-k16-L16384` | 35.3K (30.8K) | 19.9K (16.9K) | 22.7K (18.9K) |
| `flagship-dense-L16384` (dense form) | 68.9K (39.9K) | 23.9K (13.5K) | 45.8K (26.2K) |

The list form pays for F in structure, not MMAs: a write tile holds half the entries (more padding a tile), a den band
is one warp's 4 rows (the two warps of an 8-row band no longer share its tiles), the publish doubles (the skeleton alone
4.42K on alt-k4, from 4.17K). The dense form pays in MMAs: its writes' HMMAs double to 4096 a CTA-window, 32.8K cycles
at sm_86's 32 a scheduler for fp32 accumulate, and its den's IMMAs double. Earlier revisions on synthetic pools: 16
warps a CTA lifted issue from ~26% to ~38% (commit 5b26266f); the integer den and the residue order (79d058ef); the f16
den (ece3d4d6); FFMA forms, digit-major (0db7bf1e) and token-major (673c2375). Record:
`measure/lab/results/den_sweep/sm_86.json` (the inputs' digests in it), counters `sm_86.counters.json` (a kernel's
median over the cells it ran).

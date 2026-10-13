# `fold_part`

Mirrors `measure/lab/experiments/fold_part.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's fold (`csrc/rola/src/carry/fold.cuh`) over one head's real write lists (`fold_part.inputs.py`: nl64k-alt-k4,
32 windows), the pool filled once, one CTA an SM each a 32-digit box: the composed shape (a window's units behind a
barrier), the same units back to back (the step's own cost) and the MMAs alone on fixed rows -- each against the
structural floor.

Two layouts of the same lists follow (`ordered.bin`, `packed.bin`). A slot's residue mod 8 is its pool row's bank
group, and an `ldmatrix` phase reads 8 rows, so rows sharing a residue in a phase conflict. ORDERED: each unit's
entries reordered so a step's 8-entry halves hold distinct residues, and a unit's first step (at most 16 entries)
dealt across both halves with its pads on the residues each half leaves free. 8-TOKEN STEPS: the step on m16n8k8
MMAs (half an m16n8k16's cycles, `mma_k8`), heads packed live-first with pads on free residues, a second step only for
a list past 8 entries.

Measured (sm_86, nl64k-alt-k4, against the 16-token floor; ncu wavefronts per 16-token-equivalent MMA):

| form | composed | steady | wavefronts | of which bank conflicts |
|---|---|---|---|---|
| as built | 1.179x | 1.064x | 4.69 | 1.57 |
| ordered, dealt heads | 1.157x | 1.060x | 3.69 | 0.69 |
| 8-token steps | 1.012x | 0.844x | 3.48 | 1.30 |

Ordering within a half cannot help at alt-k4 (a list's ~8 entries a window fill one half); dealing them across the
step's halves does. 8-token steps do 0.70x the tensor work (1.40x the MMAs at half the cycles); their full phases
conflict more than dealt halves, so their shared-memory saving is small.

Record: `measure/lab/results/fold_part/sm_86.json`.

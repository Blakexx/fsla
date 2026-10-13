# The constitution

Blake's standing rulings: stable, changed only by Blake, and above every law. The laws -- the coordinator's procedural
and incident-derived rules -- are `docs/KERNEL_STANDARDS.md`; a law never contradicts a rule here. Each rule is stated
once, with every amendment folded in; the numbers are permanent, so a citation never breaks.

<a id="principles"></a>
## The principles

- CORRECTNESS INSTRUMENTS: the fp64 oracle is the truth, and bit-identity is the refactor instrument
  (`docs/testing.md`).
- EFFICIENCY IS CORRECTNESS: a design is built in its best form, never correct first and fast later.
- CLOSED WORLD: only measured binaries run (`docs/ratification.md`).
- ONE FORM PER MECHANISM: no fallbacks, no dead code.
- BUILD THE WHOLE DESIGN, OR STOP AND NAME THE GAP.
- THE ANCESTRAL-CODE STANDARD: code and docs are written for the reader a decade out.
- PORTABILITY: the repository references only itself (`docs/KERNEL_STANDARDS.md#s17`).
- ONE ATOMIC PRODUCT: the repository in published form is one self-contained kernel product, gated holistically at its
  current state; every test lives in one of three tiers, `tests/unit`, `tests/integration`, `tests/oracle`
  (`docs/testing.md`), plus `measure/`; no per-campaign test directories, and no campaign names in the tree -- names
  are roles.
- SELF-DOCUMENTING CODE: code is legible without comments. Comments are sparse -- a file header and one-to-three-line
  declaration blocks naming the purpose and the one invariant -- and state what the code does and why, in the present
  tense, in the library's own terms: never a past state, a superseded mechanism, a stage or campaign identifier, or a
  comparison to what was. Derivations and why-not-the-other-form prose live in `docs/internals/`, which mirrors
  `csrc/`.
- DOCS ARE CURRENT TRUTH: a doc states what is; the past appears only as the rationale for the current shape. Removals
  are recorded in `docs/internals/DELETIONS.md` and git, and nowhere else.

<a id="r8"></a>
## R8. Design mechanisms that expand to all architectures

Every mechanism states its sm_80 / sm_90 / sm_100 form at design time (wgmma and tcgen05 shared- and tensor-memory
operands, TMA box loads, clusters and distributed shared memory, vector RED); a mechanism that exists on one
architecture only is a workaround, not a design.

NO FALLBACKS AT THE ATOM. A primitive may select different native instructions per architecture; it never selects a
wider-type emulation on a target architecture. An operation an architecture lacks is a declared design fact with its
cost on the card, never an `#if` branch. (An fp32 emulation of a bf16 multiply lived behind a passing oracle while the
native instruction existed all along.)

<a id="r9"></a>
## R9. Never branch on a fuzzy threshold

No dispatch, launch policy or in-kernel path keys on a measured statistic (sparsity fraction, density, live count,
"small L"). Branch on STRUCTURE the declaration fixes (tile counts, modes, plan axes); the kernel itself handles any
data distribution. A deterministic cost model per architecture choosing among exact forms is structure.

A structure branch fails loud at every layer, through one primitive: every CTA-uniform structure branch in device code
is a `uniform_switch<Set>(value, body)` (`common/structure_switch.cuh`), whose contract is the four layers -- a
generated case set with a count assert, compile-time exhaustiveness (the derivation's outputs within the set), the
seam's launch refusal reading the same set, and a `__trap()` default with a post-build coverage sweep. A hand-written
device `switch` on a geometry field is a lint finding.

<a id="r10"></a>
## R10. Design and code for all regimes at once

Dense, sparse, mixed, clustered, small and large L, small and large N. Never overfit a mechanism, a measurement or a
verdict to one cell: every design states its cost in each regime, every stage is probed at both ends of the
distribution, and a mechanism that wins one regime by losing another is a finding to rule on, not a win. Dense is the
worst case and a test of the kernel, never a regime to concede.

<a id="r11"></a>
## R11. The axis law

A compile-time axis is legitimate only as a deterministic function of the declaration the selector can pick;
everything else is derived or a constant. The compiled axes are the carve's plan, per side (the shipped set's key);
per-level widths and the leaf order are runtime addressing; the window and its token grain are constants.

<a id="r12"></a>
## R12. Shape compile-time, addressing runtime

Only the register shape -- the plan's rows, factors and channels a warp holds -- is compiled. Every shared-memory
tenant is launch-sized from the descriptor; residency is asserted at the seam as a refusal; no runtime trip count sits
in a register-shaped loop; shared memory is never max-sized.

<a id="r13"></a>
## R13. The state owns its format

A descriptor (D, per-level widths that are powers of two from 16, the canonical order, the page rectangle as the
trailing four canonical bits, d_v) is bound at the first call and checked after. Pages hold each leaf-channel as an
s24 at its (leaf, 32-channel group)'s power-of-two exponent and the mass in fp32; a kernel converts at its own
boundary (the fold to fp32 in registers, the readout to its f16 image). The warp sub-box's spans are compile-time from
the plan; prefill and decode share one admission law; N is a multiple of 16, and a ragged N is refused, never padded.

TWO CONTRACTS, ONE TRANSLATOR. The API accepts any shape (any width from 2, any d_v, any L) and serves it by padding;
the kernel accepts only the descriptor's strict shape and its operand formats and refuses the rest -- it never pads or
masks. The seam is the only translator. Both boundaries are tested: API tests prove padded runs bit-identical, and
kernel-boundary tests call the launch surface with unpadded shapes and assert the refusal; a lint keeps logical-width
fields out of every kernel entry signature.

<a id="r14"></a>
## R14. Ownership grains nest on the page, nothing more

Decode's unit is the atom; prefill's box owns whole rows; a kernel consumes activity bits at its own grain and the
page table derives its own.

<a id="r15"></a>
## R15. The CTA is the SM

One resident CTA per SM with the warps the register file admits; warps partition the box by role inside it. Latency
hiding is the kernel's job (covers, split-phase edges): exposed latency is a coverage finding, never grounds to split
the CTA. One CTA per SM is the goal, not a measured choice -- a two-CTA launch is only a diagnostic of how far the body
is from the target.

<a id="r16"></a>
## R16. Ship without arm registration

A Python-package user never registers an arm. The shipped set is the carve plans `tools/manifests/shipped_set.json`
declares per architecture; an unknown declaration is admitted by a refusal-checked derivation, never by enumeration.

<a id="r17"></a>
## R17. One family, one form a component

Operands are the fp16 family. Each component has one form across densities and cards, unless measurements show that
no single form wins or ties everywhere. Accumulation precision is declared per component -- the fold fp32; the readout
fp32 where that is full rate, fp16 on GeForce; the output's partials exact s16 pairs -- and recorded in
`docs/architecture/precision.md`.

<a id="r18"></a>
## R18. One body per op family

Dense and sparse are plans of the one body, never a second body.

<a id="r19"></a>
## R19. Laws land with their stage

A ruling that changes a law is written into the standards in the same stage that implements it, never only on a
card; the milestone carries the drift guards (lints, docs, skills).

<a id="r20"></a>
## R20. The ownership unit is the CTA's box

The CTA's box is the carve's box (per-level extents and a channel extent), carved per side by its plan; the shape set
is the plans. A CTA loads and stores its box at entry and exit, so its read-side warps' union and its write-side
warps' union are the same leaves; the CTA's box equals the SM's at one CTA per SM, the goal (R15). Where the
accumulator lives (registers, tensor memory) changes how a warp touches its share, never who owns what.

<a id="r21"></a>
## R21. Every asynchronous operation states two edges

An asynchronous write -- `cp.async`; TMA, `cp.async.bulk`, `wgmma` on sm_90 and after -- has two hazards, and a wait
covers only one. Its COMPLETION edge (read after write) is the wait or barrier phase behind which the first read of
the new contents stands. Its ISSUE edge (write after read) is the barrier after the last read of the old contents by
any warp: a warp's wait orders that warp against its own copies, never against another warp's reads of the bytes it is
about to overwrite. Every shared tenant an asynchronous operation writes states both edges in its ledger, and a review
refuses a tenant that names only the wait. On sm_90 the two edges are a pipeline stage's empty and full barriers; on
sm_80 they are the CTA edges the frame already has, the issue placed past the edge that ends the last read.

<a id="r22"></a>
## R22. Degrade smoothly, scale, partition into independent units

Blake: "my concern is any kernel design that 'spills' at some threshold rather than just cleanly degrading. How things
scale is important too, instead of just designing for one specific point. Kernels should ideally partition atomic,
independent work across ctas that join back together in clean ways. At least, normally unless there is some strong
reason not to. Everything is about tradeoffs"; "It's also ok to join via atomics, I still don't care too much about
determinism".
1. NO THRESHOLD CLIFFS. A kernel's cost rises along a curve with density, width, depth and N. A design that changes
   path or falls off a cliff when the data crosses a capacity (a shared-memory buffer sized for the sparse case, a held
   window that only fits narrow levels) needs a stated strong reason in its design block. A capacity the PLAN fixes (a
   box, a stage, a token block) is structure, not a threshold; a capacity the DATA can overrun is a threshold.
2. DESIGN FOR THE SCALING, NOT A POINT. Every mechanism states how its cost grows with N, D, density and the
   architecture; a cache that may or may not hold a working set is priced as if it does not, so the cache is upside,
   never a cliff in the model or the gate.
3. INDEPENDENT UNITS, CLEAN JOINS. Work is partitioned into atomic, independent units across CTAs that join by
   disjoint writes, reductions or atomics (run-to-run bitwise determinism is not required), or a stage boundary
   between kernels. A CTA waiting on another CTA's progress needs a strong reason.
4. EVERYTHING IS A TRADE: a faster form that breaks 1-3 is chosen only with the reason and its numbers written down.

# The constitution

Blake's standing rulings: stable, changed only by Blake, and above every law. The laws -- the coordinator's procedural
and incident-derived rules -- are `docs/KERNEL_STANDARDS.md`; a law never contradicts a rule here. Each rule is stated
once, with every amendment folded in; the numbers are permanent, so a citation never breaks.

<a id="principles"></a>
## The principles

- CORRECTNESS INSTRUMENTS: the fp64 oracle is the truth, and bit-identity is the refactor instrument
  (`docs/testing.md`).
- EFFICIENCY IS CORRECTNESS: a design is built in its best form, never correct first and fast later.
- THE PERFORMANCE TARGETS (Blake, 2026-09-30): "at dense we are at the tensor floor, at sparse we are significantly
  below FA, down to basically the overhead floor which is ideally small"; decode the same, "such that our latency
  nearly falls exactly with sparsity"; the intra may be a constant cost at every density, as long as it is low. Every
  kernel is priced against these floors, never against FlashAttention parity.
- PARITY IS A FAILING GRADE (Blake, 2026-08-20): matching FlashAttention at sparse `N = L` while doing many times less
  useful work is a failure, never a result; it is reported as an unexplained loss with a ledger -- each resource's
  floor against its measurement, each loss classified geometry, algorithm or implementation and reconciled to the
  gap. Newer cards' asynchronous machinery is no explanation; the cause is named from receipts.
- LEAN ON EVERY AXIS, ON EVERY CARD: one kernel runs on every card, so nothing is wasted because this card does not bind
  on it -- instructions, each pipe, shared-memory wavefronts and bytes are each kept at what the design needs; a trade
  between them is made only when forced, and priced on every card.
- CLOSED WORLD: only measured binaries run: the declared targets (a kernel's arms in its `arms.bzl`, the built architectures, the pinned toolchain) and the gates every build runs (`docs/ratification.md`, `docs/build-graph.md#arms`).
- ONE FORM PER MECHANISM: no fallbacks, no dead code.
- ONE STRATEGY, APPLIED UNIFORMLY (Blake, 2026-06-26: "we shouldnt be making any 'branching' to avoid edge cases. if we
  have a strategy, we should do it uniformly"): nothing branches the algorithm to dodge one configuration's cost; that
  cost is the configuration's own property, reported so the user can choose another (a cost flat routing pays is why a
  tree or square one exists).
- INTERFACE FIRST: a subsystem's whole interface and primitives land first and every caller moves onto them at once;
  a part not yet built raises a named error, never a silent fallback; no parallel old path is kept beside the new.
- NO WORKAROUNDS: a problem the kernel can solve is solved in the kernel, at its root -- never a band-aid, a capability
  shim, a padding hack or "for now" residue.
- SPARSITY IS THE ACTIVATION'S: a skip is an exact zero the routing activation (entmax, sparsemax) produced, never a
  threshold, an epsilon floor or a straight-through operator.
- BUILD THE WHOLE DESIGN, OR STOP AND NAME THE GAP.
- FOUNDATION, THEN A NEW BODY (Blake, 2026-08-29: "build a generic foundation, then a kernel body"): a new foundation is
  never grafted onto a body that needs replacing; its contracts, derivations and primitives land first, then the body
  is built bottom-up in its final form -- final mechanisms, the pipelined structure from the skeleton -- never as a
  prototype to be fixed later.
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
- THE REPO IS ONE ATOMIC DESIGN: its docs are its whole current state and reference nothing outside it -- no log, plan
  or tracker -- save another design cited as the reason ours differs.
- DOCS ARE CURRENT TRUTH: a doc states what is; the past appears only as the rationale for the current shape. Removals
  are recorded in `docs/internals/DELETIONS.md` and git, and nowhere else.

<a id="r8"></a>
## R8. Design mechanisms that expand to all architectures

Every mechanism states its hopper (sm_90) and blackwell (sm_100) form at design time (wgmma and tcgen05 shared- and
tensor-memory operands, TMA box loads, clusters and distributed shared memory, vector RED); a mechanism that exists on
one architecture only is a workaround, not a design.

THIS LINE IS HOPPER'S AND BLACKWELL'S; MASTER IS HOPPER ONLY (Blake, 2026-11-09 and 2026-11-20). The ampere family is
the branch `ampere`, the frozen reference, and nothing of it is on master: no ampere card row, no sm_80 or sm_86 build,
no ampere atom, schedule, tool, tolerance or standard ("it may be more simple to just completely delete all ampere and
move to fully hopper, since it seems very cheap adn easy to use an h100 at this point"). Its design is no input to a
hopper design. The development box builds for sm_90a and runs host tests, lints and the SASS and checker gates; every
GPU test runs on a rented H100. Ampere may be finished later, from its branch.

THE FORM IS NAMED PER FAMILY. The architectures fall into families -- `hopper` (sm_90), built, and `blackwell` (sm_100),
tabulated and not built -- and a mechanism's form is named per family: an ATOM where only the instruction differs, the
family's SCHEDULE where the structure does (which warps do what, the pipeline's stages and barriers, the tiling). The
semantics, what the oracle defines, are one: every family's schedule shares them and is gated against the same
reference.

EVERY KERNEL ADAPTS TO ITS CARD. Every kernel is selected per product, compile-time specialized to it, and takes every
constant the product or the declaration decides from its plan; no kernel hard-codes one. A literal stays only where the
hardware fixes it (an MMA shape, the warp size), as the declaration's closed-world bound (refused by name past it), or
as an encoding.

FACTS IN THE TEMPLATE, DECISIONS IN THE KERNEL (Blake, 2026-11-24). The card's facts reach a kernel as constants of its
build, and the kernel derives from them, at compile time, every quantity the card decides: "all smem layout decisions
are made in the kernel, using the template facts or runtime facts that are able to be uniformly switched on". The
card's shared memory an SM is the fact a kernel's shared memory is serviced from ("it should be smem per sm, and we
service everything from. That, including how many CTAs"), and the budget "can just be the whole card". A layout ADAPTS
across the ranges the descriptors state ("Fail to compile means it couldnt adapt right? Ideally all the layouts adapt
to the template parameters which the descriptor gives the ranges of valid values for"): a build that fails names a
card the layout cannot serve. The rule is general and not shared memory's alone ("it's not just about smem, this is a
general thing"), and it is a rule of WHERE A DECISION'S INPUTS EXIST: "it's not really about facts vs decisions. Its about
dependency availability. Smem lZyout depends on runtime arguments so it needs to be in the kenrel. Things that don't
can be in the template in theory" -- what the card and the call's shape decide may be a template parameter the matcher
picks; what run-time values decide is a case of the kernel's switch, compiled like the rest of the case. The host decides none of it: a launch reads its grid, its threads and its bytes off the same
functions the kernel compiles ([`internals/common/smem_layout.md`](internals/common/smem_layout.md)). NOT YET SO for decode
(R12); what the tree still writes by hand is the backlog of the lints that refuse it (`tools/lint/facts_in_template.py`).

ONE ENTRY A KERNEL, BLIND TO ITS IMPLEMENTATION. A kernel's caller sees one interface: the entry finds the device's card
from its tensors' device and selects the family's schedule and the atoms' instructions behind it; the caller passes no
hardware fact. HARDWARE FACTS LIVE IN THE KERNEL PATH: Python reads one only for a purpose the C++ path cannot serve (the
fp64 oracle, the measurement tools); shaping or launching a kernel call from a fact in Python moves into the entry.

NO FALLBACKS AT THE ATOM. A primitive may select different native instructions per architecture; it never selects a
wider-type emulation on a target architecture. An operation an architecture lacks is a declared design fact with its
cost on the card, never an `#if` branch. (An fp32 emulation of a bf16 multiply lived behind a passing oracle while the
native instruction existed all along.)

<a id="r9"></a>
## R9. Never branch on a fuzzy threshold

No dispatch, launch policy or in-kernel path keys on a measured statistic (sparsity fraction, density, live count,
"small L"). Branch on STRUCTURE the declaration fixes (tile counts, modes, plan axes); the kernel itself handles any
data distribution. A deterministic cost model per architecture choosing among exact forms is structure -- each form
gated by forcing it; a tuned cutoff that changes results is not. So is a bound the USER declares and the kernel
certifies (a declared precision met per block on published values, the reference modelling the same rule): Blake, "it's
still branching on a threshold either way, in the kind that we allow, which is a user-supplied threshold". A
data-dependent path whose criterion cannot be restated as such a bound stays out.
THE ONE MEASURED STATISTIC A KERNEL KEYS ON is the DENSITY a caller hands the prefill beside the masks: a side's share
of live digit-token pairs on a level, which the library's helpers measure (Blake, 2026-11-24: "The reason I said
declared never detected was cause I couldn't think of a good way to do it. This seems like a good one."; "weights
should be outside the masks ... They need not be coupled, and are independent variables really"; "just density is fine
initially"). It chooses a LAYOUT among forms that are all exact, through `uniform_switch`, once a call ("switching
uniformly to precomputed things defined for those patterns"). It changes no result while the masks keep their
guarantee (R24), and nothing else measured keys a path. A BYTE OF 255 IS THE DENSE FLAG: that level-side is computed with
every digit live and its mask is never read (Blake, 2026-11-24: "a flag that says \"this is all dense\" that we can read
to skip loading those masks for the OR, and also to skip producing those masks"; "It's fine if we mark something as
dense when it's possible it's like 99.99 due to rounding"). Where a level-side's declared activation cannot make exact
zeros the library makes it no mask (Blake, 2026-11-24: "like if it's declared no zeroes, we just don't make the masks"; [`api.md`](api.md)
section 2.6).

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
EVERY PATTERN IS PROVIDED; ONE REGIME IS PROMISED (Blake, 2026-11-23: "it can just be something we provide, and we dont
promise it will work amazingly always, because its not the ideal regime"). Each side has a DENSITY on each level (R9), and from it a
GROUP SIZE (Blake, 2026-11-24: "multiply the sparsity by the actual level digits and round that to our allows range"):
1, 4, 16, 32, 64, 128 or 256 digits. A level walked whole is one group. A token is in a set where it is live in that
set's group on every level; the digits of a group it is not live on carry zeros. The kernels read group sizes and
nothing about how a level was produced (2026-11-23: "carry doesn't read the metadata, it just reads group size per
legel"). No side needs a level
walked whole: the floor the fp16 family kept under a dense level's zero went with that family (R17;
[`architecture/precision.md`](architecture/precision.md)). The performance targets are stated for the regime the design
is built for; outside it a pattern is correct and its cost is its own.
NOT YET SO: what is built and measured is one listed and one whole level a side at depth two
([`lab/carry_token.md`](lab/carry_token.md)); which patterns have arms is the contract's to state.

<a id="r11"></a>
## R11. The axis law

A compile-time axis is legitimate only as a deterministic function of the declaration the selector can pick;
everything else is derived or a constant. The compiled axes are the carve's plan, per side (the shipped set's key);
per-level widths and the leaf order are runtime addressing; the window and its token grain are constants.

<a id="r12"></a>
## R12. Shape compile-time, addressing runtime

The register shape -- the plan's rows, factors and channels a warp holds -- is compiled, and so is the shared-memory
layout: each branch of a kernel sizes its shared-memory tenants at compile time from the card's budget, through the one
layout ([`internals/common/smem_layout.md`](internals/common/smem_layout.md)), and may hold the whole of what its CTA
can reserve (Blake, 2026-11-24, R8's "facts in the template, decisions in the kernel": "all smem layout decisions are
made in the kernel"; "that smem budget can just be the whole card"). What a CALL needs of a tenant is checked against
the tenant's capacity at the seam, as a refusal; no runtime trip count sits in a register-shaped loop. NOT YET SO for
decode, whose launch function sizes its tenants on the host from the descriptor and refuses a step over the budget.

<a id="r13"></a>
## R13. The state owns its format

A descriptor (D, per-level widths that are powers of two from 16, the canonical order, the page rectangle as the
trailing four canonical bits, d_v) is bound at the first call and checked after. Pages hold each leaf-channel as an s24
at its (leaf, 32-channel group)'s power-of-two exponent and the mass in fp32; a kernel converts at its own boundary
(the fold to fp32 in registers, which the readout reads as its bf16 operand). The warp sub-box's spans are compile-time
from the plan; prefill and decode share one admission law; N is a multiple of 16, and a ragged N is refused, never
padded.

TWO CONTRACTS, ONE TRANSLATOR. The API accepts any shape (any width from 2, any d_v, any L) and serves it by padding;
the kernel accepts only the descriptor's strict shape and its operand formats and refuses the rest -- it never pads or
masks. The seam is the only translator. Both boundaries are tested: API tests prove padded runs bit-identical, and
kernel-boundary tests call the launch surface with unpadded shapes and assert the refusal; a lint to keep logical-width
fields out of every kernel entry signature is OWED (none exists yet).

<a id="r14"></a>
## R14. Ownership grains nest on the page, nothing more

Decode's unit is the atom; prefill's carve owns whole rows; a kernel consumes activity bits at its own grain and the
page table derives its own.

<a id="r15"></a>
## R15. The CTA is the SM

One resident CTA per SM with the warps the register file admits; warps partition the box by role inside it. Latency
hiding is the kernel's job (covers, split-phase edges): exposed latency is a coverage finding, never grounds to split
the CTA. One CTA per SM is the goal, not a measured choice -- a two-CTA launch is only a diagnostic of how far the body
is from the target.

<a id="r16"></a>
## R16. Ship without arm registration

A Python-package user never registers an arm. The shipped set is the arms each family's `arms.bzl` declares (the families of `ARM_FAMILIES`); an unknown declaration is admitted by a refusal-checked derivation, never by enumeration.

<a id="r17"></a>
## R17. One family, one form a component

Operands are the bf16 family (Blake, 2026-11-09). Every level's rows, either side, dense or sparse, are u8 unit-max
with the row's fp32 scale; the values are bf16 unit-max with an fp32 scale a token. An MMA's operands are bf16 (a u8
byte is exact in it) and its accumulation fp32 -- the fold, the readout and the den alike -- but for a window's pair
products of rows, which are integer MMAs of the bytes, exact (the data model locked 2026-11-09 has the intra so);
the output's sum is
fp32 inside a CTA, and ACROSS CTAS ITS WIDTH IS THE PLAN'S (R24; Blake, 2026-11-11): fp32, or f16 -- a CTA adds its
own partials in fp32 and issues ONE f16 line a token, scaled by a power of two from one coarse bound that is
re-based each segment and stored with the state so a continuation keeps it. No scale pass, no panel a group and no
pass that merges them, no switch of precision by card. The width is a field the caller states, a bound it declares
(R9), each width a form gated by forcing it; the producer's planner chooses it from the declaration's structure --
the levels' widths and walks -- and the op counts nothing. MODELLED on one cell (nl64k-alt-k4; a dense read side and
wider levels are not): one f16 line a CTA of 512 leaves costs 0.083% a token at the median, 0.40% at most,
against the family's own 0.115% (`docs/internals/tools/arith_model.md#f16`). The state reaches the readout through its
IMAGE in shared memory, stored once a window, WHETHER THE TWO SIDES' CARVES DIFFER OR ARE EQUAL (Blake, 2026-11-11;
amended 2026-11-13): laid across the fold's sparse digits
where they differ and along one sparse digit's dense digits where they are equal, the readout the same. A readout
from the fold's own registers is not built until a measurement shows it wins: the image's cost at the equal cell is
the lab's record (`docs/lab/carry_window.md`). Each
component has one form across densities and cards, unless measurements show that no single form wins or ties
everywhere. The family's error is its own class, 0.16 to 0.20% a token against the fp64 oracle on the flagship cells
(measured), accepted on every card and recorded in `docs/architecture/precision.md`.

<a id="r18"></a>
## R18. One body per op family

Dense and sparse are plans of the one body, never a second body.

<a id="r19"></a>
## R19. Laws land with their stage

A ruling that changes a law is written into the standards in the same stage that implements it, never only on a
card; the milestone carries the drift guards (lints, docs).

<a id="r20"></a>
## R20. The ownership unit is the CTA's carve

A CTA OWNS ITS CARVE, on hopper 512 leaves, a power-of-two extent a level (Blake, 2026-11-24: "The cta actually owns 512
leaves, not 256, with the exception of scaling down, and we are just partioning them during fold due to register limits.
During readout we don't partition, it's just one by one warpgroup"). THE CARVE IS APPORTIONED across the levels by the
two sides' densities (R10; "like the US Congress apportionment ... each one gets at least one assignment, and then after you
assign based on weight"): every level starts at one digit; each doubling goes to the level whose summed density, over one
more than the doublings it holds, is highest; a tie goes to the level the read side is denser on ("we wanted to bias for
the read dense side cause it's better"); no level goes past its own width. Two fold warpgroups each hold half of it, split
along one level between them and by values across their warps; one readout warpgroup accumulates from the state both
publish, whether the two sides' layouts differ or are equal (R17). WHICH LEVEL THE TWO FOLDS SPLIT ALONG is decided apart
from the carve (Blake, 2026-11-23: "the cta box decision is decoupled from how we execute it. we know that we want a larger
read dense allocation than fold, but once thats set, we can partition fold how we please"): along a level the write side
walks by groups smaller than the level first, otherwise along one it walks whole ("take any write sparse level first,
otherwise write dense"). A group and the carve nest on every level and a group may be the larger ("the more important thing
is the multiple rule, which we can guarantee via padding"): the CTAs inside one combination of the write side's groups
read one value stream. A smaller carve is the last resort under shared memory ("A box need not be 256 leaves we will scale
them down if we are forced to"). What a split along another level would share, and what uneven halves cost, are not
measured. The rule is one function the kernels call on the call's densities
([`cta_carve.md`](internals/common/plan/cta_carve.md), [`prefill_launch.md`](internals/common/plan/prefill_launch.md); the grid's split, the
ring and the kernels are owed: [`open-work.md`](open-work.md#the-surface), row A-f). A CTA loads and stores its carve at entry and
exit, so its read-side warps' union and its write-side warps' union are the same leaves; the CTA's carve is the SM's at one
CTA per SM, the goal (R15). Where the accumulator lives (registers, tensor memory) changes how a warp touches its
share, never who owns what.

<a id="r21"></a>
## R21. Every asynchronous operation states two edges

An asynchronous write -- TMA, `cp.async.bulk`, `wgmma`, `cp.async` -- has two hazards, and a wait
covers only one. Its COMPLETION edge (read after write) is the wait or barrier phase behind which the first read of
the new contents stands. Its ISSUE edge (write after read) is the barrier after the last read of the old contents by
any warp: a warp's wait orders that warp against its own copies, never against another warp's reads of the bytes it is
about to overwrite. Every shared tenant an asynchronous operation writes states both edges in its ledger, and a review
refuses a tenant that names only the wait. The two edges are a pipeline stage's empty and full barriers; where a
kernel has no stage ring (per-lane copies behind the CTA's own edges), the issue is placed past the edge that ends the
last read.

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
   box, a stage, a token block) is structure, not a threshold; a capacity the DATA can overrun is a threshold. Shared
   memory sets how many of a window's phases are co-active: as the area shrinks, fewer phases run at once (a fill
   overlapping only what has freed), then the phases' chunks shrink -- a continuum, never a form that fits or falls
   back (Blake: "as smem decreases, it just means less coactive and eventually smaller individual phases").
2. DESIGN FOR THE SCALING, NOT A POINT. Every mechanism states how its cost grows with N, D, density and the
   architecture; a cache that may or may not hold a working set is priced as if it does not, so the cache is upside,
   never a cliff in the model or the gate.
3. INDEPENDENT UNITS, CLEAN JOINS. Work is partitioned into atomic, independent units across CTAs that join by
   disjoint writes, reductions or atomics (run-to-run bitwise determinism is not required), or a stage boundary
   between kernels. A CTA waiting on another CTA's progress needs a strong reason.
4. EVERYTHING IS A TRADE: a faster form that breaks 1-3 is chosen only with the reason and its numbers written down.

<a id="r23"></a>
## R23. The operands are scaled blocks

Blake: "why cant it just take in keys with their shared scale and per-slot coefficients, queries with per-slot
coefficients and a shared scale too ... and just use them? ... our producers just natively do that form. if someone
doesnt use them, they use our factory which transofrms their input into the form the standard way you do any scaling";
"the gain is a thing the producer is aware of, not something that our formaat knows about ... ultaimtely qkv are stord
in the same scale format, and thats what the op uses".
1. q, k and v are each SCALED BLOCKS: a level's rows (or the values') unit-max in its storage FORM, and the row's max
   its fp32 scale. The forms are R17's, one a kind: a level's rows u8 -- every level, dense or sparse, a zero a zero
   and no floor under it -- and the values bf16. A routing entry is never negative (the den, the liveness words and
   the decay's shares need it); the factory refuses an entry its form does not hold.
2. The op is defined on the blocks' products alone, the same for every consumer and the fp64 oracle: no level is
   required to sum to 1, and nothing in the format says whether one does.
3. A producer's own quantities -- a learned gain, an activation that lands on the simplex -- are folded into the
   blocks by the producer, each block's scale formed inside the one job that holds its row, and are invisible to every
   kernel and to the oracle. A caller's own levels reach the format through the factory, which quantizes each against
   its max and nothing else.
4. A consumer that needs a scale-free quantity (the decay's per-leaf shares) computes it from the rows itself.

<a id="r24"></a>
## R24. The plan says what to do, never what the op is given

The op and every kernel under it execute a PLAN their caller supplies (Blake, 2026-11-11): the carve, each side's
walk of each level -- by its lists, or every digit -- the list groups, the window and the segment, the drain's width
and its scale's form. A plan field names an ACTION. It never names a property of the data -- dense, sparse, an
activation, a count of live digits -- and the op's inputs carry none but the
prefill's `density` (R9), which is no plan field and chooses a layout among exact forms: they carry shapes, dtypes,
scales and counts.
A level's sparse MASK is not a plan field but an OPERAND, data as its rows are (Blake, 2026-11-13): the op
reads no row to find what is live. On the prefill a bit of 0 is its supplier's guarantee that the entry's row is zero,
and a 1 guarantees nothing (Blake, 2026-11-24: "if it's zero you are guaranting to use this value is zero. If it's not
zero, you will get a wrong answer. If it's 1, it can be anything"); decode's op multiplies its rows by its mask's bits
until decode has its own design.
The producer knows what it gives, its sparsity too, and owns the rules that turn that into a plan; the layer calls
them, and any other caller supplies whatever valid plan it wants. The op refuses a plan the data's shape cannot
take and derives nothing from how the data was made. A plan is never wrong for the data in a way that changes the
result: a level walked by its lists that turns out dense, or walked whole where it was sparse, costs time and
nothing else. THE PLAN IS A ROLE, NOT AN OBJECT (Blake, 2026-11-22: "data chagnes the mathematics, plan changes the execution of those
mathematics. thats the exact spec we want"). No operator takes a plan object: it takes its data and nothing else -- `op(data...)` -- and the state is its own argument, in and out
(Blake, 2026-11-23: "have each internal call handle its descriptor stuff and not be exposed to the user, and have the public contract carry all
inputs, workpsace is something we build inside just to pass around"). The plan's host half is the descriptor, which each call builds from its inputs: the validated context of the call, every device fact and
every call fact, reusable and hashable; a call fact that states a choice names an action as above (a side of a level
walked by its lists), and everything derived from the facts -- the window, the grid, the segment -- is computed
by the family's one launch function right before the launch and stored nowhere. The plan's device half is what a
kernel decides during the call from the call's own data -- on the prefill the groups, the CTA's carve and the fold's
split, from the call's densities (R20) -- which never changes the result: forcing any of it on the same data matches the
oracle. Each call builds
its workspace itself, beside its descriptor, and its caller is trusted with the inputs it hands, the masks among them;
the layer takes only model inputs.
NOT YET SO: decode is on this surface, and so is the prefill's op; what a prefill launches is owed -- its launch function
(row A-f), the workspace's layout and builder (A-g) and the contract ranges they fix (A-h) of
[`open-work.md`](open-work.md#the-surface).

<a id="r25"></a>
## R25. A group is a layout; its steps are shaped to it, never wider than 64 columns

A GROUP of digits is a way of laying out data (Blake, 2026-11-22: "groups are just how we lay out in memory ... its a
way of laying out data that we then execute onto via properly shaped wgmmas, going down to low N and maxing at 64"): a
token live on several digits of a group is laid once, with its coefficients for those digits. Its first purpose is the
stream's bytes: a token on k digits of a group is k times fewer value lines than a list a digit, which is what lets the
digit stream scale with density (Blake, 2026-11-22: "grouping is largely for saving hbm traffic when
dense"). How a layout is EXECUTED is the kernel's: tensor steps of the shape that fits it, from narrow steps up to 64
columns and never wider. A layout wider than a step -- 8 or 16 digits -- is several 64-column steps whose A operand
names the one run of value lines, so the layout's width is the call's, from the side's density on the level (R10), and no step shape follows
from it. A step's shape pins its accumulator's registers, so the shape is chosen where no step is in flight: once a
call, from the call's densities (R20), before any window loop.
A group issues the number of steps its data needs, read from the packet: a loop on that count whose test is a warp vote
(Blake, 2026-11-22: "So voted loop achieved the goal with a small performance cost? Fine").
MEASURED: a step's own time is 32, 64 and 128 cycles at 64, 128 and 256 columns
([`lab/wgmma_width.md`](lab/wgmma_width.md)): past 64 a wider step buys no tensor time. A token on all 16 digits of its
box, executed as four 64-column steps on one run of value lines, is the four-digit window, exact; and a kernel holding a
per-list fold and a grouped fold, choosing by a word read before the loop, costs nothing a window
([`lab/carry_token.md`](lab/carry_token.md)).
The voted loop is 4,985 cycles a window against the cleared fixed block's 4,709 on the same packets, and exact on a packet whose
groups take four, two, one and one steps ([`lab/carry_token.md`](lab/carry_token.md)).
NOT YET SO: the loop is measured within one pass of the tiles' room, eight steps a box; a window that needs more is a
second pass, which is not built ([`open-work.md`](open-work.md)).
ON BLACKWELL (R8; Blake, 2026-11-22) the layout half holds unchanged. The step-shape half as written is hopper's: a wgmma
step's shape pins its accumulator's registers, and the 64-column price is an H100 record. Blackwell's widest step and
where its shape is chosen are owed with its bring-up ([`open-work.md`](open-work.md)). A group's count of steps is the
layout's and holds there; the form of the loop's test (a warp vote, what ptxas leaves unserialized on sm_90a) is hopper's,
owed with the same bring-up.

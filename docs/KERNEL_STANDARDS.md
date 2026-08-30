# KERNEL_STANDARDS — two tiers (Blake, 2026-08-28)

**CONSTITUTION** = Blake's standing rulings. Stable; changed only by Blake; they guide every law below.
They are: §10's correctness instruments (oracle = truth; bit-identity = refactor instrument); efficiency is
correctness; closed world (only measured binaries run); one form per mechanism, no fallbacks, no dead code;
build the whole design or stop and name the gap; the ancestral-code standard; §17 portability; ONE ATOMIC PRODUCT
(Blake, 2026-08-28: the repository in published form is ONE self-contained kernel product — never several kernels
being separately gated; every test lives in exactly one of the three tiers, `tests/unit`, `tests/integration`,
`tests/oracle` (docs/testing.md), plus `benchmarks/`; the whole tree is gated holistically at its current state;
no per-campaign test directories, and no campaign names in the tree — names are roles); SELF-DOCUMENTING
CODE (Blake, 2026-08-04: code is legible without comments; comments are sparse — a file header and 1-3-line decl
blocks naming purpose and the one invariant; derivations, histories and why-not-the-other-form prose live in
`docs/internals/` mirroring `csrc/`); and §R8-R10
(all architectures; no fuzzy-threshold branching — the kernel handles any distribution; design and probe
for ALL regimes at once — dense is the worst case and a test of the kernel, never a regime to concede).

**LAWS** = the coordinator's procedural and incident-derived rules (§1-§9 craft, §11-§16 and their addenda,
§R2-R7). Dated, adaptive, revisable by the coordinator; each carries the incident that made it and is
repealed when the situation changes. A law never contradicts the constitution; where a law is an inference
from one moment of development it is marked as such and re-examined at the next milestone.
§R1 ("price overheads against the sparse cell") is STRUCK 2026-08-28 — subsumed by §R10 and wrong in
direction: overhead concentrating at sparse may be the design paying off; dense remains the worst case.

# ROLA KERNEL STANDARDS — how we write CUDA here (draft for Blake's read, 2026-08-26)

SCOPE. This document is about CRAFT: how a designed piece is realized at the hardware's grain. It is not
about design. Design authority is `development/queue/P76_*.md` and Blake's rulings; nothing cited here is
an argument for or against a design choice. Where RoLA has patterns with no analogue in the reference
kernels (routed ownership, the two carves, the exchange, the fan-in ring, the cooperative expansion), §R
gives the RoLA rule directly. References are the local copies under a named sibling checkout,
`rola_references/` (`flash-attention`, `cutlass`, `ThunderKittens`); citations are `path:line`.

Every brief points here. Every review applies it line by line. A check in `ratify` (§9) is the backstop
and should catch nothing.

## 1. Register-resident data exists only as FRAGMENTS
The tensor core's unit in registers is the fragment: one MMA tile spread across the warp's 32 lanes in a
fixed thread→value layout. Anything held in registers across a loop is a stack of fragments, and code
addresses it by fragment modes — never by element, never by leaf.
- FA2 allocates the accumulator as a fragment shaped by the tiled MMA and keeps it across the whole K
  loop: `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h:194` (`partition_fragment_C(tiled_mma, …)  // MMA, MMA_M, MMA_K`).
- Its only inner loop is over a fragment mode: `csrc/flash_attn/src/utils.h:152-158` (`for (i < size<2>(tCrA)) … cute::gemm(…)`); softmax touches the accumulator through fragment modes only: `softmax.h:28-31, 100-113`.
- CuTe states the principle: an MMA atom IS three thread-value layouts — `cutlass/media/docs/cpp/cute/0t_mma_atom.md:168-172`; per-thread views fall out of the layout (`include/cute/atom/mma_atom.hpp:467-503`).
- ThunderKittens makes it a type: a register tile is a 2-D array of MMA-shaped base tiles
  (`ThunderKittens/include/types/register/rt.cuh:104`, `rt_base.cuh:3`); user code never writes element loops (`README.md:211, 217`).
RULE: a `static_for`/`#pragma unroll` nest over register-resident state iterates fragments (m-tiles × n-tiles ×
k-steps) with any finer index (leaf → slot, channel → row) folded into constexpr address math. The nest's
BODY COUNT must be stated as a function of the arm in the code's decl block and must be O(fragments).
COUNTER-EXAMPLE (S1, 2026-08-26): `page_regs` unrolled per LEAF × atom × channel-group → 4,096 bodies at the
D=4 8-warp arm; compile 5 → 50 min/arch; cicc at 6 GB segfaulting. Per-leaf iteration below the fragment
grain is a defect, not a style choice.

## 2. Reuse an accumulator as the next GEMM's operand by RELAYOUT, not by data movement
The fp32 accumulator of one GEMM becomes the A operand of the next by reinterpreting its layout in place
and narrowing the type — no store, no shuffle.
- FA2: `flash_fwd_kernel.h:378-380` (`make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()))` → `gemm_rs`); the layout math `utils.h:199-211`.
- FA3: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1156-1160` (same, plus an explicit register permute for fp8, and a documented fallback through SMEM when the register-sourced GMMA is unavailable, `:1161-1162`).
RoLA: the readout's A fragments ARE the fold accumulator's top halves (K35 M1a). Any phase change that
moves state must be the exchange the design names (§R3), never an ad-hoc copy.

## 3. Shared memory: copy atoms, swizzles by composition, transposition as a layout operation
- `ldmatrix` in both variants is a declared copy atom with an arch fallback chosen at the atom, not inline
  PTX at call sites: `csrc/flash_attn/src/kernel_traits.h:40-44` (`SM75_U32x4_LDSM_N`, `SM75_U16x8_LDSM_T`).
- The transposed atom feeds V as the B operand with no transpose pass: `flash_fwd_kernel.h:210`; copy and
  MMA thread mappings are derived from the SAME tiled MMA (`make_tiled_copy_B(…, tiled_mma)`) so they cannot drift.
- Bank-conflict-free layouts are a swizzle composed with a small atom and tiled to shape:
  `kernel_traits.h:79-90` (`composition(Swizzle<kSwizzle,3,3>{}, Layout<Shape<_8, kBlockKSmem>>)`), the swizzle
  parameter derived from the row width. A transposed view is a COMPOSITION of the same layout
  (`kernel_traits.h:92-95`) — zero data movement.
RULE: every SMEM tile has a named layout with its swizzle derived from its access pattern (writer's and
reader's), and a ROUND-TRIP test proving bit-exactness and conflict-freedom (the existing plane gate).
Changing an access pattern (e.g. 2-byte scattered stores) re-derives the swizzle and re-runs the gate.
Primitives (`ldmatrix`, `cp.async`, `mma`, `red`) are wrapped ONCE (`csrc/rola/src/common/ops.cuh`); call sites use the wrapper.

## 4. Pipelines: explicit stage state, release immediately after consumption, two stages in flight
- FA3's producer acquires a stage before issuing, the consumer waits, issues the GEMM, waits for it, and
  releases the stage at once: `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:750, 1140-1143`; pipeline types per
  operand `:283-286`; the invariant is documented where it bites (`:787`).
RULE: overlap requires two stages in flight in ANY form (F4 on P76): the second stage is registers
(single-slot segments: drain by `ldmatrix`, refill, compute from fragments) or a second SMEM slot. Sync
carries exactly two facts, "landed" and "drained"; on sm_80 by the per-batch barrier (form (b)), on
sm_90 by mbarrier. No pipeline state is implicit in loop parity without a comment naming the invariant.

## 5. Warp specialization only as register economy
- FA3 gives the producer warpgroup ~24-56 registers and the consumers 160-256 via `setmaxnreg`
  (`hopper/flash_fwd_kernel_sm90.h:82-83, 308-309, 360-361`), with the budgets `constexpr` per configuration.
RULE: on sm_80 there is no register redistribution, so a fetcher warp burns a full slice and folds nothing —
form (b) (role-alternating) everywhere; specialization is the sm_90+ mapping and only where the
accumulator binds (F4).

## 6. Unroll discipline
- CuTe pairs `CUTE_UNROLL` with `CUTE_NO_UNROLL` (`cutlass/include/cute/config.hpp:51-55`): saying "do not
  unroll" is as deliberate as saying "unroll". Unrolls are over fragment modes with compile-time bounds
  (`include/cute/algorithm/gemm.hpp:293-320`); FA2 unrolls a loop only when its trip count is `constexpr` (`flash_fwd_kernel.h:308`).
RULE: unroll = compile-time-bounded loops over fragment modes, nothing else. Dynamic loops (windows,
batches, tokens) carry `#pragma unroll 1` explicitly. `static_for` is a fragment-mode tool; a `static_for`
whose bound is a leaf count, a channel count, or a product of them is wrong.

## 7. Instantiation hygiene: one instantiation per translation unit, generated, prunable, budgeted
- FA2: each `.cu` holds ONE explicit specialization and says why: `csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu:2-3`
  ("Splitting the different head dimensions to different files to speed up compilation. This file is
  auto-generated. See generate_kernels.py"); the generator `generate_kernels.py:12-50`; shared sub-trees are
  `extern template` so they are not re-instantiated (`:44-48`).
- FA3: 451 generated TUs under `hopper/instantiations/`; feature axes deliberately COLLAPSED to cut the
  cross-product (`hopper/generate_kernels.py:99, 108`); every TU guarded by `#ifndef FLASHATTENTION_DISABLE_*`
  so a subset builds.
- FA's build sizes parallelism to memory because instantiation cost is the dominant resource:
  `setup.py:708-715` ("peak_threads * 5GB <= free_memory").
- CUTLASS emits one `.cu` per configuration with an aggregator that only declares initializers
  (`cutlass/python/cutlass_library/manifest.py:290-330`) and prunes the instantiated set for compile cost
  (`media/docs/cpp/profiler.md:14`; `quickstart.md:609`).
RULE: one arm per TU, generated from the closed-world arm list; an ARM-SUBSET build knob (declared list,
default all) for iteration; iteration builds are single-arch AND arm-subset; both-arch all-arm builds
once per stage. Each arm has a COMPILE BUDGET (wall time, cicc peak RSS) recorded in the manifest.

## 8. Arch mapping at the atom, closed-world at the arm
Feature selection lives where the primitive is defined (`kernel_traits.h:40-44`'s `DefaultCopy` fallback),
never as `#if` at call sites. RoLA adds: every structural decision states its sm_80/90/100 mapping on the
card (P76 §4); only measured binaries run (pinned ptxas, per-arch ratified manifests, device stamp).

## 9. The backstop (checks that should catch nothing)
`ratify` records per arm and arch: registers, stack, spills, residency (existing) + compile wall time,
cicc peak RSS, and the body count of each named `static_for` nest as a function of the arm (new). A budget
breach fails the gate exactly as a spill does. The device stamp is asserted before any gate.

## 10. Correctness instruments (unchanged, restated so they sit beside the craft rules)
The fp64 oracle is the truth. Bit-identity against a previous binary is a REFACTOR instrument, never a
design constraint (P76 amendment 2026-08-26). No fallbacks, no dead code, no special-case paths; comments
are file-header/decl blocks only, prose goes to `docs/internals/` mirroring `csrc/`. Build the whole design
or stop and name the gap; efficiency is correctness.

## R. RoLA-specific patterns (no reference analogue) — the rule comes from P76
- R1 OWNERSHIP CARVES: a stream owns an affine rectangle of digits per level (per-side span vectors);
  the private slot order puts carved (sparse) levels outermost; liveness is the exact AND over levels.
  Executors ≠ streams (an executor folds the union of its streams' boxes).
- R2 COOPERATIVE FETCH + EXPANSION (F2/F4): one union walk per batch; writer-side placement into per-stream
  segments of value rows AND fully expanded coefficient rows; memo internal to the fetcher; consumers are
  pure MMA. The per-tile liveness mask is produced by the expansion (it knows which slots it wrote nonzero).
- R3 THE EXCHANGE (order law): every phase change re-lays the state into the other side's order on every
  arch; publisher places each leaf's row at the READER's slot (per-reader `[leaf][channel]` segments,
  16-bit stores at compile-time addresses); the reader pulls its contiguous segment with `ldmatrix.trans`
  and ejects. No pair-level or run-alignment constraint on carves.
- R4 THE FAN-IN RING (§2.6-C): union rank keys a C-row SMEM ring; first writer stores, others `red.shared`;
  the NEXT writer reclaims a complete slot by CAS and flushes it; spin on incomplete; drain at the phase barrier.
- R5 LO PARK (F5): hi in registers, lo parked into the transit's drained slots, freed registers become
  leaves (box size); a per-arch/width declaration.

## 11. Debugging a device hang or race (recorded 2026-08-26 after S2d burned 2.5 h reasoning blind)
`cuda-gdb` works on this machine when the process is LAUNCHED under it; ATTACHING (`-p`) is refused by
Yama `ptrace_scope=1` (a Linux setting, liftable with `sudo sysctl kernel.yama.ptrace_scope=0`).
Recipe: `flock /tmp/rola_gpu.lock cuda-gdb -q --args <venv>/bin/python repro.py` → `run` → Ctrl-C on a
hang → `info cuda kernels` / `info cuda warps` / `cuda kernel 0 block (..) thread (..)` / `bt` / `print`.
`compute-sanitizer --tool synccheck|racecheck|memcheck` for barrier misuse, shared-memory hazards and
addressing (it found the decode pool race). Device `printf` flushes only on kernel completion — useless
for a hang; a bounded loop + `__trap()` converts a hang into a fault but skips barriers, so prefer the
debugger. RULE: a hang or a run-to-run-varying miscount is diagnosed with the debugger or the sanitizer
within the first hour, never by inspection alone; the finding goes in FINDINGS with the debugger's output.

## 12. Warp collectives and fences (recorded 2026-08-26 from S2d/S2d-2)
- NEVER place a warp-collective (`__shfl_sync`, `__ballot_sync`, `ldmatrix`, `__syncwarp`) inside a
  short-circuit or a lane-divergent predicate: `part && lane_broadcast(x)` splits the warp at an
  instruction that needs every lane, and the split never completes (a hang that survives removing every
  explicit wait). Hoist the collective; predicate its RESULT.
- A fence orders only the FENCING THREAD's own accesses. In any publish/consume pair (data -> counter,
  counter -> data), EVERY lane that issued the data accesses executes the fence before the `__syncwarp`
  that precedes the one-lane counter update; and EVERY lane that reads the data after a one-lane
  counter check executes a fence after the broadcast and before its loads. A leader-only fence covers
  the leader only (S2d bug 1: seven of eight lanes unordered).
- Debugging recipe under WSL (cuda-gdb launch mode, attach is refused by Yama): run gdb with a FIFO on
  stdin (`mkfifo in; cuda-gdb -q --args <venv>/bin/python repro.py < in &`), `echo run > in`, wait for
  the hang, `kill -INT <gdb pid>`, then `echo 'info cuda kernels' > in`, `info cuda threads`, `bt` —
  it names the split warp and the PC in one shot.

## 13. A mechanism's brief carries the WHOLE protocol (recorded 2026-08-26 after the ring's four patches)
When a synchronization protocol or data-path mechanism changes, the brief restates the ENTIRE mechanism
as pseudocode and the implementer rebuilds that function from it — deleting the previous one — never
editing around the previous protocol's skeleton. Scaffolding that only made sense for a superseded
protocol (a deferral rule, a sentinel state, a counter) must not survive; the review checks the function
against the pseudocode line by line. "Build the whole design" applies at mechanism grain.
Addendum (S2d-3): a whole rebuild is diffed against the BUG LIST of the thing it replaces, not only against
the pseudocode — S2d-3 reintroduced S2d-2's `part && lane_broadcast(...)` short-circuit hang while
rewriting the acquire, three lines below its own warning. The previous pass's findings are a checklist
the rebuild is walked against before its first build.

## 14. Waiting on a build (recorded 2026-08-26 from the agent time ledger: 327 min of `until pgrep … sleep` loops)
A long build is launched as a BACKGROUND task writing to a log file, and the agent then ENDS ITS TURN —
the harness re-invokes it when the command exits, with the exit status. Never wait by polling: no
`until pgrep … sleep` loops, no `pgrep -f` on a process name (it self-matches, it matches other tracks'
builds in the same venv, and a live waiter re-wakes a stopped agent into a worktree it no longer owns).
Never edit a source file while a build of it is in flight. One gate build per stage; iteration builds
single-arch and arm-subset.
§14 addendum (Blake: background tasks have died silently before): the launch records the build's PID and
appends an exit marker (`echo BUILD_EXIT=$? >> log`) as the command's last step; on re-invocation for ANY
reason the agent reads the marker first, never trusts the notification alone. The wait itself is a
BOUNDED TIMER with early exit — `timeout 600 tail --pid=$PID -f /dev/null` (≤10 min, returns at once when
the build ends) — after which the agent INSPECTS ITS OWN STATE (the marker, the build dir) and decides;
repeat if still building. Never trust the pid or the notification alone; the marker is the fact. Name-matched polling (`pgrep -f`) stays forbidden: it self-matches, matches other tracks, and
re-wakes stopped agents.

## 15. Stage cycle budget (recorded 2026-08-27 from the ledger: S2d-3 ran 63 builds and 22 ratifies in 346 min)
- RATIFY ONCE per stage, after the last source edit. It recompiles every TU on both arches; it is a gate,
  never an iteration tool.
- A stage has a budget of THREE full build+gate cycles. Iteration is single-arch, arm-subset, oracle-first.
  When the budget is spent and a gate is still red, the stage STOPS and reports the residual — it does not
  keep chasing green.
- MID-STAGE RE-SCOPING IS ALLOWED when it changes WHAT is built (a design or mechanism ruling) — Blake:
  "I'd rather re-scope than build something wrong and fix it later." A re-scoped stage restarts its
  cycle budget from the amended spec. Amendments that do not change the design (count formats, report
  shape, housekeeping) go in the NEXT brief instead.
- Every kernel family carries an arm-subset knob (the carry's ROLA_CARRY_ARMS); a family without one gets it
  before its next build.
§15 addendum (build parallelism, 2026-08-27): cicc peaks ~1 GB post-S1, on a 16-CPU / 24 GB machine.
Gate builds: exclusive lock, MAX_JOBS=8, NVCC_THREADS=2. Iteration builds: one of two parallel slots
(`tools/build/build_lock.sh`), MAX_JOBS=4, single-arch, arm-subset. Re-derive these numbers if
cicc's peak changes (a per-leaf nest can push it back to 6 GB — §1).

## §16 — Ratify at milestones, not stages (Blake, 2026-08-28)

A stage's gate is SMALL: an iteration build (single arch, arm subset), the fp64 oracle on the arms
it touched, the census on those arms, and the model tests. A stage does NOT run the both-arch
all-arm gate build and does NOT ratify. Manifests go stale between milestones by design; a stage
commit says "unratified" in its title. RATIFICATION — the full closed-world measurement — runs
ONCE per cohesive product: at the milestone that merges (the receipt stage), and after the last
edit of any kind. Rationale: a 40-minute ratify (plus a 20-35 minute gate build) after every
small piece was 1-2 hours of waiting per stage; the closed-world guarantee is a property of what
SHIPS, and nothing ships between milestones. The stage-level guarantee is the oracle.
Corollary: a stage that needs the SASS-body/instantiation gates for its OWN claim (a byte-identity
refactor) runs `ratify --check` against the LAST milestone's manifest on its subset only.
§16 addendum (Blake, 2026-08-28) — EVERY STAGE GATE INCLUDES A PERF PROBE. Since the stage compiles
anyway, its gate runs the fixed two-cell probe (dense flagship k_tok=None + k16 scattered; L=768;
the stage's touched arm; ncu carry `gpu__time_duration` 3 warm + the step wall) and records it in
FINDINGS as a running per-stage ledger next to the previous stage's numbers and the mainline
reference. A >1.10x regression on either cell vs the previous stage is a RED like a spill: the stage
reports it, does not rationalize it, and Blake rules. This is the instrument that would have caught
the 3-9x loss PROBE-K35 found after nine stages.
§16 addendum 2 (2026-08-28, from R-A): AN INSTRUCTION LEDGER IS NECESSARY, NOT SUFFICIENT. The M1a ->
S1-parent span cut instructions 5.6% and cost 22% of time (IPC -27%, long-scoreboard x3.6, bank
conflicts x2.6). The stage probe is TIME; every ledger table carries IPC and the top stall reasons
(ncu `smsp__warp_issue_stalled_*`) beside instruction counts, and a stage whose instructions fall while
its time rises is RED, not a win.
§16 addendum 3 (Blake, 2026-08-28) — THE COMPILE LEDGER. Build time is a gated number like the census.
Every stage gate records, per built arm TU, the compile wall and cicc peak RSS beside the previous
stage's; >1.2x on any arm is RED (report, do not rationalize; diagnose against §7 first). Arm TUs never
include torch headers (the ~34 s header floor lives in the dispatch TU only). Arms are added as reviewed
rows in the one arm list (TEST-tagged unless shipped). Ratify stays incremental per arm.

## §R addenda — LAWS (coordinator, 2026-08-28; incident-derived, revisable; R1 struck)
R1. [STRUCK 2026-08-28 — subsumed by §R10; see the preamble.] PRICE OVERHEADS AGAINST THE SPARSE CELL. A fixed per-window / per-contribution cost that is "cheap"
    next to a dense window's MMA becomes the kernel once sparsity removes that MMA (the ring: 230 M
    instructions at dense AND at k16; the replicated schedule; the per-lane gate). The design's payoff and
    its overheads scale with different variables; state both per mechanism.
R2. THE FAN-IN'S FIGURE OF MERIT IS 128 B SEGMENTS PER RED INSTRUCTION, not sector fullness and not
    instruction count: one token's 32 contiguous channels per issue (M1a's panel layout) = 1 segment;
    4 rows x 8 channels = 4 segments at identical RED count = +93%; fragment-direct = 8 half sectors.
R3. FAN-IN ON AMPERE IS NATIVE GLOBAL RED FROM AN ALIGNED LAYOUT. Shared-memory fp32 atomics are a CAS
    loop; a locked SMEM ring paid 18x its floor to halve DRAM. The DRAM question belongs to the A100
    parity-curve receipt, not to the local cell.
R4. NO PER-WARP WORK THAT ANOTHER WARP ALSO DOES (F2): CTA-identical derivations (union lists, owner
    ballots) are derived once and published behind a barrier the kernel already pays, or split by
    executor share — never recomputed per warp to avoid a barrier.
R5. THE SCHEDULE HAS THREE PARTS (fetch over TOKENS once; per-stream SENDS from each stream's own ballot;
    seal-and-consume). A union over (token, stream) pairs collapses the first two and re-fetches k times.
R6. A KERNEL'S SASS DEPENDS ON ITS TU NEIGHBORS: byte-identity is a claim about a partition, not a body;
    per-arm TUs make each body canonical (K36: 7 movers, proven inert and free).
R7. STAGED-vs-STRAIGHT IS DECIDED BY LATENCY, NOT COUNTS: "identical sectors and issues" lost 9.5% on
    long-scoreboard (e390764). Time, IPC and stall reasons decide; counts describe.

## §17 — Portability: what the repo points at (Blake, 2026-08-28)
The repository is SELF-CONTAINED: anything it DEPENDS ON — a tool it runs, a doc it cites, a record it
reads, a baseline it compares against — lives in the repo or in a NAMED sibling repository with a stable
layout (the bench/results repo), never in an ad-hoc place (a scratch directory, a home path, a session
scratchpad, a campaign journal, a worktree path). Ad-hoc material never enters the repo either — no
script blowup: briefs, transcripts, per-session scripts and campaign journals stay outside and are
cited FROM the queue, not from repo files. Absolute paths under a home directory are a lint finding.
Committed records carry repo-relative or host-keyed locations. Skills that ship with the repo point at
repo docs and the measurements record, never at a journal.

## §R — CONSTITUTION (Blake, 2026-08-28)
R8. DESIGN MECHANISMS THAT EXPAND TO ALL ARCHITECTURES. Every mechanism states its sm_80 / sm_90 /
    sm_100 form at design time (wgmma/tcgen05 SMEM- and TMEM-sourced operands, TMA box loads, clusters
    and DSM, vector RED); a mechanism that only exists on one arch is a workaround, not a design.
R9. NEVER BRANCH ON A FUZZY THRESHOLD. No dispatch, launch policy or in-kernel path keys on a measured
    statistic (sparsity fraction, density, live count, "small L"). Branch on STRUCTURE the declaration
    fixes (tile counts, modes, arm axes). The kernel itself handles any data distribution.
R10. DESIGN AND CODE FOR ALL REGIMES AT ONCE — dense, sparse, mixed, clustered, small and large L, small
    and large N. Never overfit a mechanism, a measurement or a verdict to one cell: every design states
    its cost in each regime, every stage is probed at both ends of the distribution, and a mechanism
    that wins one regime by losing another is a finding to rule on, not a win.

## CONSTITUTION addendum (Blake, 2026-08-28): NO FALLBACKS AT THE ATOM
A primitive may select DIFFERENT NATIVE INSTRUCTIONS per architecture; it may never select a wider-type
emulation on any target architecture. An operation an architecture genuinely lacks is a declared design
fact with its cost on the card, never an `#if` branch. (Origin: mul_bf16x2's fp32 emulation on sm_80
lived for months behind a passing oracle; fma.rn.bf16x2 was native all along — K42.)

## §14 addendum (law, 2026-08-28, from K37): CONTAINERS AND THE GPU LOCK
`flock /tmp/rola_gpu.lock` is per-filesystem: a lock taken inside a container serializes nothing on the
host or in another container. Every container that touches the GPU bind-mounts the host's lock
directory so one flock governs all. One gate container at a time: stop the previous run before
launching the next (three concurrent in-container probes turned a minutes-long gate into an hour).

## §18 — Tools refuse, they do not choose (law, 2026-08-28, from Blake's "why keep two linkers")
A tool or build step that could run one of two ways — a linker present or not, a cache pinned or not,
a lint with or without its compdb, a lock taken or a slot — REFUSES with the command that fixes it; it
never silently takes the other path. Legitimate MODES (the §16 strict-manifest tiering; a comparison
arm kept as a yardstick) are keyed on an explicit declared variable and carry their exit condition on
a card. Applied: mold as the only linker; sccache pin miss refuses; run_clang_tidy never skips; one
build-lock discipline (slots as the parameter).

## §10 addendum (law, 2026-08-28): THE COMMENT-DENSITY GATE
The constitution's self-documenting-code clause is gated mechanically: `tools/lint/lint_standards.py`
measures comment lines / total lines per file under csrc/ and is RED above the mandate's threshold
(named constant; decl-block level, well under 15%). A `//:` prose block outside a file header is a
finding. A stage never raises a file's ratio; S5b moves existing prose to the mirrored docs
(byte-identical SASS gate). Measured 2026-08-28 before S5b: carry.cu 48%, ops.cuh 43%, box.cuh 41%,
carry_kernel.cuh 35%, decode.cu 32%, facts_kernel.cuh 32%.

## §14 addendum 2 (law, 2026-08-28, after Blake had to shut WSL down): ONE MACHINE, ONE COMPILE BUDGET
Three tooling agents each ran a full rebuild concurrently (MAX_JOBS=4 each -> 12+ cicc + 3 links) and
pegged the host at 100% CPU for 10+ minutes. The compile budget is MACHINE-WIDE and TOOL-ENFORCED, not
brief-enforced: setup.py (via the one build-lock discipline, K39) takes a machine-wide slot before
compiling and REFUSES/waits when the slots are full — a brief cannot forget it. Until that lands: at
most TWO agents that build at once; every build goes through tools/build/build_lock.sh; an agent checks
`pgrep -c cicc` before launching and never wraps a full build in hyperfine (bounded `--runs 3`, on the
LINK step only). Coordinator error recorded: the K41/K42 briefs omitted the build lock.

## §18 addendum (law, 2026-08-29): `--no-verify` is never a precedent
A commit bypasses the hooks only with an explicit, per-commit coordinator authorization naming the
pre-existing hook failure and the stage that fixes it. "Another stage did it" is not authorization. Red
hooks are fixed (the hook, or the tree), never routed around.

## CONSTITUTION addendum (Blake, 2026-08-29): THE FOUNDATION LAWS (cards K50, P80c, P79, G_FOUNDATION)
- R11 THE AXIS LAW: a compile-time axis is legitimate only as a deterministic function of user input the
  selector can pick; everything else is derived or a constant. Free axes = D and DV. MO and per-level B
  are runtime addressing; BC, K, M, kWarps, S are derived; W and C are constants.
- R12 SHAPE COMPILE-TIME, ADDRESSING RUNTIME: only the register shape (leaves-per-warp × DV) is compiled;
  every SMEM tenant is launch-sized from the state's descriptor; residency is asserted at the seam as a
  refusal; no runtime trip count in a register-shaped loop; never max-size SMEM.
- R13 THE STATE OWNS ITS FORMAT: a descriptor (D, per-level B_l powers of two ≥ 16, canonical order,
  the page rectangle = trailing 4 canonical bits, DV, split bf16 planes) bound at first call and CHECKED
  after; the warp sub-box's spans are compile-time from (D, DV) and never exceed a level's width under
  the floor (D=1 requires B ≥ 32); prefill and decode share ONE
  admission law; N is a multiple of 16 (ragged N refused, never padded); pages are [hi][lo][mass] bf16
  planes; readout operands are the hi plane in every kernel; a reading token moves hi only.
- R14 OWNERSHIP GRAINS NEST ON THE PAGE, nothing more: decode's unit is the atom; prefill's box owns whole
  rows; the kernel consumes activity bits at its own (warp-box) grain and the page table derives its own.
- R15 THE CTA IS THE SM (target): one CTA per SM with the warps the register file admits (8 × 256 regs),
  streams partition inside it; latency hiding is the kernel's job (covers, split-phase edges) — exposed
  latency is a coverage finding, never grounds to split the CTA; ONE CTA per SM is the goal, not a
  measured choice — a two-CTA launch is only a diagnostic of how far the body is from the target.
- R16 SHIP WITHOUT ARM REGISTRATION: a Python-package user never registers an arm; the shipped set is
  D × DV per arch; unknown descriptors are admitted by refusal-checked derivation, not by enumeration.
- R17 BF16 EVERYTHING: operands are bf16 (one form; float operand paths deleted); state accumulates fp32
  as split planes; truncation at the readout operand is the declared arithmetic.
- R18 ONE BODY PER OP FAMILY: the split (channel-carve) body is deleted; dense is the S = 1 point of the
  one body; a tied carve is a transit with the identity map, never a second body.
- R19 LAWS LAND WITH THEIR STAGE: a ruling that changes a law is written here in the same stage that
  implements it, never only on a card; the milestone carries the drift guards (lints, docs rev, skills).

## CONSTITUTION addendum (Blake, 2026-08-29): R9 ENFORCEMENT — A STRUCTURE BRANCH FAILS LOUD AT EVERY LAYER
A CTA-uniform branch on structure (e.g. a switch over admissible (span, shift) pairs) is legal only with:
(1) its cases GENERATED at compile time from the arm's shape, never hand-enumerated, with a static_assert
on the count; (2) a compile-time exhaustiveness assert (the derivation's possible outputs ⊆ the covered
set) so an arm that could receive an uncovered case does not build; (3) a launch-time refusal at the seam
naming the uncovered case; (4) a trapping default (`__trap()`, never a fallthrough) and a post-build
coverage sweep (every arm at every admissible case on a tiny cell).

## CONSTITUTION addendum (Blake, 2026-08-29): R13 — TWO CONTRACTS, ONE TRANSLATOR
The API accepts any shape (any b_l ≥ 2, any d_v, any L) and serves it by padding; the KERNEL accepts only
the descriptor's strict shape (powers of two ≥ 16, a shipped DV, N a multiple of 16, bf16 operands) and
REFUSES the rest — it never pads or masks. The seam is the only translator. Both boundaries are tested:
API tests prove padded runs bit-identical; kernel-boundary tests call the launch surface with unpadded
shapes and assert the refusal; a lint keeps logical-width fields out of every kernel entry signature.

## CONSTITUTION addendum (Blake, 2026-08-29): R9 ENFORCEMENT IS ONE PRIMITIVE
Every CTA-uniform structure branch in device code is a `uniform_switch<Set>(value, body)`
(`common/structure_switch.cuh`): the four layers — generated Set with a count assert, compile-time
exhaustiveness (derivation outputs ⊆ Set), the seam's launch refusal reading the same Set, a `__trap()`
default with the post-build coverage sweep — are the primitive's contract, never re-implemented per site.
A hand-written device `switch` on a geometry field is a lint finding.

## CONSTITUTION addendum (Blake, 2026-08-29): R20 — THE SM OWNS THE STATE; WARPS PARTITION IT; CTAs GROUP WARPS
On every architecture the unit of state ownership is the multiprocessor: its box is derived on its own
from the architecture's state capacity (`leaves_per_sm`, from DV) balanced over the D levels. Warps
partition that box — each side's level order carves it, sparse levels first, until every warp owns a
segment — and a CTA is only a grouping of warps holding a contiguous subset of those segments (how many
is a launch parameter). Where the accumulator lives (registers, tensor memory) changes how a warp touches
its segment, never who owns what. Every mechanism — the carve, the transit, the activity bits, the
launch shape, the schedule — is stated in these terms, and a design that couples ownership to the CTA
or to the accumulator's home is misfiled.

## CONSTITUTION amendment (Blake, 2026-08-29): R20 — THE OWNERSHIP UNIT IS THE CTA's BOX
R20 read "the SM owns the state; CTAs group warps". Corrected by the arithmetic (G2c): a CTA loads and
stores its box at entry and exit, so its read-side warps' union and its write-side warps' union must be
the SAME leaves; carving the SM's box first and handing a CTA half the segments gives the two sides
different halves at two CTAs per SM. Therefore the ownership unit is the CTA's box — carved per side
among the CTA's warps — and it equals the SM's box exactly when there is one CTA per SM (the goal).
The shape set is f(D, leaves_per_warp, kWarps); the architecture law (the SM's box) is asserted at
kWarps = warps_per_sm.

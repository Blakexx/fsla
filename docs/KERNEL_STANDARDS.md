# ROLA KERNEL STANDARDS — how we write CUDA here (draft for Blake's read, 2026-08-26)

SCOPE. This document is about CRAFT: how a designed piece is realized at the hardware's grain. It is not
about design. Design authority is `development/queue/P76_*.md` and Blake's rulings; nothing cited here is
an argument for or against a design choice. Where RoLA has patterns with no analogue in the reference
kernels (routed ownership, the two carves, the exchange, the fan-in ring, the cooperative expansion), §R
gives the RoLA rule directly. References are the local copies under `/home/blake/rola_references/`
(`flash-attention`, `cutlass`, `ThunderKittens`); citations are `path:line`.

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
reason the agent reads the marker first, never trusts the notification alone. If an explicit wait is
needed it is ONE bounded, pid-specific call — `timeout 1800 tail --pid=$PID -f /dev/null` — then the
marker. Name-matched polling (`pgrep -f`) stays forbidden: it self-matches, matches other tracks, and
re-wakes stopped agents.

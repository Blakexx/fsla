# The planner

> **Derive launch geometry from launch-static facts. Never score candidates,
> never read a routing statistic to decide how to run.**

The planner's whole output is one integer: `BC`, the number of leaves one owner
block owns — the CTA's resident state tile. Nothing else is planned. The
consumer instantiates one executor, the token tile is the fixed `GROUP_TOKENS`
quantum (`rola/planner/derive.py`), and every other axis is a model fact
(`topology`, `normalization`, `decay`) rather than a decision. So `BC` is not
chosen AMONG candidates by a cost model or an argmin; it is DERIVED, in
`rola/planner/derive.py`, as a closed form over inputs that are fixed before the
routing is computed:

```
ladder  = feasible_bc(widths, instantiated=box_admitted(machine, d_v, D, decay))
ceiling = min(subtree_ceiling(widths), fill_ceiling(N, BH, sm_count))
BC      = the largest member of `ladder` at or below `ceiling`,
          or `ladder`'s smallest member when none is
```

`derive_bc` evaluates three independent ceilings and takes the largest compiled
block that clears all three. None of them is a threshold on a measured
distribution — each is a property of the topology or the machine, true before
any token is routed, which is why a plan cannot go stale within a launch and
there is nothing left to calibrate.

## 1. The three ceilings

**`subtree_ceiling(widths) = widths[-1]`** — how large a block can be before it
owns leaves from INDEPENDENT SUBTREES. Leaves are the MSB-first mixed-radix
enumeration of the level digits, so an aligned block of `BC <= widths[-1]`
leaves shares every digit above the innermost level: it is a slice of one
parent's children. A block wider than that spans two or more distinct digit
tuples above the innermost level — it unions leaves whose supports are
unrelated, so the union's candidate bitmap is their OR and the block is
visited whenever *either* subtree is live, paying the whole `BC` k-extent for
whichever subset was actually routed. A property of the tree alone, evaluated
on `widths`. It would be FALSIFIED by a measurement showing a block spanning
independent subtrees retiring no more work than the subtree-aligned blocks it
replaces at the same realized support.

**`box_admitted(machine, d_v, D, decay)`** — the compiled `BC` values whose
`constexpr` shared-memory carve fits the per-CTA budget
(`cudaFuncAttributeMaxDynamicSharedMemorySize`). The carve is read off
the dispatch table of the *built binary* — `MachineFacts.arm_smem_bytes`, a
`{(BC, d_v, D, decay): bytes}` map taken from `DispatchTable.from_extension`
after the staleness join — so the box arithmetic is the kernel's own
arithmetic, not a model of it. This is a FEASIBILITY ceiling, not a
preference: a `BC` it refuses is one the driver would refuse to launch, so
there is no candidate above it to prefer. An empty result is a loud
`ValueError` naming every carve and the device's grant, never a smaller block
substituted silently — refusing everything is the correct answer when nothing
fits.

The cooked schedule ring is ALREADY IN that number — `kSchedSlots` group
descriptors, their per-slot row scales and the producer's cursor state are
carved by `consumer_smem_bytes` beside the state slab. So the trade the box
arbitrates is not modelled anywhere: a deeper `ROLA_SCHED_DEPTH` and a wider
`BC` compete inside ONE compiled number, and the box responds to a depth change
with no constant to keep in step with it. Rings are what replace co-residency
as the latency-hiding mechanism, which is why residency is not a term here:
the executor hides latency inside one CTA.

**`fill_ceiling(N, BH, sm_count) = BH*N // (sm_count * WAVES_TARGET)`**, with
`WAVES_TARGET = 2` — how large a block still leaves enough CTAs to fill the
machine. The launch is one CTA per `(batch, head, owner)`, i.e. `BH*N/BC`
CTAs, and `BC` is the only term that shrinks that count. `BH = batch * heads`
is launch-static, not a routing statistic, so this term cannot drift within a
launch. `WAVES_TARGET` is a TARGET, not a floor: a topology whose whole grid
is smaller than the machine cannot be made to fill it by any `BC`, so the
derivation falls back to the smallest compiled block rather than refusing —
the guard's job is to stop a large block from starving a grid that would
otherwise have filled the machine, not to punish a grid that structurally
cannot.

## 2. The clamp, and which constraints refuse

The ceilings bound a raw integer; `derive_bc` clamps it DOWNWARD onto the
dispatch table's own `BC` axis — the formula picks among rows the binary
carries and never invents one.

Two of the four constraints REFUSE and two SATURATE, and the split is what
each one means. `box_admitted` and `feasible_bc` are FEASIBILITY: an arm that
does not fit shared memory and a `BC` that names no owner rectangle cannot be
launched at all, so an empty set is a loud `ValueError` and there is nothing
below it to fall back to (`widths = (4, 6)` against the `{16, 64}` ladder — 16
does not divide 24 — has no plan, and says so). `subtree_ceiling` and
`fill_ceiling` are EFFICIENCY: the kernel runs correctly beneath them, they
bound how wide a block SHOULD be rather than whether one exists, and the
smallest compiled block is the closest the ladder comes to satisfying them. So
a topology whose innermost level is narrower than one MMA quantum, or a grid
too small to fill any machine, derives the ladder's floor — not a refusal.

## 3. The expert pin

`PlanOverrides.state_block` (`rola/expert.py`) REPLACES the derivation outright
rather than biasing it: when set, `derive_bc` skips all three ceilings and
returns the pinned value after the same alignment validation every derived
value gets. That is what makes it a pin over a derived axis and not an escape
from one — an inadmissible pin is an error on the line that wrote it, not a
plan-time surprise.

## 4. What the derivation reads

`MachineFacts` (`rola/planner/derive.py`) is everything the formula is allowed
to know about the machine — `sm_count`, `max_dynamic_smem_bytes`, and
`arm_smem_bytes` — measured once per `(device, install)` via
`MachineFacts.for_device` and carried in, never queried ambiently from inside
`derive_bc`. That is what keeps the function pure: the same inputs give the
same plan, and a CPU-only gate can supply a synthetic `MachineFacts` instead of
mocking a device.

`rola/planner/schedule.py` turns a derived or pinned `BC` into the launchable
objects: `build_plan(topology, BC, *, normalization, decay, sm_shared_bytes=None)
-> Plan` fills the plan's derived fields (the warp tile, the SMEM budget) over
one `BC`, and `plan_and_schedule(...) -> (Plan, Schedule)` builds the plan and
its §4 schedule from one pass over the routing tensors.

## 5. What the exact statistics are for

The tile-union census (`rola/planner/calibration/census.py`) and the exact
per-`(owner, tile)` statistics (`rola/planner/calibration/union_stats.py`) are
BENCH MEASUREMENT INSTRUMENTS: they characterize a realized routing for a
benchmark or a calibration sweep to read, and no dispatch path reaches them.
They answer "what did this routing do", which is a question about a step that
already ran, not an input the planner is allowed to consult before it decides
how to run one.

## 6. Where this sits relative to the field

A closed-form derivation over launch-static facts is the same category of
answer FlashAttention, xformers and TensorRT-LLM give: they encode "which
kernel arm" as a static condition on shape, dtype and architecture, and none
of them read a per-step distribution to decide. RoLA's routing IS a per-step
distribution, but `BC` — the one quantity this planner plans — does not need
to see it: the geometry that would starve or overflow at a given topology and
machine is a property of the topology and the machine, decided once, the same
way it would be for a shape-keyed dispatch table.

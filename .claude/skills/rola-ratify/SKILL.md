---
name: rola-ratify
description: Running tools/ratify.py — milestone-only closed-world codegen ratification, --arm scoping, incremental cache, reading register/SASS movers, the device stamp
---
Read KERNEL_STANDARDS §16 (and its three addenda) before running this — ratify is a MILESTONE act,
not a per-stage one. `tools/ratify.py`'s own module docstring is the authority on every flag below.

## When to run it
ONE ratify per cohesive product: the milestone that merges, and after the very LAST edit of any
kind. A stage's own gate is small (iteration build, oracle on the arms it touched, census, model
tests) and explicitly does NOT ratify — manifests go stale between milestones by design, and a
stage commit says "unratified" in its title. `ratify-after-the-last-edit`: a clean `git status`
never proves the manifest matches the source; only a NO-OP RE-RATIFY does. Order is always
**edit → build → ratify --write → commit** — never ratify before the final source edit.

## The commands
Run `python tools/ratify.py --help` (or read its argparse block) for the current flag set. The
shapes to know:
```bash
python tools/ratify.py                              # default arches/arms, check only
python tools/ratify.py --arch <a> --arch <b>         # explicit arch scope
python tools/ratify.py --arch <new-arch> --write     # bringing up a new arch (fresh measurement)
python tools/ratify.py --shard <name>                # MEASUREMENT scope only, triage one shard
python tools/ratify.py --self-test                   # the gate must FAIL on injected defects
```
`--arm` / a carry-arm scope measures a SUBSET (e.g. the stage's own touched arms) but `--write`
REFUSES a partial scope — a manifest filed from a subset would silently drop every arm it didn't
compile. `--shard`/arm scoping is for INVESTIGATING a ratchet refusal one slice at a time, never for
filing.

## Incremental cache
Since the carry family moved to one-TU-per-arm, `measure_units` caches per translation unit, keyed
on the exact argv plus the CONTENT hash of every file `nvcc` reported reading (never a timestamp).
An untouched arm's registers/spills/SASS body are REPLAYED, not recompiled — see FINDINGS `## K36`
for the wall-time drop this produced as an illustration of the effect, not a number to expect.
Ratify stays incremental per arm; don't force a full recompile to "be safe" — the cache key already
covers the whole include closure.

## Reading a "mover": module-context vs anon-namespace path trap
When SASS bodies differ across a rebuild, first ask whether the NAME still matches. Two different
failure shapes:
- **Anon-namespace / path-hash trap (fixed, still watch for regressions)**: an anonymous namespace's
  mangled name embeds a discriminator derived from the SOURCE DIRECTORY, so the SAME code built in a
  different worktree path names a DIFFERENT symbol — a worktree build then fails the fatbin gate
  ("N ratified instantiation(s) name no entry point") even though the code is identical. Fixed by
  moving every kernel into a NAMED internal namespace; if you see this shape again, check for a
  reintroduced anonymous namespace, not a build bug.
- **Module-context reschedule (expected and benign)**: names match exactly, registers/spills/
  instruction counts match exactly, but a FEW bodies differ byte-for-byte purely because of what
  ELSE compiles alongside them in the same translation unit (see FINDINGS `## K36` for the proof
  case). DIAGNOSE this by REGROUPING — rebuild the suspect arms together in one probe TU and see if
  the hash reproduces the "before" or "after" binary exactly; if it does, this is composition, not a
  defect. Byte-identity across a TU-partition change is then UNREACHABLE by construction and is not
  a valid gate for that change — accept the reschedule if every ratified NUMBER (regs/stack/spills)
  is unchanged, and re-ratify at the milestone.

## The device stamp — before ANY gate
Assert the device-side build stamp (`carry_ops.build_stamp(...)`, magic word check) before trusting
any gate result in the same process/session — a stale extension survives path and hash checks
(`extension-trap-device-side-check`, a recurring trap). This is cheaper than a ratify run and catches
a wrong binary before you spend the ratify's wall time on it.

## The subset-flags refusal
`ratify.py` refuses to certify a manifest whose flags don't match `tools/build_flags.py`'s current
`ratified_flags()` (FLAG DRIFT), whose `ptxas --version` doesn't match the pinned toolchain
(TOOLCHAIN DRIFT), or whose recorded `csrc` digest doesn't match the tree (SOURCE DRIFT) — all three
are hard FAILs demanding `--write`, never a warning. Don't hand-edit a manifest entry to make a
ratchet pass; a manually-asserted spill number is exactly the failure this file exists to replace.
Re-ratifying is a MEASUREMENT, never an edit.

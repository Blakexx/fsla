---
name: rola-ledger
description: The PC-range/SourceCounters attribution method for costing a kernel section by section — how K4-LEDGER, STAGE-LEDGER and R-A were built
---
This is the method behind FINDINGS `## K4-LEDGER`, `## STAGE-LEDGER`, and `## R-A` — use it to cost
a kernel by MECHANISM/SECTION, not just as a black-box wall-time number. Read those three sections
before running a new ledger; each records a caveat the next one had to correct.

## Why it works without a probe build
Check `tools/build_flags.py`'s `nvcc_flags` for whether `-lineinfo` is unconditional in the ratified
flag set — if so, every shipped binary already carries SASS→CUDA line correlation and no probe
binary is needed; profile the shipped `.so` directly.

## The commands
```bash
ncu --section SourceCounters --import-source no --page source --csv \
    -c 1 -k regex:carry.*kernel <venv>/bin/python <one-cell driver>.py > cell.csv
```
One launch (`-c 1`) per (arm, cell). Cross-validate against a clean counter pass on the SAME cell:
```bash
ncu --metrics smsp__inst_executed.sum -c 1 -k regex:carry.*kernel ...
```
The source-page total should equal the clean counter closely when the body has no spin loop. On a
body WITH spin loops (the carved/ring bodies) expect an inflation — SourceCounters' extra
serialization makes the spin loops spin more. Report this ratio explicitly per stage; it is itself
informative (see STAGE-LEDGER for an example of the ratio moving across stages and what that meant).
Don't silently divide it out — record it and note ratios/shares are robust even when the absolute
total isn't.

## Section anchoring: ADDRESS ORDER, and headers inherit
`nvdisasm -c -g` emits only the INNERMOST inline frame, so a header instruction carries the header's
line, never the call site's. Sections are therefore assigned by walking the disassembly in ADDRESS
ORDER against the body's own phase/lambda-defining line ranges (banner comments, lambda names) — a
header instruction inherits whichever section it falls inside. STAGE-LEDGER's refinement: anchor
sections in EACH TIP'S OWN SOURCE by regex — a stage that hasn't yet grown a mechanism simply has no
anchor, and that absence IS the delta (don't force a zero-width section into existence to make
tables line up across tips).

## First-principles floors
Every section needs a DERIVED minimum before you can rate it, not just a measured number. The
floor is always a formula built from the section's own structural parameters (fragment counts,
issue/RMW/instruction ratios, load-and-compare cost) — never a fitted constant. See K4-LEDGER/
STAGE-LEDGER for worked examples of deriving a floor for a specific mechanism. State the floor's
formula in the ledger table beside the measured number — a ratio with no stated floor is not a
floor.

## The (a)/(b)/(c) classification
Every ledger entry gets exactly one of:
- **(a)** inherent to the design, near its own minimum — a genuine cost the mechanism needs.
- **(b)** the mechanism is right but the IMPLEMENTATION sits above its floor (a fixable
  inefficiency inside the correct design — "fix-to-floor", never "revert" or "redesign").
- **(c)** work the CELL does not need at all (e.g. a fan-in mechanism running at a declaration that
  needs none of it; per-warp duplication of work one warp could do once).
Never leave a section unclassified, and never classify from instruction share alone — see R-A for
an example of a section that looked improved by instruction count alone but was (b) once stalls
were checked.

## Report time + IPC + stalls, always together
Per KERNEL_STANDARDS §16 addenda 1-2: an instruction ledger is NECESSARY, NOT SUFFICIENT. Every
ledger table carries wall/cycles, IPC, and the top `smsp__warp_issue_stalled_*` reasons beside
instruction counts. A stage whose instructions FALL while its TIME RISES is RED — report it as a
regression, never rationalize it as "fewer instructions, so it must be fine" (§16 addendum 2).

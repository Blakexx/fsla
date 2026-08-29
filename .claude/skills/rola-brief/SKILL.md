---
name: rola-brief
description: Template and rules for writing a RoLA kernel stage brief — worktree ownership, one mechanism with its floor, whole-protocol pseudocode, small gates, cited baselines
---
> YOU ARE THE WORKER when a brief is handed to you: the coordinator wrote it; you execute it yourself,
> never delegating to another agent and never writing a brief for one. (Coordinator-only skills —
> rola-dispatch/review/queue/card/recall — do not apply to you.)

Exemplar: the M1P port brief (BRIEF_M1P_port.md in the campaign brief directory the dispatch prompt names). A brief is short
and every clause below maps to a line in it — read it to see the shape, don't quote its specific
numbers or names as if they generalize to every brief.

## The template, in order
1. **Title line**: `BRIEF <ID> — <one-line what>; GATE = <the acceptance rule>` with who ruled it
   and when.
2. **Ownership line**: exact worktree path, branch, tip SHA (with "verify clean; sole ownership"),
   the venv python path, and any READ-ONLY reference worktree(s) named explicitly as read-only.
   Never leave ownership implicit — two agents in one worktree is how work gets clobbered.
3. **Design authority**: which FINDINGS sections and which doc (KERNEL_STANDARDS §-range,
   `docs/build.md`, a queue card) the implementer must read before touching code, and where to
   APPEND the report (a named FINDINGS `## <ID>` heading, never a new file).
4. **The ruling**: one paragraph, what stays and what changes, stated as a POSITIVE list ("keep ALL
   of X — name each piece — and REPLACE Y with Z") so nothing is ambiguous by omission.
5. **ONE mechanism per stage**, its first-principles floor stated BEFORE building (KERNEL_STANDARDS
   §12): the derived minimum (a formula, not a vibe) that the actual implementation will be
   measured against in the report. A brief that ships without a floor cannot produce a graded report
   — the `rola-ledger` (a)/(b)/(c) classification has nothing to compare to.
6. **Whole-protocol pseudocode** when a synchronization/data-path mechanism changes (KERNEL_STANDARDS
   §13): restate the ENTIRE function, not a diff of it — the implementer deletes the old body and
   rebuilds from the pseudocode, never edits around the previous protocol's skeleton. Name what the
   rebuild must delete too (dead sentinels, deferral rules, counters that only made sense for the
   superseded protocol) — "no dead code" applies at mechanism grain.
7. **Numbered "what to build"** steps, each naming the file(s) it touches and what is explicitly OUT
   (per the exemplar's own step on removed vs. kept rows: a row kept "as a commented row" is NOT
   allowed — delete). If something is deferred to a later stage, say which stage, don't half-build
   it now.
8. **Gates**, named exactly (§16 SMALL): the oracle test file(s) + GPU-flock command
   (`rola-oracle-tests`), the perf probe cells and its pass bar (`rola-probe`), what SASS comparison
   IS and ISN'T expected (state it explicitly per-brief — never assume identical-to-parent unless
   the mechanism is unchanged), the exact commit message, and the compile ledger (`rola-build`).
9. **Cited baselines, never re-measured cold**: point at the FINDINGS row/table a number comes from
   (`rola-ledger`/`rola-probe` skills cover HOW those were taken) — a brief re-deriving a baseline
   from scratch wastes the stage's build budget on a number that already exists.
10. **Standing rules folded in by reference**: the build-cycle budget (§15), background builds + exit
    marker (§14, `rola-build`), "no polling" — cite the sections, don't restate their text.
11. **Report format**: a hard line cap, and the exact fields expected (SHA; what was ported/adapted;
    the arm list after; the parity table; oracle result; census; walls; what could not be served and
    why).
12. **Stop-and-report conditions**, explicit: e.g. "If parity fails: STOP and report the per-section
    ledger of the delta — the port changed something; do not tune." A brief without a named stop
    condition invites an agent to keep iterating past its budget chasing green.

## Standing rules every brief inherits (don't restate, do cite)
- Build-cycle budget (§15): a stage gets a fixed number of full build+gate cycles; a still-red gate
  at budget-out STOPS and reports the residual. Read §15 for the current count rather than quoting
  one here.
- Mid-stage re-scoping is allowed only for a DESIGN change (Blake's ruling), never for a report-shape
  tweak — those go in the next brief.
- Never merge on an agent's self-report — the coordinator reads the code before calling a stage done
  (`coordinator-reads-the-code`).
- Model tier: name it (open-design kernel batch vs. frozen-interface execution) —
  `agent-model-tiering`.

## Skills feedback (Blake, 2026-08-28 — the skills are approved on the condition they are actively maintained)
Every brief's Report section ends with a `SKILLS FEEDBACK` line: which skills the agent loaded, what
was wrong, stale, or missing in each, and what it had to discover the hard way. The coordinator folds
that into the skill the same day (process/pointers only — never facts).

- COMMENTS: the stage adds NO `//:` prose blocks to source; decl blocks only, derivations go to the mirrored docs/internals page; the report states the per-file comment ratio before/after (must not rise) — constitution: self-documenting code.

- NO NESTED AGENTS: the stage agent does the work itself; it never spawns a sub-agent to do the stage and never monitors/polls one (subagent-token-discipline). Waiting on a build = end the turn and resume on its notification.


- FINDINGS.md is the campaign journal (the out-of-repo file the dispatch prompt names) — NEVER a file inside the repo/worktree (§17). A tool that REFUSES an action is a stop-and-report signal; never route around a refusal with another tool.

- `--no-verify` NEVER without an explicit per-commit authorization from the coordinator naming the hook failure; a red hook = STOP and report which (§18 addendum).

- THE FLOOR HAS A COVERAGE TERM (F1, 2026-08-29): a mechanism's floor states bytes, instructions, conflicts AND what covers its latency (which in-flight work overlaps each serial step; a CTA-wide round trip with nothing under it is a cost even at zero bytes). A floor without a coverage line cannot grade a stage.

# tools/lint — every lint, its rule, its gating status, the flip plan

Mirrors `tools/lint/` file-for-file. `docs/build.md`'s "Standards lint" section
covers the FIRST generation of these checks (ast-grep rules, `lint_standards.py`,
`run_clang_tidy.sh`) — this page adds the LINT2 batch (G_FOUNDATION G5's "dead-code
and readability lints", ratified 2026-08-29, Blake) and states, per lint, whether it
gates a commit today or only reports.

| lint | rule | gating today? | flip plan |
|---|---|---|---|
| `ast_grep_gate.py` + `tools/lint/rules/*.yml` | KERNEL_STANDARDS shapes an AST can see (§12 fences/collectives, K36 torch-in-arm-TU, anon-namespace kernels, width-shaped `if`) | **yes** | — |
| `lint_standards.py` | unroll discipline (§6), arm-TU torch include (belt), decode-prologue barrier (P79), `docs/`/`tools/` flock spelling (K46), `check_supported_table`/`check_shard_partition`/`check_arch_coverage`/`check_vendored_pin`/`check_entmax_pin`/`check_repo_portability` | **yes** | — |
| `run_clang_tidy.sh` | `bugprone-*` + small `readability-*` on HOST TUs (`.clang-tidy`) | **yes** | — |
| `run_clang_tidy_device.sh` + `clang-tidy-device.yaml` | `clang-analyzer-deadcode.DeadStores`, `misc-unused-parameters`, `bugprone-unused-*`, `readability-redundant-*` on the torch-free carry ARM TUs (clang `--cuda-host-only`) | report-only | G5, after a sweep of whatever this profile finds across the full arm set (widen with `ROLA_CARRY_ARMS=all` first) |
| `host_warnings.py` | `-Wall -Wextra -Wunused` dry-compile (`-fsyntax-only`) of the host TU(s) in `compile_commands.json`, on-target findings only (`csrc/rola/` lines; vendored-header warnings reported separately, never counted) | report-only | G5, once 0 on-target findings is the steady state (currently is: 0 measured 2026-08-29) |
| `unused_instantiations.py` | every `__global__` function TEMPLATE in `csrc/rola/` must demangle-match a symbol in a committed manifest (`tools/manifests/*.json`); `__device__` templates are found and counted but explicitly NOT CHECKED (manifests only ever record `ptxas -v` ENTRY functions — see the module docstring for the measured false-positive rate that ruled `__device__` out) | report-only | G5 deletes any `__global__` finding that still holds after a fresh ratify; `__device__` liveness needs a different instrument (a `cuobjdump --dump-elf` STT_FUNC symbol dump) not built here |
| `run_vulture.sh` + `vulture_whitelist.py` | vulture over `rola/ tools/ tests/ benchmarks/`, whitelist = genuine dynamic uses only (pytest's `pytestmark` convention, pytest fixture injection), each with a one-line reason | report-only | G5, after ruling on the remaining findings (46 measured 2026-08-29) one at a time — a whitelist entry is earned by a genuine dynamic use, never added to silence a real dead-code hit |
| `run_ruff_e3.sh` | ruff's E30x (pycodestyle blank-line spacing, E301-E306) — preview-gated in ruff 0.16.5, so NOT in `pyproject.toml`'s gating `[tool.ruff.lint].select`; the codes list lives in `[tool.rola_lint.report_only]`, read by this script only | report-only | G5: fix the findings (41 measured 2026-08-29, `--fix` covers all of them), then move the codes into the real `select` and delete the report-only table |
| `readability_spacing.py` | (a) any function/kernel body >= 40 lines with ZERO blank lines, csrc AND Python; (b) > 1 consecutive blank line, **csrc only** (Python's own convention is 2 between top-level defs — PEP8/ruff's `E303` territory, not this rule's) | report-only | G5: segment the long bodies (10 measured 2026-08-29) and adopt `.clang-format`'s `MaxEmptyLinesToKeep: 1` / `SeparateDefinitionBlocks: Always` (below) as the mechanical enforcement for (b) |
| `r9_enforcement.py` | KERNEL_STANDARDS "R9 ENFORCEMENT" constitution addendum: a device `switch`/if-chain on a structural field must have a `__trap()` default (never a bare fallthrough) AND a `// R9-CASES: <name>` marker comment naming its generated case set — the marker THIS LINT DEFINES, since no generator emits it yet (G2/G5 adopt it) | report-only | G2/G5: once the generator emits the marker and defaults trap, this lint's finding count is the exhaustiveness backstop (§9: "should catch nothing") |
| `.clang-tidy` | HOST-only `bugprone-*` + small `readability-*` | **yes** (via `run_clang_tidy.sh`) | — |
| `clang-tidy-device.yaml` | the LINT2 device check set (above) | report-only (via `run_clang_tidy_device.sh`) | see that row |
| `clang-format-readability.yaml` + `run_clang_format_readability.sh` | `MaxEmptyLinesToKeep: 1` + `SeparateDefinitionBlocks: Always`, over the root `.clang-format`'s other settings | report-only | G5: fix the 153 diffs measured 2026-08-29 in one dedicated reformat commit, merge these two lines into the root `.clang-format`, delete this file |

## The readability format rules are a SEPARATE file, not a root-`.clang-format` edit — the reason is measured, not stylistic

Tried first: adding `MaxEmptyLinesToKeep: 1` / `SeparateDefinitionBlocks: Always`
directly to the root `.clang-format`, which the currently-**GATING**
`clang-format-check` pre-commit hook reads (`--dry-run --Werror`, scoped to
`^csrc/rola/.*\.(cu|cuh|cpp|h|hpp)$`, run via `pre-commit run --all-files` against
every matching file in the tree, not just a commit's diff). Measured 2026-08-29:
this immediately produced 153 diffs on pre-existing files (`carry_kernel.cuh`,
`entmax.cuh`, `decode.cuh`, `facts_kernel.cuh`, ...) that predate
`SeparateDefinitionBlocks: Always`, which took `pre-commit run --all-files` from
green to red on an EXISTING gating hook — directly contradicting this stage's own
"report-only, no hook fails on them yet" and "hooks green" gates, and this stage's
own "CONFIG ONLY: do NOT reformat any source" instruction (reformatting the tree to
fix it was the only way to keep the root file's edit green, and that is explicitly
out of scope here). `tools/lint/clang-format-readability.yaml` restates the root
config's other settings (clang-format style files do not chain/include one
another) plus the two new ones, invoked only by
`run_clang_format_readability.sh`'s own report-only pre-commit stage — the root
`.clang-format` and the `clang-format-check` hook are UNTOUCHED by this stage.
G5's flip is: fix the 153 diffs (one dedicated, byte-identity/SASS-checked reformat
commit, the comments-to-docs mandate's own precedent for this shape of change),
merge the two lines into the root file, and delete this one.

## Fixtures (`tools/lint/fixtures/`, K46 convention)

Every LINT2 lint has a fixture pair under `tools/lint/fixtures/` (a "must fire"
case and a "must not fire" case, most often in ONE file) and a `--test-fixtures`
self-test mode on its own driver:

```bash
python3 tools/lint/host_warnings.py --test-fixtures
python3 tools/lint/unused_instantiations.py --test-fixtures
python3 tools/lint/readability_spacing.py --test-fixtures
python3 tools/lint/r9_enforcement.py --test-fixtures
tools/lint/run_clang_tidy_device.sh --test-fixtures
tools/lint/run_vulture.sh --test-fixtures      # needs vulture on PATH
```

Fixtures are EXCLUDED from every lint's normal-mode scan (`readability_spacing.py`'s
`py_files()`, `run_vulture.sh`'s `--exclude '*/tools/lint/fixtures/*'`) — the K46
lesson (`ast_grep_gate.py`'s own header) is that a fixture's deliberate violation
re-triggering the real report is exactly how a hook drifts silently disabled.

## Running everything report-only, by hand

```bash
tools/lint/run_clang_tidy_device.sh
python3 tools/lint/host_warnings.py
python3 tools/lint/unused_instantiations.py
tools/lint/run_vulture.sh
tools/lint/run_ruff_e3.sh
python3 tools/lint/readability_spacing.py
python3 tools/lint/r9_enforcement.py
tools/lint/run_clang_format_readability.sh
```

Every one of these exits 0 unconditionally (report-only, per G_FOUNDATION G5's
"GATE = ... wired REPORT-ONLY (no hook fails on them yet)") and is wired into
`.pre-commit-config.yaml` as a non-blocking stage; a hit is a finding for a human
to rule on, the same backstop philosophy KERNEL_STANDARDS §9 states for `ratify`.

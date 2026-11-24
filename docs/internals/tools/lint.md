# `tools/lint` — every lint, its rule, and whether it gates

Mirrors `tools/lint/` file for file: the whole roster, and, per lint, how it gates.

**A GATE IS A CHECK SOMEBODY CAN SATISFY.** The gating set below is at zero on the tree. A lint the tree does not pass
yet is not thereby wrong -- its findings are the work -- but refusing every commit until that work is done teaches
contributors to bypass the gate. So such a lint is a RATCHET: it refuses only what a commit adds, and the findings the
tree carried when it began gating are a baseline that only shrinks.

## The gating set

| lint | rule |
|---|---|
| `ast_grep_gate.py` + `rules/*.yml` | the standards' shapes an AST can see (§12 fences and collectives, a torch header in a per-arm translation unit, an anonymous-namespace kernel, a width-shaped `if`) |
| `lint_standards.py` | unroll discipline (§6), the arm-TU torch include, the barrier-free regions (no CTA barrier between a file's `//: @barrier-free` and `//: @barrier-free-end`), the whole `docs/internals/` mirror contract (including that a mirror page names no identifier the tree does not carry), the flock spelling, the supported matrix, the vendored pin, the comment-density mandate, repository portability |
| `unit_lints.py` + `.clang-tidy`, `clang-tidy-device.yaml` | the pinned clang-tidy (`tools/clang_tidy_pin.json`) on every unit the library links, for each built architecture: `.clang-tidy` on the host units (every finding fails), `clang-tidy-device.yaml` on the device units through clang's CUDA front end (dead stores, unused parameters, redundant expressions); and the host units' `-Wall -Wextra -Wunused` front-end compile. A finding in the unit's own package fails, and so does any compiler error -- a unit the compiler cannot parse is a unit no lint read ([below](#unit-lints)) |
| `unused_instantiations.py` | every `__global__` function template must demangle-match an entry function the graph's units compiled (the census's, `bazel test //csrc/fsla:unused_instantiations`). `__device__` templates are found, counted and explicitly NOT checked: `ptxas -v` reports entry functions only, and a wrapped-once helper has no standalone symbol |
| `run_ruff_e3.sh` | pycodestyle's blank-line family. It keeps its own invocation because ruff preview-gates that family, and turning preview on in `pyproject.toml` would gate every other family's preview rules too |
| `readability_spacing.py` | a function or kernel body of forty lines or more with ZERO blank lines in it, and a run of blank lines in `csrc` (Python's own two-line convention is ruff's `E303` territory). The rule asks for segmentation and does not say where |
| `python -m fsla_devtools.mirror --check` | every tracked path is declared ships or private in `.github/mirror/declarations.json`, so a new file is never left undecided (the tool is fsla-devtools', linked into the venvs; its README describes the declarations and the tripwire) |
| `wgmma_width.py` | no library kernel issues a tensor step wider than 64 columns (CONSTITUTION R25): a `wgmma.mma_async` mnemonic of `csrc/fsla` with a shape `m<M>n<N>k<K>` of N over 64 fails, comments excepted; `measure/lab/` is exempt (`wgmma_width` measures wide steps on purpose); its fixtures, `wgmma_width_wide.txt` (three wide steps, and one in a comment that must not fire) and `wgmma_width_narrow.txt`, are run by `//tools/lint:wgmma_width_fixtures`, the lint on the tree by `//:wgmma_width` |
| `rules_table.py` | every rule of the constitution and the standards has a row in [`docs/rules.json`](../../rules.json) naming its mechanism, and every enforcer a row names resolves ([below](#rules-table)) |
| `learnings.py` | every row of [`LEARNINGS.md`](../LEARNINGS.md) names, as `path::text`, a mechanism the tree holds: a row that names none, a file that is gone or a string its file no longer contains fails the commit |
| `clang-format` (root `.clang-format`) | the formatting floor, including `MaxEmptyLinesToKeep: 1` and `SeparateDefinitionBlocks: Always` — the formatter's half of the readability rule above |

<a id="primitives-census"></a>**`primitives_census.py`** is a graph test over one probe translation unit
(`tools/lint/primitives_census_probe.cu`) that calls each `common/ops.cuh` primitive from its own `__global__` kernel. The graph's
CUDA rule compiles it for the build's architectures (`BUILT_ARCHS`), and the test reads the object's SASS with the graph's
`cuobjdump -sass` and asserts an exact count of each opcode family a primitive must lower to, so a primitive that falls back to an
emulation fails the day it is written. No GPU is needed.

<a id="unit-lints"></a>
## The unit lints in the build graph

The clang-tidy profiles and the host warnings read each unit's own compile: `bazel/defs.bzl`'s `unit_lints` aspect hands
`tools/lint/unit_lints.py` the compile action's command line, and the lint's inputs are the compile's, so it reads
what the build compiled and reruns only when that compile's inputs change. The units are the library's own, and each
built architecture is its own configuration, whose generated headers are that architecture's build set. A verdict
counts a warning in the unit's own package and any compiler error, and a run that fails with no located diagnostic
is red too; a fixture's verdict must hold exactly its expected findings ([below](#fixtures)).

<a id="portable-agent-files"></a>
## The agent files are portable

`tools/lint/portable_claude.py` (gating, KERNEL_STANDARDS §17) fails a commit whose agent files -- everything under
`.claude/` and every `CLAUDE.md` -- name a machine's own path: a home directory, a mount, a drive. The hooks and
`CLAUDE.md` ship with the code to every clone and worktree, so they cite repo-relative paths, hooks resolve
through `$CLAUDE_PROJECT_DIR`, and a site's own locations come from the dev config or the environment
(`FSLA_SCRATCH`).

<a id="rules-table"></a>
## Every rule has a row

EVERY RULE HAS A ROW (Blake, 2026-11-17; the standards' preamble). `docs/rules.json` is a JSON object of two sections. `rules` is a list with a row for each
rule of [`CONSTITUTION.md`](../../CONSTITUTION.md) and [`KERNEL_STANDARDS.md`](../../KERNEL_STANDARDS.md), in the documents'
order: its `site`, an `id`, a `statement` (25 words at most) and a `mechanism` -- `gate` (a check of `bazel test //:gate`),
`tool` (a tool refuses when run), `hook` (a git hook tracked in `tools/git-hooks/`, or an action hook: a file under `.claude/hooks/` listed in
[`.claude/action-hooks.json`](../../../.claude/action-hooks.json), [below](#action-hooks)), `review` (the LLM review, whose `block`
refuses the next commit, [`review_gate.md`](review_gate.md#blocking)), `owed` (no mechanism yet) or `none` (a definition, a
history note, a section that states no rule). Each carries what that mechanism needs: `enforcer` (a list of paths,
`path::symbol` and Bazel labels) and `coverage` (`full`, or `partial: <what is not checked>`); `why_not_mechanical` for a
`review` row; `would_be` for an `owed` row (the forcing point it will take and how); `why` for a `none` row. A `review` row's
statement reaches the reviewer's prompt, which lists them as the rules only the review enforces. `checker` is not a standard and
holds no row: it is fsla's guidance on the rules of the pinned checker (check 5 [below](#rules-table)).

A SITE is derived from the documents by the lint, never listed twice. `<DOC>#<anchor>` names every `<a id>` anchor (every
`## ` heading of either document, each addendum paragraph and each paragraph of the standards' preamble carries one, and
the lint fails on one that does not); `<DOC>#<anchor>.<n>` a numbered item under it, `.<n>.<letter>` a lettered item under
that, and `CONSTITUTION#principles.<slug>` each named principle. A site with items is their container and needs no row of its
own; a section's unnumbered bullets share its site, so a row's `id` is its site, or the site, a colon and a slug where one
site states more than one rule.

`rules_table.py` (gating, `//:rules_table`) fails on

1. a row that breaks the schema, a duplicate `id`, a mechanism's required field missing;
2. a site of either document with no row, and a row whose site is in neither;
3. an enforcer that does not resolve: a path that does not exist, a `path::symbol` whose symbol the file does not contain, a
   Bazel label its package's `BUILD.bazel` does not define, and for a `gate` row a label that is not a member of `//:gate`.
   The build files are read as text, because the lint runs inside the gate and cannot run Bazel, so a target a macro names
   by a computed string is not seen. A `hook` row's enforcer holds a git hook of `tools/git-hooks/` or an action hook that is
   a file AND is listed in the manifest (a hook file the manifest does not list fails, so a hook that never fires is not
   a mechanism);
4. a new `owed` row, through the ratchet: `rules_table.py --owed` prints each `owed` row's id as a finding and
   `//:ratchet_rules_owed` holds them against `tools/lint/baselines/rules_owed.json`. A row that stops being `owed` leaves
   the baseline in the same commit (`ratchet.py rules_owed --shrink`), so the list only shrinks;
5. the `checker` section, against the rules of the pinned checker (`warpline.checker.declare.rule_names`, over the checker's base
   ledger with the project's `tools/notice_ledger.json`): a rule with no entry, an entry whose key is a rule the pin does not have,
   and an id an entry lists under `rules` that is no row. The section is the checker's own map to fsla's mechanism, not a standard:
   an entry is keyed by the checker rule's name and carries `guidance`, one line -- the atom or
   practice, the standards anchor, the lab page with the evidence, or "nothing beyond the generic fix" where fsla has nothing to
   add -- `triggers`, regexes over kernel source (a list, empty where no source text shows the rule), and `rules`, the ids of the
   standards rows it serves (a list, empty where no row states the rule); it holds no mechanism, enforcer or coverage, so it claims
   no enforcement, and a row's `enforcer` names `tools/lab.py::check` only where that check holds the rule. The lint needs the
   checker on its path (`//:rules_table` carries `@warpline` as data) and fails without it. `tools/lab.py` hands each entry's guidance to the checker
   as its notes, so a refusal prints the finding, the checker's fix, then `note: <guidance>`
   ([`lab.md`](lab.md#hazards)); the edit hook [below](#action-hooks) injects it.

A resolving enforcer is not proof that it checks the rule: the table names the check a person opened and saw check the rule,
and `coverage` says what it leaves unchecked. A NEW RULE lands with its mechanism and never as `owed`. The fixtures are
`tools/lint/fixtures/rules_table/`: a small tree (two documents, a table, build files) that passes, and twenty edits of its
table, each failing one check with its own message, the last seven for the `checker` section (a rule with no entry, an entry naming a rule
the pin lacks, an entry listing an id that is no row, an entry with no guidance, an entry with a key it does not carry, a trigger that is no regex,
a table with no section); its tree holds a manifest with one listed and one unlisted action hook. `tests/unit/test_lab_guidance.py` covers the lab's
note and the hook.

<a id="action-hooks"></a>
## The agent files: action hooks and their dispatcher

`.claude/` holds three kinds of tracked file: `settings.json` (the session-start hook), `references.lock.json`, and the
action hooks, which let a rule the rules table marks `hook` refuse a tool call as it is made rather than be found in review.
`.claude/action-hooks.json` lists the hooks a call fires (`{"PreToolUse": [{"matcher": "<regex of tool names>", "command":
"<path in the checkout>"}]}`) and `.claude/hooks/` holds them; a hook reads the call's JSON on stdin and refuses by printing
a `permissionDecision` of `deny` with its reason. One generic dispatcher, `.claude/hooks/repo_hooks.py`, fires them from any
session, runs only checkouts under the dev config's `workspace.hook_roots` (a cloned stranger's manifest never runs) and
returns the first refusal; the install is [`dev.md`](dev.md#action-hooks), and a hook holds only where it is installed.
Four exist, each with its patterns and its override in its own docstring: `no_polling.py` (KERNEL_STANDARDS §14),
`no_verify.py` (`#no-verify`), `graph_only.py` (`KERNEL_STANDARDS#s14-graph`: the toolchain is the graph's, so a toolchain or install command typed by hand is refused, and the refusal names the graph target that does the step; `tests/unit/test_graph_only.py`) and `guidance.py`, which is advisory and never refuses: on an Edit, MultiEdit or Write of a kernel
source (`csrc/**/*.cu*`, `measure/lab/**/*.cu*`) it injects the `guidance` of each entry of the `checker` section of `docs/rules.json` whose
`triggers` match the edit's new text (a branch around a tensor step, a new shared store, a collective), and says nothing of an edit
that matches none. `portable_claude.py` gates them like every file under `.claude/` ([above](#portable-agent-files)).

<a id="graph-only"></a>
## The graph runs the toolchain: `graph_only.py`

`tools/lint/graph_only.py` (`bazel test //:graph_only`, in `//:lints`; self-test `//tools/lint:graph_only_fixtures`) is the commit
gate of KERNEL_STANDARDS#s14-graph. It reads every tracked `*.py`, `*.sh`, `*.bzl`, `BUILD*` and shebang script (the hooks) as syntax and fails
on a step that runs `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm`, `cu++filt`, `ncu`, `nsys`, `compute-sanitizer`, `cmake`, `make`, `ninja`, a pip or uv
install or `pytest`, or makes a temporary directory a build would stage in (`tempfile.mkdtemp`/`TemporaryDirectory` named for a build, or in a file
that runs a tool; `mktemp -d`): a Python call of `subprocess`/`os`/`asyncio` or a tool's own runner (`run`, `_run`, `sh`) whose command is
a list or shell string headed by the tool, a shell command word, a `genrule`'s `cmd`. A tool named in a comment, a docstring or a string no call runs
is text and passes. There is no baseline: `ALLOWLIST` holds one entry a call site (file, function, tool), each citing the graph rule whose action runs
it or the clause of KERNEL_STANDARDS#s14-graph that keeps it outside (the profiler on the host that grants its counters, measurement's own workers and locks, the sanitizer lane's pair, what runs on a rented box), with its words quoted; an entry that matches no site fails, so the list is the sites. A new step belongs in a
graph rule; `GRAPH_OVERRIDE` is the hook's exception for a typed command, never the lint's.

<a id="ratchets"></a>
## The ratchets

`tools/lint/ratchet.py <lint>` runs a lint and compares its findings to `tools/lint/baselines/<lint>.json`. A finding is
known by its path, the first sentence of its message and a digest of the line it names, so an edit above it does not
move it. The gate fails on a finding the baseline does not hold, on a baseline entry the lint no longer finds (run
`--shrink` and commit the smaller baseline with the fix), and on a baseline holding more of an entry than HEAD's.
`--grown` makes that last comparison, every baseline at once, in the checkout; each lint's own ratchet runs over a copy
of the tree, which has no HEAD ([`build-graph.md`](../../build-graph.md#checks)). A HEAD git cannot read fails it. Editing
a line that carries a finding changes its digest, so the commit that touches the line fixes the finding. A lint that
cannot run fails; it never reads as zero. `--init` records a new lint's baseline once.

A RE-KEY is the stated rename of a finding's key, for a finding whose message was reworded (the message is part of the key): `--rekey OLD NEW`
rewrites the baseline's entries whose message is OLD to NEW, their counts unchanged, and records `{"from", "to", "count"}` in the baseline's `rekeys`.

`--grown` reads HEAD's entries through the baseline's own re-keys, so a re-keyed entry is the same entry and a re-key that adds to a count is growth
(`tests/unit/test_lint_ratchet.py`); `--shrink` drops a re-key whose `to` the baseline no longer holds.

A FOLLOW is the stated move of a file and reword of a line (the line digest is part of the key): `--follow` takes each entry's line in HEAD's file, follows it through git's rename detection and a line diff to the working file, and rewrites the entry to the file and digest it now has, counts unchanged; the digests that changed are recorded in `follows`, which `--grown` reads HEAD's entries through and re-derives from git, so a hand-written follow is refused.

A finding that is not a defect is exempted the lint's own way, with its reason: vulture's whitelist
(`vulture_whitelist.py`), a `reserved:` marker for a constant parameter. "A ruling per hit" is that work: the lint cannot
tell a real finding from a false one, so a human decides each, and the decision lands as a fix or a stated exemption.

| lint | baseline (2026-09-14) | what burns it down |
|---|---|---|
| `work_codes.py` | 558 | prose: `DELETIONS.md` 162 and `open-work.md` 59 need rows that say what each change did; the rest are test and tool docstrings |
| `constant_parameters.py` | 134; 63 (2026-09-29) | a ruling per finding: inline the constant, or mark the parameter `reserved:` with its reason. The tests count as callers of the public surface (the names `fsla` exports), and a constructor's parameters are keyed by its class -- keyed `__init__`, they matched no call and every class shared one key, which filed 17 exercised options as dead. Callers outside the tree and positional calls are invisible, so it over-reports |
| `run_vulture.sh` | 60 | a ruling per finding: delete the dead code, or whitelist a demonstrated dynamic use (name-dispatched methods, fixtures, a library's reads) |
| `drift_guards.py` | 20 over 11 rules | the table below |
| `facts_in_template.py` | 79 over 3 rules (2026-11-24) | the rules' findings in kernels and planner headers: [the facts-in-the-template rules](#facts-in-the-template) |
| `r9_enforcement.py` | 1 | `decode_lattice.cuh`'s `switch (kind)` folds its fourth case into a bare default; the fix is a `uniform_switch`, a decode kernel change |
| `rules_table.py --owed` | 63 (2026-11-17) | the `owed` rows of `docs/rules.json`: each leaves when its rule gains a mechanism, a `gate`, `tool`, `hook` or `review` row, in the commit that lands it ([above](#rules-table)) |
| `coverage.py` | 0 (2026-09-29) | nothing: it gates the undeclared gaps (a kernel not in `tools/coverage.json`, a pointer to nothing); the owed obligations -- budgets, phase ledgers, sanitizer rows and oracle tests for the entmax solves, three kernels -- are the backlog it prints ([`coverage.md`](coverage.md)) |
| `burst_tier.py` | 0 (2026-11-09) | KERNEL_STANDARDS §24: an MMA issued outside a `//: @burst` function or a vote, sync, barrier, `while`, data `if` or `unroll 1` inside one, over every device function (a `static` or `__host__` one among them); `//: @burst-exempt <reason>` names a debt (the dense fold's and the dense readout's loops over a chunk's slots, `steps_dense`, `read_dense`, `dense_den`). It reads only a function that issues an MMA itself: a loop above a burst function that hand-rolls what `Burst` units do is a debt it cannot see, named on that kernel's page (owed with the kernel) |

<a id="drift-guards"></a>
## The drift guards, and the inventory they exist to size

Each rule is a law that a stage ratified by DELETING something, and each is a law
because the deleted thing is cheap to put back: a template axis that "only needs one more
parameter", a second kernel body for the case the first does not fit, a `float*` view of
a page, an `if` on the depth in a host entry, a comment explaining what the code used to
be. A law that lives only in a document is re-broken by the next person who has not read
it.

Every rule is a heuristic over text and says so in its own docstring: a finding is a
question for a reviewer, never a proof. Four rules were tightened after their first
inventory was read, each with its reason in the source — an inventory nobody trusts is an
inventory nobody flips.

**Measured 2026-09-14, 20 findings over 11 rules** (`bazel run //tools/lint:drift_guards` prints each):

| rule | findings |
|---|---|
| `arm_axes` | 1 |
| `bytes_equal` | 5: decode's batteries assert state identity with `torch.equal` |
| `comments_describe_the_present` | 3 |
| `device_switch_is_uniform` | 1: `decode_lattice.cuh`'s `switch (kind)`, the site `r9_enforcement.py` finds |
| `host_dispatch_is_arm_switch` | 0 |
| `one_admission_law` | 0 |
| `one_body_per_family` | 1 |
| `one_derivation_site` | 1 |
| `second_arm_table` | 5: test-local arm tables |
| `single_value_axis` | 0 |
| `state_float_view` | 3 |

`device_switch_is_uniform` and `r9_enforcement.py` overlap on purpose and are not the
same rule: `r9_enforcement.py` asks whether a structure branch has a trapping default and
a case-set marker, and this one asks whether it is a `uniform_switch` at all — which is
the form the constitution now requires. They find the same site from two directions,
which is the corroboration that both are connected; reconciling them into one rule waits
on the fix, so that the flip can decide which message a finding carries.

<a id="facts-in-the-template"></a>
### The facts-in-the-template rules

Three rules of their own lint, `facts_in_template.py` (ratchet target `//:ratchet_facts_in_template`, fixtures target
`//tools/lint:facts_in_template_fixtures`, baseline `tools/lint/baselines/facts_in_template.json`), enforce [`CONSTITUTION.md#r8`](../../CONSTITUTION.md#r8)'s "FACTS IN THE TEMPLATE, DECISIONS IN THE
KERNEL": the card's facts reach a kernel as constants of its build, every quantity the card decides is derived from them
at compile time, and a kernel's shared memory is declared only through `SmemLayout`, whose sizes are expressions and
never numbers ([`../common/smem_layout.md`](../common/smem_layout.md)). Their scope is `csrc/fsla/src/**/*.cu` and `*.cuh`
and the planner's headers `csrc/fsla/src/common/plan/*.h` (the planner is where the window and a CTA's leaves are
written, and a rule's own definition is no unexplained number either); not `measure/lab/` (an experiment is where a form is
tried before it is a kernel), not the host tools `plan_tool.cpp` and `arms_gen.cpp`. Comments are blanked before any
match, `//` and `/* */`. Each has its fixture pair `drift_facts_bad.cuh` / `drift_facts_ok.cuh` and its tests in
`tests/unit/test_drift_facts_rules.py`; each has a `gate` row of `docs/rules.json` at site `CONSTITUTION#r8`.

- `shared_memory_through_the_layout`: the token `__shared__` on a line, in any file of the scope but
  `common/smem_layout.cuh`, is a finding, one a line. It misses a declaration spelled by a macro or an alias.
- `layout_sizes_are_expressions`: in `Fixed<Tag, T, COUNT>` and `Share<Tag, T, LEAST, WEIGHT>` (the name followed by `<`,
  the arguments split at the top level of `<>`, `()`, `[]`, `{}`, across lines), a `COUNT` or `LEAST` that is a bare integer
  literal (decimal or hex, `u`/`l` suffixes) is a finding; a `WEIGHT` is a ratio and may be one; `smem_layout.cuh` is exempt
  (its `smem_proof` namespace states literals as its fixture). It misses a number behind a name, a cast or parentheses,
  and a tile through an alias.
- `card_quantities_are_derived`: a one-line `[inline] [static] constexpr <integer type> <name> = <bare integer literal>;`
  at namespace or class scope is a finding unless the comment lines directly above it contain `FIXED` in capitals and a
  reason (`//: FIXED by the format: a mask word is 32 tokens`). Scope is brace depth: each `{` is classified by the text
  since the last `;`, `{` or `}`, and a brace is a scope only when that text names `namespace`, `struct`, `class`,
  `union` or `extern "C"` and holds no parenthesis. It misses function-scope constants, literals inside expressions,
  other types (`bool`, `float`, `unsigned int`, an alias), a split declaration and several declarators on a line; a
  trailing comment does not state the reason.

Their findings on the tree of 2026-11-24 are in the baseline: 14 of the first rule, all `decode/decode.cu`'s (the entmax
forward's shared memory is a tile of `SmemLayout`); 0 of the second (that tile's count is an expression); and 65 of the
third.

## <a id="standards"></a>The standards lint, in detail

**`ast_grep_gate.py`** scopes `ast-grep scan --config sgconfig.yml` to `csrc/`: a bare scan defaults to the whole repo
and re-fires every rule's must-fire fixture under `tools/lint/fixtures/` as a gating error. Each rule's `message` names
its standards section and incident; `--test-fixtures` is the separate self-test that each rule still fires on its own
fixture.

**`lint_standards.py`** carries what ast-grep cannot express: the unroll discipline (§6: a dynamic-bound loop in device
code carries `#pragma unroll 1`, scoped to `__global__`/`__device__` bodies by brace counting), the torch include as a
text belt to the ast-grep rule, a CTA or grid barrier inside a declared barrier-free region (`__syncwarp` is intra-warp
and excluded), a committed doc or script wrapping a command in an external `flock` on the GPU lock path (scoped
to `docs/`, `measure/`, `tools/`, `CLAUDE.md` and `README.md`, the standards among them), the
file-reading checks
(`check_vendored_pin`, a function of the
tree's files, never of an imported `fsla`), `check_facts_reader_confined` (no file under `csrc/` outside `csrc/fsla/src/common/plan/` names
`Reader`, the planner's accessor of the facts: [`../common/plan/planner.md#the-reader`](../common/plan/planner.md#the-reader)), and
`check_repo_portability` (§17): every tracked file scanned for an absolute home-directory path, a citation of an
out-of-repo journal by path, a real worktree path or a session's scratch path, with no allowlist (its own source is
excluded by name, because its patterns spell what they catch). Every finding prints its section, `file:line` and the
incident.

<a id="compile-commands"></a>**The compile database.** `bazel build //:compile_commands` makes `compile_commands.json`
under `.bazel/bin` for an editor, which the tracked `.clangd` names (`CompilationDatabase: .bazel/bin`): each unit's
clang command, as the unit lints run it, for the first built architecture, every path absolute (an unsandboxed,
uncached action: the paths are the editor's own).

<a id="device-profile"></a>The device profile runs clang's CUDA frontend `--cuda-host-only` on the compile's own include roots (the
toolkit's `include/cccl` among them, which nvcc's profile adds and clang does not), and a compiler error anywhere
fails it: the parse is part of the gate, which is why the
analyzer is pinned (a distro clang could not parse a torch-bearing unit and reported used parameters as unused). Both
profiles were validated against planted findings, so a clean pass is the tool running, not an inert config.

Two settings keep a unit's time on its own code. THE ANALYZER RUNS WITHOUT INLINING (`-analyzer-config ipa=none`): the
profile's one analyzer check, `deadcode.DeadStores`, reads each function's own liveness, so inlining callees adds
nothing to it -- it was ~80% of a template-heavy unit's time (`entmax/factor.cu`: 69 s without, 389 s with, the same
findings), and the self-test's fixture plants a dead store the check must still find. THE LINT READS ITS ONE
ARCHITECTURE'S BUILD SET: the host pass has every carried build in the image (`common/select.cuh`'s `in_image`), so a
`build_switch` would instantiate every kernel once a carried build; each architecture is the lint's own
configuration, whose generated `builds_header` (`hardware/registry.py`'s) carries that architecture's builds alone.

## Fixtures

Every lint owns a fixture pair under `tools/lint/fixtures/` (a "must fire" case and a
"must not fire" case, most often in one file). A rule that has never been shown to fire is a rule nobody knows is
connected; one that has never been shown to stay quiet is a rule nobody can afford to make gating. The self-tests are
the commit gate's `//tools/lint:lint_fixtures`: each Python lint's `--test-fixtures` mode, and the unit lints' fixtures
linted as units, each verdict holding exactly the fixture's expected findings.

Fixtures are EXCLUDED from every lint's normal-mode scan (`readability_spacing.py`'s
`py_files()`, `run_vulture.sh`'s `--exclude`, `drift_guards.py`'s `_walk`). The lesson —
`ast_grep_gate.py`'s own header records it, and it has since bitten three lints in this
directory — is that a fixture's deliberate violation re-triggering the real report is
exactly how a hook drifts into being silently disabled.

## Running the roster by hand

```bash
bazel test //:gate                  # the whole roster: every check the commit gate runs
bazel test //:lints                 # the checks over files alone

bazel test //:ratchet_work_codes    # a ratchet's gate
bazel run //tools/lint:work_codes   # the lint alone: its inventory
bazel run //tools/lint:drift_guards
bazel run //tools/lint:constant_parameters
bazel run //tools/lint:r9_enforcement
bazel test //:ratchet_vulture
```

# `tools/` — the map: what each entry point is, who calls it, and what it records

The tools inventory. Every script under `tools/` is here with one line of what it is, what runs it, and
what it writes; a tool with more to say has its own page beside this one, and a tool without one is fully stated by its
own module docstring. A tool that nothing runs is a tool to delete (`DELETIONS.md`), so the CALLED BY column is the
column that matters.

**No tool writes a measurement record except through a declared target.** A measurement is a Bazel target of
`measure/BUILD.bazel`: a timing session, or an instrument whose JSON its store target records (`results_store`), and the
tool itself prints or writes `--json`. A tool run by hand prints or writes `--json` and stores nothing; a stored reading
is its target's ([`measurement.md`](../../measurement.md#instruments)).

## The build

| tool | what it is | called by | writes |
|---|---|---|---|
| [`dev.py`](dev.md) | the one command that sets up, checks and describes a machine (a host, or the devbox container), and checks the devbox image | a person, through its run targets; the commit gate's §23 check | the `environment` record (its container check), `tools/devbox_image.json` |
| [`dev_config.py`](dev_config.md) | every machine-dependent value, read from one directory of JSON files | every tool here, the tests | — |
| [`build_flags.py`](../../build.md) | THE ONE SOURCE of the nvcc flag list | the graph (`bazel/flags.py`) | — |
| `build_record.py` | the build record `_build_config.py`, rule 3 (every architecture a built one) and rule 5 (the vendored tree's pin) | the graph (`bazel/record.py`), `wheels.py` | the record |
| `csrc_stamp.py` | the csrc content hash and the header that compiles it into the binary as the device stamp | `ratify.py` (`--json`), the graph's stamp action | the stamp header |
| [`bootstrap.py`](dev_config.md#bootstrap) | what a tool knows before the dev config's reader is importable: the sibling checkouts, the fsla-devtools checkout; run, the machine's one step outside the graph (the pinned bazelisk) | `dev_config.py`, the graph's front door, a person setting up a machine | the `bazel` link |
| [`toolchains.py`](toolchains.md) | each toolchain as a named record of the assembler a fatbin is built under | `ratify.py`, `dev.py`, the graph's flags and record (`bazel/flags.py`, `bazel/record.py`) | — |
| [`wheels.py`](wheels.md) | the graph's library of a clean commit, packaged as the pure-Python wheel and the binary plugin | a person, at a release (`bazel run //:wheels`) | the wheels |
| `no_source_build.py` | `pyproject.toml`'s build backend, which refuses a source install by name | pip | — |
| `supported.py` | the README's supported-configuration table, generated from the contract (`hardware/`) — a REPORT, not a claim | a person, after an arm change | the table |
| [`kernels.py`](kernels.md) | THE KERNEL LIST: every device unit the build compiles and every kernel the source defines, derived from the tree | `ratify.py`, every kernel gate | — |
| `git-hooks/pre-commit` | THE COMMIT GATE: `bazel test //:gate` on the commit as git will write it ([`build-graph.md`](../../build-graph.md#checks)) | git, every commit | — |
| `git-hooks/pre-push` | THE PUSH'S TRIPWIRE: `bazel run //:tripwire -- --range` over every commit the push sends ([`build-graph.md`](../../build-graph.md#checks)) | git, every push | — |
| `git-hooks/post-commit` | THE REVIEW: `review_gate.py` on HEAD in the background (`review_gate.md`) | git, every commit | — |
| `git-hooks/commit-msg` | A REVIEW'S BLOCK STOPS THE NEXT COMMIT: `review_gate.py --unanswered` on the commit's message ([`review_gate.md`](review_gate.md#blocking)) | git, every commit | — |
| [`review_gate.py`](review_gate.md) | a commit, range or staged diff reviewed by an LLM against CONSTITUTION and KERNEL_STANDARDS, each finding citing a rule and a line; a landed `block` refuses the next commit until a message answers it, a `warn` is advisory | the post-commit and commit-msg hooks, a person | `fsla-review/` in the git common dir |

## The census and the assembled code

| tool | what it is | called by | writes |
|---|---|---|---|
| [`ratify.py`](ratify.md) | the census (every instantiation's registers and spills, from the graph's compiles) | a person, `bazel run //:ratify -- --census`; the template lint | — |
| [`devbox_packages.py`](../../setup.md#the-devbox-image) | the devbox image's package layer adds packages, never changes the base's, and carries nothing at its merged-/usr links | the graph's `//:devbox_packages` | — |
| [`oci_layout.py`](../../setup.md#the-devbox-image) | an OCI image layout, read: its one image's manifest digest (the devbox image's identity), the tree's tag for it, and its layers | `dev.py`'s container commands, the graph's `//:image_tag`, `usrmerge_layer.py`, `devbox_packages.py` | — |
| [`usrmerge_layer.py`](../../setup.md#the-devbox-image) | the package layer for a merged-/usr base: what the packages ship under the base's top-level links into /usr, read from the base, goes under /usr | the graph's `//:devbox_debs` | the image's package layer |
| `sass.py` | THE DISASSEMBLER SEAM: the one `cuobjdump`/`nvdisasm` invocation and line grammar; a frame map is one function's (`section`: offsets restart in every function) | `ratify.py`, `region_ledger.py`, `life_ranges.py` | — |
| [`arith_model.py`](arith_model.md) | the bf16 family's arithmetic (u8 rows, bf16 values and operands, fp32 sums) modelled against the fp64 recurrence on a cell's draw, each rounding alone and all together | a person (a precision form's design) | prints, `--json` |
| [`carry_floor.py`](carry_floor.md) | the hopper carry's floor a token for a routing shape, a box and a cluster, term by term, from the H100 labs' records and stated estimates | a shape, a box, a cluster | the terms, their total, the call's time beside FlashAttention's floor |
| `realized_support.py` | a layer construction's realized routing support: each level's and side's nonzero digits a token in the bundle the producer emits (`realized_support.md`) | a person (a layer reading's regime) | `--json` |
| `checked.py` | THE LAB'S SEAM TO THE CHECKER: warpline's checker imported from the directory the graph hands in, and what the lab asks of it: the model of the texts a built experiment's disassembly wrote, the one judgment (a manifest compared with the declarations), the profile rule (`lab.md#hazards`) | `lab.py` | -- (a module) |
| `sass_bodies.py` | per-instantiation SASS body hashes, the codegen-equivalence instrument | a person, comparing two builds | `--json` |
| `header_selfcheck.py` | every header has its compiled unit: the test over the graph's compile of each header alone (`bazel test //csrc/fsla:header_selfcheck`) | the gate's lints | — |

## The instruments

| tool | what it is | called by | writes |
|---|---|---|---|
| [`op_timeline.py`](op_timeline.md) | a subject's calls under torch's profiler: kernel busy time against the GPU's idle, the gaps and the host work in them | a person | — |
| [`pipe_counters.py`](pipe_counters.md) | the standard counters: the profiler's counters the lab reads for every experiment kernel, and their reading by short name | `lab.py` | -- (a module) |
| [`overlap_gate.py`](overlap_gate.md) | each kernel's perf rows against the largest of their floors (§28) | -- | exit 1 when red |
| [`life_ranges.py`](life_ranges.md) | peak LIVE registers per source region, beside ptxas's allocation | a person | `--json` |
| `region_ledger.py` | executed instructions and stall samples per COMPONENT, against the kernel's budget file | a person, attributing a phase | `--json` |
| [`budgets.md`](budgets.md) | the budget's form and price table: an instruction budget by class, per MMA (KERNEL_STANDARDS §29); no `budgets/` directory exists yet, each kernel's budget file is owed | `region_ledger.py`, the lab | — |
| [`bank_model.py`](bank_model.md) | the shared-memory wavefronts a warp instruction costs, from its lanes' byte addresses (§30) | a person, before a GPU run; the lab | — |
| [`lab.py`](lab.md) | the lab: small SASS-gated CUDA experiments whose measurements become kernel rules ([`docs/lab/`](../../lab/README.md)) | a person | `data:lab/` |
| [`lab_facts.py`](lab.md#graph) | what the graph's disassemblers read of one lab experiment's binary, written once: the cubin, the listings, the demangled names, the toolkit's version | the graph (`bazel/lab.bzl`'s `lab_experiment`) | `//measure/lab:<name>`'s outputs |
| [`lab_check.py`](lab.md#graph) | the lab's checks (the SASS checker, the notices' ledger, the declarations) over a built experiment's files: a refusal fails the build | the graph (the target's validation) | `<name>.sm_90a.check` |

## The oracle's own gates

| tool | what it is | called by | writes |
|---|---|---|---|
| `sanitize_oracle.py` | the compute-sanitizer gate on the facts, decode and entmax oracle cells | the sanitizer lane, `//tests:sanitizer`, which passes it the toolkit's own compute-sanitizer | — |

`tools/lint/` is its own family with its own page ([`lint.md`](lint.md)): the ratchets, the standards lint and their
baselines, and the coverage declaration `tools/coverage.json` ([`coverage.md`](coverage.md)).

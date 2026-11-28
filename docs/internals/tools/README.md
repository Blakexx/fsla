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
| `host_abi.py` | the host toolchain and the library's ABI: the pinned GCC's version and sysroot, and every symbol version `_C.abi3.so` needs, under torch's and Ubuntu 22.04's floors | the graph's `//tests:host_abi` | — |
| `supported.py` | the README's support summary and docs/support.md's contract tables, generated from the contract (`hardware/`) — a REPORT, not a claim | the graph's `//:readme_generated` and `//:support_generated`; a person, after an arm change | the two blocks |
| [`kernels.py`](kernels.md) | THE KERNEL LIST: every device unit the build compiles and every kernel the source defines, derived from the tree | `ratify.py`, every kernel gate | — |
| `git-hooks/pre-commit` | THE COMMIT GATE: `bazel test //:gate` on the commit as git will write it ([`build-graph.md`](../../build-graph.md#checks)) | git, every commit | — |
| `git-hooks/pre-push` | THE PUSH'S TRIPWIRE: `bazel run //:tripwire -- --range` over every commit the push sends ([`build-graph.md`](../../build-graph.md#checks)) | git, every push | — |
| `git-hooks/post-commit` | THE REVIEW: `review_gate.py` on HEAD in the background (`review_gate.md`) | git, every commit | — |
| `git-hooks/commit-msg` | A REVIEW'S BLOCK STOPS THE NEXT COMMIT: `review_gate.py --unanswered` on the commit's message ([`review_gate.md`](review_gate.md#blocking)) | git, every commit | — |
| [`review_gate.py`](review_gate.md) | a commit, range or staged diff reviewed by an LLM against CONSTITUTION and KERNEL_STANDARDS, each finding citing a rule and a line; a landed `block` refuses the next commit until a message answers it, a `warn` is advisory | the post-commit and commit-msg hooks, a person | `fsla-review/` in the git common dir |

## The census and the assembled code

| tool | what it is | called by | writes |
|---|---|---|---|
| [`ratify.py`](ratify.md) | the census (every instantiation's registers and spills, from the library's static manifests, warpline's) | a person, `bazel run //:ratify -- --census`; the template lint | — |
| [`devbox_packages.py`](../../setup.md#the-devbox-image) | the devbox image's package layer adds packages, never changes the base's, and carries nothing at its merged-/usr links | the graph's `//:devbox_packages` | — |
| [`oci_layout.py`](../../setup.md#the-devbox-image) | an OCI image layout, read: its one image's manifest digest (the devbox image's identity), the tree's tag for it, and its layers | `dev.py`'s container commands, the graph's `//:image_tag`, `usrmerge_layer.py`, `devbox_packages.py` | — |
| [`usrmerge_layer.py`](../../setup.md#the-devbox-image) | the package layer for a merged-/usr base: what the packages ship under the base's top-level links into /usr, read from the base, goes under /usr | the graph's `//:devbox_debs` | the image's package layer |
| [`carry_floor.py`](carry_floor.md) | the hopper carry's floor a token for a routing shape, a box and a cluster, term by term, from the H100 labs' records and stated estimates | a shape, a box, a cluster | the terms, their total, the call's time beside FlashAttention's floor |
| `checked.py` | THE SEAM TO WARPLINE: warpline's checker imported from the directory the graph hands in, and what the tools ask of it: the model of the texts a built experiment's disassembly wrote, the one judgment (a manifest compared with the declarations), the profile rule (`lab.md#hazards`); a built extension's image from warpline's cubin cache, a function's model with its life ranges, its inline frames, a SourceCounters dump's source page; warpline reads the SASS and the profiler, this tree attributes and budgets | `lab.py`, `lab_facts.py`, `ratify.py`, `region_ledger.py`, `life_ranges.py`, `sass_bodies.py` | -- (a module) |
| `sass_bodies.py` | per-instantiation SASS body hashes, the codegen-equivalence instrument, on warpline's line grammar | a person, comparing two builds; `tests/unit/test_arm_family.py` | `--json` |
| `header_selfcheck.py` | every header has its compiled unit: the test over the graph's compile of each header alone (`bazel test //csrc/fsla:header_selfcheck`) | the gate's lints | — |

## The instruments

| tool | what it is | called by | writes |
|---|---|---|---|
| [`pipe_counters.py`](pipe_counters.md) | the standard counters: the profiler's counters the lab reads for every experiment kernel, and their reading by short name | `lab.py` | -- (a module) |
| [`life_ranges.py`](life_ranges.md) | peak LIVE registers per source region, beside ptxas's allocation: warpline's pressure, this tree's regions | a person | `--json` |
| `region_ledger.py` | executed instructions and stall samples per COMPONENT, against the kernel's budget file: warpline reads the dump, this tree attributes it | a person, attributing a phase | `--json` |
| [`budgets.md`](budgets.md) | the budget's form and price table: an instruction budget by class, per MMA (KERNEL_STANDARDS §29); no `budgets/` directory exists yet, each kernel's budget file is owed | `region_ledger.py`, the lab | — |
| [`bank_model.py`](bank_model.md) | the shared-memory wavefronts a warp instruction costs, from its lanes' byte addresses (§30) | a person, before a GPU run; the lab | — |
| [`lab.py`](lab.md) | the lab: small SASS-gated CUDA experiments whose measurements become kernel rules ([`docs/lab/`](../../lab/README.md)) | a person | `data:records/fsla/measure/lab/` |
| [`lab_facts.py`](lab.md#graph) | what the graph's disassemblers read of one lab experiment's binary, written once: the cubin, the listings, the demangled names, the toolkit's version | the graph (`bazel/lab.bzl`'s `lab_experiment`) | `//measure/lab:<name>`'s outputs |
| [`lab_check.py`](lab.md#graph) | the lab's checks (the SASS checker, the notices' ledger, the declarations) over a built experiment's files: a refusal fails the build | the graph (the target's validation) | `<name>.sm_90a.check` |

## The oracle's own gates

| tool | what it is | called by | writes |
|---|---|---|---|
| `sanitize_oracle.py` | the compute-sanitizer gate on the facts, decode and entmax oracle cells | the sanitizer lane, `//tests:sanitizer`, which passes it the toolkit's own compute-sanitizer | — |

## The kernels' floors

| tool | what it is | called by | writes |
|---|---|---|---|
| [`bazel/floor.bzl`](floor.md) | the floor gate (§28): a kernel's declared work on its perf cell priced by warpline's floor at the card's rates, the bound written by a build action and judged over its stamped run manifest | `kernel_floor` in `measure/floors/BUILD.bazel`, a kernel each; `//measure/floors:floor` in `//:gate` | `<name>_bound.json` |

`tools/lint/` is its own family with its own page ([`lint.md`](lint.md)): the ratchets, the standards lint and their
baselines, and the coverage declaration `tools/coverage.json` ([`coverage.md`](coverage.md)).

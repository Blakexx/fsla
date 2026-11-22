# Provenance

This repository's git history begins at the extraction, on 2026-08-01. The development before it is not retained.

## Where this code came from

FSLA was developed in a fork of [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
until 2026-08-01, when the FSLA subsystem was extracted into this repository as a standalone package (its first two
commits: the layout, license and this record; then the subsystem, cut from flash-linear-attention). That fork's
history, including the branch the extraction was taken from, was retired on 2026-09-14 and is not published. A commit
hash in this tree that predates the extraction (the "Pre-extraction removals" table in
[`internals/DELETIONS.md`](internals/DELETIONS.md), and older citations in the internals) names a commit of that
retired history: it records what happened and resolves in no published repository. Every later hash resolves here.

The extraction did not filter the fork's history in. It renamed essentially every file and the C++ symbols the
kernels' mangled names carry, so `git blame` continuity, the one benefit of filtering, would not have survived
without an unverifiable rename script, and the filtered history would have carried the superseded V2, CuTe and Triton
lineages and hundreds of flash-linear-attention commits. Starting fresh means every commit here passed this
repository's own gate.

The FSLA layer and model for flash-linear-attention now live on the `fsla` branch of a new fork,
[`Blakexx/flash-linear-attention`](https://github.com/Blakexx/flash-linear-attention): an upstream release plus one
commit series, importing this package.

## Licensing lineage

This package is licensed **Apache-2.0** (see `LICENSE`, `NOTICE`).

The upstream project, flash-linear-attention, is MIT-licensed. The FSLA CUDA kernels, routing producers, census
tooling, and evidence machinery are original work and are not derived from it. The portions whose lineage does trace
to the fork, chiefly the layer scaffolding conventions and small cache helpers, are covered by flash-linear-attention's
copyright and permission notice, which `NOTICE` reproduces as it stood at the extraction; MIT permits distributing
derivative work under another license provided that notice is retained.

### The vendored CUTLASS/CuTe subtree

`csrc/third_party/cutlass` is NVIDIA CUTLASS, BSD-3-Clause, vendored **unmodified**:
the `include/` subtree there is byte-identical to upstream tag `v3.5.1`, commit
`f7b19de32c5d1f3cedfc735c2849f12b537522ee`, which is what makes the pin checkable
against something other than itself (`git clone --branch v3.5.1 && diff -r`).

It is vendored rather than fetched because the build is closed-world: a codegen input
that a network fetch supplies at build time is an input the closed world
cannot describe. `csrc/third_party/cutlass/PIN.json` records the tag, the commit, the
licence and a digest of the whole subtree; the graph's record action (`tools/build_record.py`'s
`vendored_pin`, closed-world rule 5) re-computes that digest on every build and refuses a build whose
vendored headers drifted, and the pin is stamped into `fsla_cu13/_build_config.py` so an
installed wheel answers "which CUTLASS is inside this `.so`?" with no checkout.

BSD-3-Clause requires the copyright notice, the condition list and the disclaimer to
be retained in redistributions; `csrc/third_party/cutlass/LICENSE.txt` is the upstream
text verbatim and `NOTICE` points at it.

## Related repositories

| Repository | Role |
|---|---|
| `Blakexx/flash-linear-attention` | The fla fork: its `fsla` branch holds the FSLA layer and the HuggingFace `FSLAConfig`/`FSLAForCausalLM` model over this package. |
| `Blakexx/fsla-zoology` | Synthetic-task (MQAR and relatives) evaluation harness. |
| `Blakexx/fsla-bench` | The benchmarks and the local measurement suite. |
| `Blakexx/fsla-results` | The measurement records every reported number is read from. |
| `Blakexx/fsla-devtools` | The development tools every FSLA repository shares: the dev config reader, the machine's locks, the commit and push content checks, the central cells, the declared build system and its timing system, and the build graph's shared machinery (a Bazel module). Pinned in `MODULE.bazel`. |
| `Blakexx/fsla-paper` | The paper, the canonical **claims** document; the engineering contract is this repository's docs. They cite each other; neither subsumes the other. |

# `tools/wheels.py` — the two wheels

Mirrors `tools/wheels.py`. It packages the graph's library of a clean commit as the two distributions
([`../../build.md#wheels`](../../build.md#wheels)): `fsla`, pure Python, and `fsla-cu13`, the binary plugin.
`bazel run //:wheels` runs it (`bazel/defs.bzl`'s `fsla_wheels`), handing it the library and its record from its
runfiles.

## The refusals

A wheel names a commit, so the tool refuses before it packages anything: a tree with changes (`_clean_commit`); and a
record that is not this version's build under this toolchain, or is an iteration build (`_verify`).

## The packages

The two wheels are one build's output split by package, so `fsla` never carries a binary and `fsla-cu13` never carries
the Python package: `fsla` is the tracked `fsla/` package and the project's metadata, which `metadata` writes from
`pyproject.toml`'s `[project]` in the form setuptools wrote it, so a resolver reads the same fields it read before;
`fsla-cu13` is the tracked `fsla_cu13/` package, the graph's library and record,
with metadata that requires `fsla` at exactly its version. A commit's wheels are the same bytes each time (fixed member
order and timestamps).

## The check

`check` makes a scratch venv whose only borrowed package directory is the one holding the running interpreter's torch,
installs `fsla` alone and runs `_PROBE` (a kernel call must refuse, naming `fsla[cu13]`), then installs `fsla-cu13` and
runs it again (the binary must load and list its decode arms). It prints both verdicts and `wheels check OK` or `FAIL`.

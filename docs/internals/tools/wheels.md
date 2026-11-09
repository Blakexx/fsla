# `tools/wheels.py` — the two wheels

Mirrors `tools/wheels.py`. It packages the graph's library of a clean, ratified commit as the two distributions
([`../../build.md#wheels`](../../build.md#wheels)): `rola`, pure Python, and `rola-cu13`, the binary plugin.
`bazel run //:wheels` runs it (`bazel/defs.bzl`'s `rola_wheels`), handing it the library, its record and the build's
census from its runfiles.

## The refusals

`_clean_commit` refuses a tree with changes: a wheel names a commit. `_ratified` runs ratify's verdict on the build
(`tools/ratify.py --graph`, `bazel test //:ratified`'s) and refuses on anything but a pass: a wheel ships a ratified
build. `_verify` refuses a record that is not a shippable build of this version: `version`, `iteration` false, `archs`
equal to the toolchain's ratified architectures, `toolchain`, and `manifest_sha256` equal to `ratify.manifest_digest`
over the manifest files the plugin will carry.

## The packages

`rola` takes the tracked files of the `rola/` package (`_package`, its `.gitkeep` placeholders left out, so the
entmax pin's `PIN.json` is its one data file) and the project's metadata: `metadata` writes core metadata 2.4 from
`pyproject.toml`'s `[project]` as setuptools wrote it (the licence's text folded under `License`, the README its body,
the licence files by setuptools' patterns), with the `cu13` extra, `rola-cu13` at exactly this version, first. Its
WHEEL is `py3-none-any`. `rola-cu13` takes the tracked `rola_cu13/` package, the graph's `_C.abi3.so` and
`_build_config.py`, the manifests under a `manifests/` directory inside it, and metadata of its own: its name and
version, the core's author, licence files, classifiers and Python floor in the core's order, `Requires-Dist:
rola==<version>` and the core's torch requirement, under `PLUGIN_TAG`. `_wheel_file` writes RECORD and stores members
in name order with the fixed `EPOCH` timestamp, so a commit's wheels are the same bytes each time.

## The check

`check` makes a scratch venv whose only borrowed package directory is the one holding the running interpreter's torch,
installs `rola` alone and runs `_PROBE` (a kernel call must refuse, naming `rola[cu13]`), then installs `rola-cu13` and
runs it again (the binary must load and list its carry arms). It prints both verdicts and `wheels check OK` or `FAIL`.

---
name: rola-build
description: Load before building the extension (pip install -e, setup.py, nvcc, bazel) or reading a SASS gate on a build: iteration vs gate builds, the shared host budget, waiting on a build, proving the binary.
---
# Building

The build and its parameters: `docs/build.md#building`. What a stage owes and what only a milestone owes:
`docs/KERNEL_STANDARDS.md#s16`.

1. An iteration build compiles one architecture and the arms it needs (`ROLA_CUDA_ARCHS`, `ROLA_CARRY_ARMS`,
   `ROLA_DECODE_ARMS`); a gate build the shipped set, with `ROLA_STRICT_MANIFEST=1` only when it precedes a ratify
   (`docs/build.md#building`, the table). In a checkout with its venv (`bazel run //:dev_env`), iterate with
   `python tools/dev.py build` (`docs/build.md#iteration`), never through a scratch compile.
2. Never set job counts: `setup.py` takes the machine-wide host budget and sizes the jobs
   (`docs/build.md#the-host-budget-every-build-shares-one-pool`, `docs/internals/tools/build_lock.md`).
3. A long build runs in the background with the exit marker as its last step, and is waited on by the marker with a
   bounded wait (`docs/KERNEL_STANDARDS.md#s14`).
4. Prove the binary is the source by the device stamp before using it
   (`docs/measurement.md#preconditions-which-the-targets-acquire`,
   `docs/ratification.md#the-device-stamp-keys-on-the-csrc-content-hash`); then `python tools/sass_gate.py <.so>`
   (`docs/KERNEL_STANDARDS.md#s20`).
5. A build that fails to start: `python tools/dev.py check` first (`rola-dev`).
6. The build graph (`docs/build-graph.md`): `bazel` is the pinned bazelisk (`toolchain.bazel` links it), which runs the
   front door `bazel/bazel`; it takes the host budget for every build phase and sets the jobs, so never pass `--jobs`
   or a cache path (`docs/build-graph.md#front-door`).

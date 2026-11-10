---
name: rola-build
description: Load before building the extension (bazel run //:library, nvcc, bazel) or reading a SASS gate on a build: iteration vs shipped builds, the shared host budget, waiting on a build, proving the binary.
---
# Building

The build and its parameters: `docs/build-graph.md#compile`. What a stage owes and what only a milestone owes:
`docs/KERNEL_STANDARDS.md#s16`.

1. The shipped build is `bazel run //:library`, every ratified architecture and arm; an iteration build compiles one
   architecture and the arms it needs, `bazel run //:library --//:archs=86 --//:decode_arms=3` (`--//:carry_arms`
   likewise; `docs/build-graph.md#compile`), never through a scratch compile.
2. The build graph (`docs/build-graph.md`): `bazel` is the pinned bazelisk (`toolchain.bazel` links it), which runs the
   front door `bazel/bazel`; it takes the machine-wide host budget for every build phase and sets the jobs, so never
   pass `--jobs` or a cache path (`docs/build-graph.md#front-door`).
3. A long build runs in the background with the exit marker as its last step, and is waited on by the marker with a
   bounded wait (`docs/KERNEL_STANDARDS.md#s14`).
4. Prove the binary is the source by the device stamp before using it
   (`docs/measurement.md#preconditions-which-the-targets-acquire`,
   `docs/ratification.md#the-device-stamp-keys-on-the-csrc-content-hash`); the SASS gate runs on every build, and by
   hand it is `bazel run //tools:sass_gate -- <.so>` (`docs/KERNEL_STANDARDS.md#s20`); on an instrument's library,
   `--instrument <its declarations>`, a step signature its probes keep declared there with the cost it measures
   (`docs/internals/tools/sass_gate.md#instrument`).
5. A build that fails to start: `bazel run //:check` first (`rola-dev`).

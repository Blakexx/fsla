# `common/arm.cuh` — one arm's key and constants, any family's

Mirrors `csrc/fsla/src/common/arm.cuh`. `ArmKey<N>` is an arm's key: the N integers of a family's declared key
(`<FAMILY>_KEY` in `csrc/fsla/src/<family>/arms.bzl`), in its order; `Arm<INDEX, VALUES...>` hands one row's values to a kernel
or a host body as compile-time constants (`Arm::values`). The header includes nothing, so a per-arm translation unit
(the graph's, `bazel/arms.bzl`) never parses the torch headers the host dispatch needs
(`docs/internals/common/arm_switch.md`).

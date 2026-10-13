---
name: rola-build
description: Load before building the extension (pip install -e, setup.py) or compiling a single TU: iteration vs gate builds, the shared host budget, waiting on a build.
---
# Building

The build: `docs/build.md#building`. When a build is owed: `docs/KERNEL_STANDARDS.md#s16`.

1. An iteration build compiles one architecture and the arms it needs (`docs/build.md#arm-subset`); a gate build the
   shipped set, once a stage.
2. Every compile takes the shared host budget (`docs/internals/tools/build_lock.md`); never set job counts by hand.
3. A long build runs in the background and ends by writing an exit marker; wait on the marker with a bounded wait,
   never by matching process names, and never edit a source while its build runs.
4. Prove the binary is the source before using it (`docs/build.md#verifying-a-built-binary`).
5. At a stage gate, record the per-TU compile ledger (`docs/build.md#compile-ledger`).

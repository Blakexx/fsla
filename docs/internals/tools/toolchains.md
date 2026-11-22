# `tools/toolchains.py` — the toolchain records

Mirrors `tools/toolchains.py`. A **toolchain** is one exact assembler a fatbin is built under. Everything
that asks "which compiler" reads it through this module: `tools/ratify.py`,
`tools/supported.py`, the lints, and the graph's flags and record actions (`bazel/flags.py`, `bazel/record.py`). Rule 1
is one function here, `rule1`, the graph's record action's (docs/build-graph.md#gates).

## The record

`tools/toolchains/<name>.json`, exactly six fields:

| field | meaning |
|---|---|
| `ptxas` | the whole `ptxas --version` output, whitespace-normalized: the string closed-world rule 1 compares |
| `readers` | `cuobjdump` and `nvdisasm`, each its whole `--version` output: the disassemblers the binary gates read with, which rule 1 compares the same way (`rule1`'s `readers`) |
| `cuda_major` | the CUDA major torch must be built for; torch's extension builder only refuses another major once it compiles, the build refuses before |
| `torch_index` | the PyTorch wheel index serving torch for that major |
| `sanitizer` | compute-sanitizer's whole `--version` text: the sanitizer lane (`tools/sanitize_oracle.py`) refuses another release's, so a binary is read with the toolkit that built it |

Two records naming one assembler are refused: one assembler is one toolchain
([`../../ratification.md`](../../ratification.md)). The built architectures are not the record's: they are the hardware
table's (`measure.hardware.built`).

## Selection

The build takes no toolchain parameter. The graph's toolkit (`@cuda_tools`, pinned by `tools/cuda_toolkit_pin.json`, read by `dev_config.cuda_home()`) is the choice, and the
record whose `ptxas` equals that toolkit's is the toolchain (`for_ptxas`); a toolkit no record names refuses, naming the
declared ones. A second spelling of the choice would be one that can disagree with the compiler that actually runs.
`tools/ratify.py`'s census resolves it per run; a lint that checks a built binary takes the toolchain from that binary's own
`BUILD_CONFIG["ptxas"]`, never from whatever toolkit the checking host has.

`fsla_cu13/_build_config.py` records the toolchain's name beside its `ptxas`, and `tools/supported.py` writes one README row
per (toolchain, built architecture).

`built_extension(root)` is the binary a checkout's build produced, `.bazel/bin/fsla_<toolchain>/_C.*so`, inside the binary
plugin the toolchain's build ships as ([`../../build.md#wheels`](../../build.md#wheels)). It refuses a checkout with none
or with more than one toolchain's, and the tools that fingerprint or disassemble the binary take it from here.

Adding or changing a toolchain is [`../../bringup.md`](../../bringup.md#changing-the-toolchain).

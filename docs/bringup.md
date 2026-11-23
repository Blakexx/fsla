# Bringing up a new architecture

An architecture this package does not build **will refuse to build and
refuse to load**. That is the product statement, not a defect. Bringing one up is a
measurement, and this is the procedure.

You need: the target GPU, a declared toolchain's `nvcc`/`ptxas` (or a new toolchain —
see the last section), and the ability
to open a pull request.

## 1. Tabulate the architecture's resources

A DEVICE PART of `hardware/parts/device` states each architecture whole: its `arch` and `family`, and the device facts of the schema (the INSTRUCTION
capabilities its atoms read among them); the graph generates the header's `Caps`, `tabulated(cc)` and `caps_of(cc)` from it (`//csrc/fsla:hardware_header`;
nothing checks the header in) ([`hardware.md`](internals/common/hardware.md)). The architecture's rates are `measure/rates.json`'s.

Take the FACT fields (`smem_per_sm` from `cudaDevAttrMaxSharedMemoryPerMultiprocessor`,
`max_threads_per_sm` from `cudaDeviceProp::maxThreadsPerMultiProcessor`, `regs_per_sm` from
`cudaDeviceProp::regsPerMultiprocessor`, and the rest) from the CUDA C Programming Guide's
*Compute Capabilities* table, state them in the device part (the rates too, in `measure/rates.json`: the documented
ones with their document, the undocumented ones measured by `smem_rates` and the other labs on the
card). **Do not guess**: an overstated resource makes `ptxas`
silently discard the launch bound, which is a silent occupancy loss on the architecture
nobody checked. `check_arch_table()` (`csrc/fsla/src/common/arch_runtime.cu`, called from
every family's launching entry, carry included) admits the device's card by the planner's contract check: the driver's own
numbers must lie in a registered context ([`arch_runtime.md`](internals/common/arch_runtime.md#check-arch-table)), so a number
the contract does not hold is refused by a user's first launch -- unless you probe it first.

Then PROBE THE DEVICE, which admits it to the contract: run `//measure/lab:device_probe` on the card (`bazel run //tools:lab -- run device_probe --record`, then
`bazel run //tools:lab -- page device_probe --instance LABEL`), read the candidate part it writes (`hardware/parts/device/<id>.candidate.json`), complete what
it names in `unestablished`, and commit it renamed to `<id>.json` with a descriptor composing it. The registry refuses a candidate file, so only a reviewed part
counts; the build's contract check then covers it ([`internals/hardware/registry.md#probe`](internals/hardware/registry.md#probe)).

## 2. Add the architecture as a device part, and read the census

An architecture is added as a device part: it is BUILT when a device part states it (`hardware.registry.built`,
[`hardware.md`](internals/common/hardware.md#builds)); stating it is what makes `--//:archs`'s default name the
architecture, the build gate's rule 3 admit it and the graph compile for it. Then:

```bash
bazel build //:library
bazel run //:ratify -- --census
```

The census reads the graph's compile of the whole dispatch closure (`ptxas -v` of every unit and the SASS of every
object). No GPU is required: it is a compile-time measurement. Read the output before committing the row. Two things are
not automatic:

- **Unhonored bounds.** Any *"Value of threads per SM … is out of range"* means
  that instantiation compiled with no launch bound at all (the census exits 1). Fix step 1 rather than
  recording the numbers.
- **New spills.** If instantiations spill on this architecture and not on the
  built ones, that is a real finding about the register file, and it belongs in
  the pull-request description with the numbers (`--json PATH` keeps them).

## 3. Gate

```bash
bazel test //:gate
bazel run //:ratify -- --self-test
bazel run //:readme_update                 # the README's generated block (the support table)
```

`--self-test` must still fail on every mutant it is required to reject. The
supported-configurations table will move `sm_89` from *not built* to *built*
on its own — it is derived from the arch table, and CI fails if it is edited by hand.

## 4. Build and test on the device

```bash
bazel test //:lane --//:archs=90a,100a     # every tier, on a library built for the new architecture too
```

The ORACLE tier says the kernel computes the right function on this hardware. The
INTEGRATION tier says it computes the same function at every geometry the built
matrix carries there, and
under both state backings — which
is the part a new register file is most likely to disturb, because it is the part
that depends on how the compiler laid the kernel out.

## 5. Open a pull request

**An architecture is always a reviewed PR, never a CI auto-commit.** The diff
should be: the architecture's device part, the regenerated README block, and a description carrying the measured
spill and register profile with any anomaly explained.

## Changing the toolchain

Adding a CUDA version is not a CI matrix cell; it is a **re-measurement of every
architecture**, because every register and spill number is a property of one
assembler. The wheel matrix axis and the toolchain axis are the same axis — that
is the whole closed-world point. Budget it as a measurement run, not as a config
change.

1. **Pin the toolkit**: `tools/cuda_toolkit_pin.json` names NVIDIA's redistributable manifest by its digest, the
   components and the folder; the graph fetches it (`@cuda_tools`) and compiles with it, the one toolkit
   (`tests/unit/test_cuda_toolkit_pin.py` holds the manifest to the pin's version and its compiler to a declared toolchain), and
   `bazel run //:check` holds its nvcc to the pin. The archives place
   libraries in `lib/`, where `nvcc` and torch look in `lib64/`, so the pin's `links` add `lib64 -> lib`.
2. **Write the record**, `tools/toolchains/<name>.json`: the whole `ptxas --version` output, the whole
   `--version` output of the readers (`cuobjdump`, `nvdisasm`: the disassemblers the gates read with), the whole
   `--version` output of compute-sanitizer (`sanitizer`: the sanitizer lane's), the CUDA major, the PyTorch wheel
   index for that major. A torch of another major is refused.
3. **Census before writing.** `bazel run //:ratify -- --census --json <file>` under the new toolkit
   against the current toolchain's census: every entry's spill, registers and in-loop locals. A
   worse entry is a kernel finding, not a verdict on the toolkit.
4. **Build** under the new toolchain, and run the
   suite's timing A/B against the current toolchain's build of the same source in one interleaved
   session, repeated; isolate any regression (compiler against library) before choosing.
5. **Retire the old toolchain** in a deletion-only commit: its record.

# `csrc/rola/src/common/hardware.cuh` — the hardware configuration

The hardware a kernel runs on, mapped to the constants its code and its choices read, in one file:
`csrc/rola/src/common/hardware.json`. `hardware.cuh` is generated from it (`python -m measure.hardware --write`)
and must never be edited by hand; `measure/hardware.py` reads it, resolves it and checks it, and
`tools/lint/lint_standards.py` gates the check on every commit.

<a id="layers"></a>

## Architecture defaults, product overrides

A compute capability fixes the SM's design -- its capabilities, its shared memory and registers, its schedulers,
its MMA instructions, its shared atomic unit -- but not every rate: products of one architecture differ. GeForce
cards run fp16 x fp16 -> fp32 tensor math at half the fp16-accumulate rate, the professional and datacenter
parts of the same chip at full rate (the GA102 whitepaper: Table 3, the RTX A6000 and A40 at 154.8 and 149.7
dense TFLOPS either way; Table 9, the RTX 3090 at 71 against 142). So the file holds:

- `arch.<sm_XX>`: the architecture's defaults -- its `caps` row and its `rates`, each rate at what the SM design
  does (sm_86's fp32-accumulate HMMA at 16 cycles, the full rate).
- `product.<slug>`: a product's overrides over its architecture's defaults (the RTX 3080 Ti's fp32-accumulate
  HMMA at 32 cycles, and every rate measured on it). The slug is `measure/hw_profile.py`'s: the device name with
  the vendor words and the memory suffix dropped (`NVIDIA GeForce RTX 3080 Ti` -> `rtx3080ti`).

A product's rates are its architecture's under its overrides (`hardware.resolve(product=...)`); an
architecture's alone are `hardware.resolve(arch)`. A device with no product row runs on its architecture's
defaults, which are what the SM design does.

<a id="consumers"></a>

## Two consumers

- COMPILE TIME reads capabilities only, and only an architecture's: a binary is built per compute capability and
  cannot know the product it will meet. `hardware.cuh` carries every tabulated architecture's `Caps` row,
  `tabulated(cc)` and `caps_of(cc)`; [`arch_caps.cuh`](arch_caps.md) includes it and derives residency, staging
  depth and the register budget, and [`arch_runtime.cu`](arch_runtime.md) checks the row against the driver.
- RUN TIME reads rates: a choice among exact forms (the readout's accumulation, a sparse side's digit groups) prices
  them with the device's resolved rates -- the binary carries every architecture's and product's MMA rates
  (`kMmaRows`), found by the device's name ([`arch_runtime.md#mma-quanta`](arch_runtime.md#mma-quanta)) -- which reach
  a kernel as quanta ([`cost.cuh`](cost.md)): the same device always gets the same numbers.

`hardware.cuh` also carries the MMA kinds (`Mma`), the order a product's quanta carry their MMA rates in.

<a id="sources"></a>

## Every rate has a source

A rate is `{"cycles": ..., <source>}` with exactly one source:

- `measured` -- a lab (`measure/lab/results/<lab>/sm_XX.json`), its result `row`, and the product it was measured
  `on` (a product's own overrides are measured on it). The check refuses a value off its record's median by more
  than its rounding, or a record taken on another device.
- `documented` -- the public document and the figure in it.
- `announced` -- a value no source supports there yet, with why (sm_80's atomics are the RTX 3080 Ti's until a
  rented A100 measures them): a known gap in plain sight, never a silent default.

The rates: `schedulers` (an SM), `issue` (an instruction a scheduler), `wavefront` (a shared-memory load
wavefront an SM), `atomic` and `atomic_wavefront` (a shared atomic instruction's least and each of its
wavefronts an SM), `mma.<kind>` (an MMA a scheduler) and `tax.<kind>.<class>` (what an instruction of the
class adds to the kind's MMA chain, read by the issue-hiding gate, not by a choice). NVIDIA documents issue,
the shared-memory bank rate and peak tensor rates; a shared atomic's cost and the taxes are measured
([`smem_rates`](../../lab/smem_rates.md), [`tensor_tax`](../../lab/tensor_tax.md)).

<a id="check"></a>

## The check

`python -m measure.hardware --check`: the header is what the file generates; every architecture's capabilities
are the schema's; every built architecture (one with a manifest under `tools/manifests/`) has a row whose
defaults can price a choice (`schedulers`, `issue`, `wavefront`, `atomic`, `atomic_wavefront`, every MMA kind);
every product names a tabulated architecture and its device is that product; every rate is a field the schema
names, with exactly one source that holds.

<a id="extend"></a>

## Extending it

- A product: its block under `product`, with the overrides that differ from its architecture (and a
  confirmation lab on the card).
- An architecture: its `caps` row and its default rates (documented where a document gives them, the rest
  measured), then `--write`; [`bringup.md`](../../bringup.md) walks it.
- A rate: a field in the schema's vocabulary, a value with its source in each row that has it.

# The hardware configuration (the generated `hardware.cuh`)

THE ONE TABLE OF HARDWARE FACTS is the registry's device parts (`hardware/parts/device/`, [`../hardware/registry.md`](../hardware/registry.md#architectures)):
a device part states an architecture whole -- its name, its family and its device facts. The owner's ruling (2026-11-22): "isnt the descritpr a thing we
will want to reuse for other kernels, not just carry? each kernel just matahces on the fields it cares about on these immutable glboal hardware facts". The
kernels' header `common/hardware.cuh` is generated from the device parts by the build graph (`//csrc/fsla:hardware_header`, `hardware/emit.py --arch-header`)
into the units' include path and is checked in nowhere. The measured and documented RATES and the PRODUCT table are measurement inputs, not hardware facts: they
live in `measure/rates.json`, which `measure/rates.py` reads, resolves and checks, and `tools/lint/lint_standards.py` gates the check on every commit.

<a id="layers"></a>

## Architecture defaults, product overrides

A compute capability fixes the SM's design -- its capabilities, its shared memory and registers, its schedulers,
its MMA instructions, its shared atomic unit -- but not every rate: products of one architecture may differ. So
`measure/rates.json` holds:

- `arch.<sm_XX>`: the architecture's default `rates`, each at what the SM design does (sm_90a's schedulers, issue and shared
  wavefront from the programming guide, its shared atomics measured, its MMA rates the whitepaper's peak). An architecture named there is one the
  registry builds.
- `product.<slug>`: a product's overrides over its architecture's defaults (the H100's has none yet). The slug is
  `measure/rates.py`'s `slugify_gpu`: the device name with the vendor words dropped and cut at the first `-`
  (`NVIDIA H100 80GB HBM3` -> `h10080gbhbm3`); the measurement side finds a device's product by it, the binary does not.

A product's rates are its architecture's under its overrides (`rates.resolve(product=...)`); an
architecture's alone are `rates.resolve(arch)`. A device with no product row runs on its architecture's
defaults, which are what the SM design does; the binary runs a device on its architecture's build whatever its name.

<a id="families"></a>

## Families, builds and products

Every device part names its FAMILY -- `hopper` (sm_90a: the part is named as it is compiled, with the architecture's own features; its compute capability
900) -- the name a kernel's schedule is selected by (`Family`: one member a family the device parts state, in the order of their compute
capabilities; the constitution's R8). Instruction differences stay at the atom, read at the compiling architecture.

<a id="builds"></a>
A BUILD is the key a kernel compiles at: `Build{cc}`, one for each architecture a device part states (`registry.architectures()`); a device runs its
architecture's build, so a kernel is compiled once an architecture, never once a product. Every kernel that reads a build trait
compiles at each carried build. An architecture is BUILT when a device part states it (`registry.built`: the graph's default `--//:archs`,
`BUILT_ARCHS` in `@fsla_env//:flags.bzl`); an architecture no part states is not tabulated either (`tabulated(cc)` is false for it).

<a id="builds-renamed"></a>
A kernel templated on its build is the same code under a new name, so a refactor compares by signature:
`tools/sass_bodies.py --compare A B --drop-builds` keys both maps by the demangled signature with each kernel's
`BuildTag<Build{...}>` argument dropped -- refused where the argument is not its own image's, since a build's kernels compile in its image
alone -- and a kernel whose only template argument was its build keyed as the untemplated kernel it replaced.
Every comparison, with the flag or without, reports the source stamp (`csrc_stamp_kernel`, which moves with
every source edit) apart, never as a verdict.

<a id="build-tag"></a>
A kernel takes its build as a TYPE, `BuildTag<B>` (its `build` and `traits` as constants): nvcc 13.0's host stub
cannot name a kernel's class-type template argument whose type sits in a namespace (it prints the type's mangled name),
while a class template carrying it compiles -- so host templates and plans take `Build` itself, kernels its tag.

<a id="traits"></a>
`traits_of(Build)` is what a kernel's plan reads: the family and the architecture's `Caps`. The window W is no trait: it is derived from the device's
facts by the prefill launch function's matcher ([`window.md`](window.md)).

<a id="consumers"></a>

## Two consumers

- COMPILE TIME reads a BUILD's traits (above): its architecture's capabilities. `hardware.cuh` carries every device part's `Caps` row (`cc`, `regs_per_sm`, `smem_per_sm`, `smem_per_cta_optin`
  and the three instruction flags the atoms select code on, the only fields a kernel source reads), `tabulated(cc)` and `caps_of(cc)`, the families
  and every build's `traits_of(Build)`; [`arch_caps.cuh`](arch_caps.md) includes it and names the compiling target's row, and
  [`arch_runtime.cu`](arch_runtime.md) admits the device by the planner's contract check (the driver's numbers lie in a registered
  context), not by a second table of them.
- RUN TIME may read rates: `measure/rates.py` resolves a product's rates by the device's name, for the measurement tools. No form is chosen by a card's
  rates (`docs/CONSTITUTION.md#r17`).

<a id="sources"></a>

## Every rate has a source

A rate (`measure/rates.json`) is `{"cycles": ..., <source>}` with exactly one source:

- `measured` -- a lab (`measure/lab/results/<lab>/sm_XX.json`), its result `row`, and the product it was measured
  `on` (a product's own overrides are measured on it). The check refuses a value off its record's median by more
  than its rounding, or a record taken on another device.
- `documented` -- the public document and the figure in it.
- `announced` -- a value no source supports there yet, with why: a known gap in plain sight, never a silent default.

The rates: `schedulers` (an SM), `issue` (an instruction a scheduler), `wavefront` (a shared-memory load
wavefront an SM), `atomic` and `atomic_wavefront` (a shared atomic instruction's least and each of its
wavefronts an SM), and `mma.<kind>` (an MMA a scheduler); the schema lists the MMA kinds. NVIDIA documents
issue, the shared-memory bank rate, peak tensor rates and the arithmetic pipes' throughputs; a shared atomic's cost,
the shared pipe's and the global path's other prices are measured
([`smem_rates`](../../lab/smem_rates.md), [`drain`](../../lab/drain.md)); the instruction taxes and latencies the
ampere line's MMA-chain labs measured are the branch `ampere`'s, and a hopper schedule's are owed to its own labs.

<a id="shared"></a>

### The shared-memory pipe's other prices

What the pipe takes, an SM, beyond a load's wavefront (`wavefront`) and a shared atomic's (`atomic`,
`atomic_wavefront`, measured in [`smem_rates`](../../lab/smem_rates.md) on the H100). The table holds no rate for the kinds
below; the figures are the `ampere` branch's measurements: `store` and `store_wavefront` (a plain shared store takes the larger of 2
cycles an instruction and ~1.14 a wavefront: 4 bytes a lane costs twice what 8 or 16 do a byte); `copy_bypass` and
`copy_access` (a `cp.async.cg` copy's write into shared memory, 0.5 cycles a 128 bytes; a `.ca` copy's, 5.4);
`global_read` and `global_fill` (a global load's data read out of the banks, 1.19 a 128 bytes, and what a line from L2
adds by being written into them first, 0.38).
No rate row states them. Where a row does state a rate, a count times it bounds perfect overlap and full use
(`docs/KERNEL_STANDARDS.md` §28, §29 item 14); what a composed kernel reaches is read from its timeline
(`docs/KERNEL_STANDARDS.md#stepped`).

<a id="global"></a>

### The SM's global path

Measured with every SM busy, on the `ampere` branch; the table holds no rate for them:
`l2_line`, the cycles an SM's share of L2 takes to deliver a 128-byte line (`smem_rates`' `cp.async.cg` stream alone:
~18 bytes a cycle an SM), and `red_request`, a global atomic's line request (the [`drain`](../../lab/drain.md) lab's
skipping form: ~5 cycles).

<a id="pipes"></a>

### The arithmetic pipes

Cycles a warp instruction holds its pipe a scheduler, from the CUDA C++ Programming Guide's throughput table (results
a clock an SM, over four schedulers and 32 lanes): `alu`, `fma`, `fmaheavy` (integer multiply and multiply-add) and
`xu` (special functions, conversions). The table holds no rate for them.

<a id="check"></a>

## The check

`python -m measure.rates --check` (the lint's `check_hardware`): every architecture named in `measure/rates.json` is one the registry builds; every built
architecture has a row whose defaults carry its pipes' and MMAs' rates (`schedulers`, `issue`, `wavefront`, `atomic`, `atomic_wavefront`, every MMA kind);
every product names an architecture priced there and its device is that product; every rate is a field the schema
names, with exactly one source that holds. The registry's own check (`registry.check`) refuses a device part without its `arch`, its `family` or its
facts, and an architecture stated by two parts.

<a id="extend"></a>

## Extending it

- A product: its block under `product` in `measure/rates.json`, with the overrides that differ from its architecture (and a
  confirmation lab on the card).
- An architecture: its device part, then its default rates (documented where a document gives them, the rest
  measured); [`bringup.md`](../../bringup.md) walks it.
- A rate: a field in the schema's vocabulary, a value with its source in each row that has it.

# The hardware configuration (the generated `hardware.cuh`)

The hardware a kernel runs on, mapped to the constants its code and its choices read, in one file:
`csrc/fsla/src/common/hardware.json`. `hardware.cuh` is generated from it by the build graph (`//csrc/fsla:hardware_header`, `python -m measure.hardware --header`)
into the units' include path and is checked in nowhere; `measure/hardware.py` reads it, resolves it and checks it, and
`tools/lint/lint_standards.py` gates the check on every commit.

<a id="layers"></a>

## Architecture defaults, product overrides

A compute capability fixes the SM's design -- its capabilities, its shared memory and registers, its schedulers,
its MMA instructions, its shared atomic unit -- but not every rate: products of one architecture may differ. So
the file holds:

- `arch.<sm_XX>`: the architecture's defaults -- its `caps` row and its `rates`, each rate at what the SM design
  does (sm_90a's schedulers, issue and shared wavefront from the programming guide, its shared atomics measured, its
  MMA rates the whitepaper's peak).
- `product.<slug>`: a product's overrides over its architecture's defaults (the H100's has none yet). The slug is
  `measure/hardware.py`'s `slugify_gpu`: the device name with the vendor words dropped and cut at the first `-`
  (`NVIDIA H100 80GB HBM3` -> `h10080gbhbm3`); the measurement side finds a device's product by it, the binary does not.

A product's rates are its architecture's under its overrides (`hardware.resolve(product=...)`); an
architecture's alone are `hardware.resolve(arch)`. A device with no product row runs on its architecture's
defaults, which are what the SM design does; the binary runs a device on its architecture's build whatever its name.

<a id="families"></a>

## Families, builds and products

Every architecture row names its FAMILY in its `build` block -- `hopper` (sm_90a: the row is
named as it is compiled, with the architecture's own features; its compute capability 900),
`blackwell` (reserved) -- the name a kernel's schedule is selected by (`Family`; the constitution's R8). Instruction
differences stay at the atom, read at the compiling architecture.

<a id="builds"></a>
A BUILD is the key a kernel compiles at: `Build{cc}`, the architecture's own `build` block (`hardware.builds()`); a device runs its
architecture's build, so a kernel is compiled once an architecture, never once a product. Every kernel that reads a build trait
compiles at each carried build. A product row holds no build of its own: the check refuses one.

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
`traits_of(Build)` is what a kernel's plan reads: the family, the window W ([`window.md`](window.md): a card's,
chosen per architecture) and the architecture's `Caps`. An architecture is BUILT when its build chooses a window
(`measure.hardware.built`: the graph's default `--//:archs`, `BUILT_ARCHS` in `@fsla_env//:flags.bzl`); a tabulated
architecture that is not built leaves it unchosen. sm_90a's is 1,024, provisional until the sm_90 carry's design
picks it (Blake, 2026-11-09). The check
(`measure/hardware.py --check`) refuses an architecture whose build names no family of `FAMILIES` or a trait outside
the family and `BUILD_TRAITS`, and a product holding a `build`.

<a id="consumers"></a>

## Two consumers

- COMPILE TIME reads a BUILD's traits (below): its architecture's capabilities and the build choices. `hardware.cuh`
  carries every tabulated architecture's `Caps` row, `tabulated(cc)` and `caps_of(cc)`, and every build's
  `traits_of(Build)`; [`arch_caps.cuh`](arch_caps.md) includes it and names the compiling target's row, and
  [`arch_runtime.cu`](arch_runtime.md) admits the device by the planner's contract check (the driver's numbers lie in a registered
  context), not by a second table of them.
- RUN TIME may read rates: the binary carries every architecture's and product's MMA rates (`kMmaRows`), a
  product's found by the device's name. No entry reads them on this line: no form is chosen by a card's rates
  (`docs/CONSTITUTION.md#r17`).

`hardware.cuh` also carries the MMA kinds (`Mma`), the order a product's quanta carry their MMA rates in.

<a id="sources"></a>

## Every rate has a source

A rate is `{"cycles": ..., <source>}` with exactly one source:

- `measured` -- a lab (`measure/lab/results/<lab>/sm_XX.json`), its result `row`, and the product it was measured
  `on` (a product's own overrides are measured on it). The check refuses a value off its record's median by more
  than its rounding, or a record taken on another device.
- `documented` -- the public document and the figure in it.
- `announced` -- a value no source supports there yet, with why: a known gap in plain sight, never a silent default.

The rates: `schedulers` (an SM), `issue` (an instruction a scheduler), `wavefront` (a shared-memory load
wavefront an SM), `atomic` and `atomic_wavefront` (a shared atomic instruction's least and each of its
wavefronts an SM), `mma.<kind>` (an MMA a scheduler), and `shared.<kind>`, `global.<kind>` and `pipe.<name>` (below; the schema's lists). NVIDIA documents
issue, the shared-memory bank rate, peak tensor rates and the arithmetic pipes' throughputs; a shared atomic's cost,
the shared pipe's and the global path's other prices are measured
([`smem_rates`](../../lab/smem_rates.md), [`drain`](../../lab/drain.md)); the instruction taxes and latencies the
ampere line's MMA-chain labs measured are the branch `ampere`'s, and a hopper schedule's are owed to its own labs.

<a id="shared"></a>

### The shared-memory pipe's prices (`shared.*`)

What the pipe takes, an SM, beyond a load's wavefront (`wavefront`) and a shared atomic's (`atomic`,
`atomic_wavefront`, measured in [`smem_rates`](../../lab/smem_rates.md) on the H100). The kinds below are not tabulated
for the hopper yet (the ampere branch measured them): `store` and `store_wavefront` (a plain shared store takes the larger of 2
cycles an instruction and ~1.14 a wavefront: 4 bytes a lane costs twice what 8 or 16 do a byte); `copy_bypass` and
`copy_access` (a `cp.async.cg` copy's write into shared memory, 0.5 cycles a 128 bytes; a `.ca` copy's, 5.4);
`global_read` and `global_fill` (a global load's data read out of the banks, 1.19 a 128 bytes, and what a line from L2
adds by being written into them first, 0.38). [`tools/pipes.py`](../tools/pipes.md) prices a launch's traffic by these.

<a id="global"></a>

### The SM's global path (`global.*`)

Measured with every SM busy (not tabulated for the hopper yet):
`l2_line`, the cycles an SM's share of L2 takes to deliver a 128-byte line (`smem_rates`' `cp.async.cg` stream alone:
~18 bytes a cycle an SM), and `red_request`, a global atomic's line request (the [`drain`](../../lab/drain.md) lab's
skipping form: ~5 cycles). [`tools/pipes.py`](../tools/pipes.md) prices a launch's L2 lines and atomic requests by
them.

<a id="pipes"></a>

### The arithmetic pipes (`pipe.*`)

Cycles a warp instruction holds its pipe a scheduler, from the CUDA C++ Programming Guide's throughput table (results
a clock an SM, over four schedulers and 32 lanes): `alu`, `fma`, `fmaheavy` (integer multiply and multiply-add) and
`xu` (special functions, conversions). Not tabulated for the hopper yet.

<a id="check"></a>

## The check

`python -m measure.hardware --check`: the header is what the file generates; every architecture's capabilities
are the schema's; every built architecture (one whose build chooses a window) has a row whose
defaults carry its pipes' and MMAs' rates (`schedulers`, `issue`, `wavefront`, `atomic`, `atomic_wavefront`, every MMA kind);
every product names a tabulated architecture and its device is that product; every rate is a field the schema
names, with exactly one source that holds.

<a id="extend"></a>

## Extending it

- A product: its block under `product`, with the overrides that differ from its architecture (and a
  confirmation lab on the card).
- An architecture: its `caps` row and its default rates (documented where a document gives them, the rest
  measured), then the build; [`bringup.md`](../../bringup.md) walks it.
- A rate: a field in the schema's vocabulary, a value with its source in each row that has it.

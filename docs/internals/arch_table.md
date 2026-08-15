# `csrc/rola/src/arch_table.cuh` — the closed-world arch table

The compile-time SM-resource rows every `__launch_bounds__` in this extension is
derived from, and the gate that refuses an architecture nobody measured. It is
header-only and `constexpr` on purpose: `__launch_bounds__` arguments must be
compile-time constants, so there is nothing here to link against.

Symbols: `arch` is a compute capability in the `__CUDA_ARCH__` encoding (`860` =
SM 8.6); a CTA is one thread block; "residency" is CTAs resident per SM.

---

## <a id="own-header"></a>1. Why this is its own header

The closed-world codegen policy's **rule 4** (runtime refusal of unratified
architectures) and **rule 3** (no arch in the fatbin without a ratified manifest)
both key on exactly this table:

- `sm_arch_tabulated()` is what fails the BUILD for an untabulated compile target;
- the chunk dispatch's `check_arch_table()` (`chunk/chunk.cu`) is what fails the
  FIRST LAUNCH on a device the tables disagree with.

The runtime half moved with the arm it protects. It was
`rola_consumer_check_arch_table()`, in the tiled consumer's `consumer.cu`, until
P67 D2 retired that arm (`DELETIONS.md`); `check_arch_table()` is the same refusal
made at the same moment — a static host function, first statement of every chunk
entry that launches, admitting a device only after it passes — and it carries one
clause the consumer's did not: an untabulated ARCHITECTURE and an unbuilt ARM are
now distinct refusals. It reads `chunk/arch_caps.cuh`'s capability row and, from
`cudaGetDeviceProperties` / `cudaDeviceGetAttribute`, the four FACT columns this
table's own rows declare. Those four are exactly the numbers every staging depth,
CTA target and shared-memory fit was derived from at compile time, which is why
checking them is checking this table.

A supported-architecture list buried in a kernel header is invisible to review and
to the `tools/manifests/` cross-check. The list of architectures this binary claims
to support must be a file one can open.

## <a id="sm-resources"></a>2. The SM's resources, per architecture — a table, never a bare literal

`__launch_bounds__` needs compile-time constants, so these cannot be read from
`cudaDeviceProp` at the point of use. They are therefore a COMPILE-TARGET table
keyed on `__CUDA_ARCH__` — and because a table can be wrong or stale in a way a
driver query cannot, `check_arch_table()` VALIDATES IT AGAINST `cudaDeviceProp` on
the running device and fails loudly on a mismatch.

**The two halves are the point: the table is what the compiler can use, the check
is what makes the table trustworthy.**

Field provenance:

| `SmResources` field | driver attribute it mirrors |
|---|---|
| `smem_per_sm` | `cudaDevAttrMaxSharedMemoryPerMultiprocessor` |
| `max_threads_per_sm` | `cudaDeviceProp::maxThreadsPerMultiProcessor` |
| `regs_per_sm` | `cudaDeviceProp::regsPerMultiprocessor` |

The rows themselves come from the CUDA C Programming Guide, "Compute Capabilities"
table. Two rows cover the ratified set: `8.0` (GA100) at
`{167936, 2048, 65536}`, and `8.6 / 8.7 / 8.9` at `{102400, 1536, 65536}`.

**This helper is a HINT GENERATOR ONLY.** Spec §6.5: the DRIVER's occupancy API
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) remains the sole planning
authority, and every residency number this batch reports is read from it. What this
table decides is one thing: which `__launch_bounds__` the compiler is told to hit.

## <a id="untabulated"></a>3. An untabulated architecture fails the build

Rather than guessing. A wrong launch bound is silently discarded by the driver, so
a guess here would be a silent occupancy loss on the arch nobody checked. The
`static_assert` on `sm_arch_tabulated(kConsumerArch)` is the enforcement point, and
its message names both places a new row must land: `sm_resources()` and the chunk
dispatch's runtime check.

`kConsumerArch` is the arch this translation pass is compiling FOR — the name
predates the tiled consumer's retirement and outlived it, because what it names is
the compile target of every arm in the extension and not one kernel family. In the HOST
pass `__CUDA_ARCH__` is undefined and no launch bound is emitted, so the value is
inert there; it is pinned to the SMALLEST tabulated configuration so that if it
ever stopped being inert it would under-promise rather than over-promise.

## <a id="driver-reserve"></a>4. `kDriverSmemReservePerCta` — the driver's per-CTA shared-memory reserve

MEASURED, and not an arch table entry because **no attribute reports it**. The
driver takes ~1 KiB of shared memory per resident CTA for its own use, so the
residency condition is

```
ctas * (smem_per_cta + 1024) <= smem_per_sm
```

NOT `ctas * smem_per_cta <= smem_per_sm`.

The measurement the value rests on: a build at **32.94 KiB/CTA** satisfies the naive
form for 3 CTAs (98.8 KiB < 100 KiB) and **the driver reports 2**. Any residency
desk-check written in the naive form — §10i.14's ledger among them — over-promises by
exactly this term.

## <a id="mma-quantum"></a>5. `kMmaKQuantum = 16` — the hardware `mma.sync` k-extent

The k-extent of the `mma.sync.aligned.m16n8k16` instruction the chunk consumer
issues on the sm80-class bf16 path, and therefore **the** named constant every quanta-shaped
quantity in this engine is an expression in (spec §1 invariant I1 forbids a
`BC`-derived literal). Two derived names appear everywhere:

| name | definition |
|---|---|
| ATOM | one `kMmaKQuantum`-leaf block in canonical leaf order |
| `QPO` | quanta per owner, `BC / kMmaKQuantum` |

It is a SHAPE of an instruction, never a tuning threshold, so it belongs with the
other compile-time hardware facts rather than with any one consumer of it.

**Why this header owns it.** It is one number for two readers that share no other
header: the chunk consumer (the atom granularity of the state block) and the decode
sibling. A per-file constant in each is two definitions of one hardware fact, which is
how a re-key drifts. `arch_table.cuh` is host-safe and reachable from both. The atom is
also §4.5's page granule, which is why the paging plane counts in the same unit.

The Python side mirrors the same value at `rola.ops.paging.MMA_K_QUANTUM`; a C++
`constexpr` cannot be imported there. `tests/unit/test_paging_atoms.py` and the paged
equivalence fixtures are what keep that mirror honest.

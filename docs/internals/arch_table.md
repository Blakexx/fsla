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
- `rola_consumer_check_arch_table()` (`consumer.cu`) is what fails the FIRST LAUNCH
  on a device the table disagrees with.

A supported-architecture list buried in a kernel header is invisible to review and
to the `tools/manifests/` cross-check. The list of architectures this binary claims
to support must be a file one can open.

## <a id="sm-resources"></a>2. The SM's resources, per architecture — a table, never a bare literal

`__launch_bounds__` needs compile-time constants, so these cannot be read from
`cudaDeviceProp` at the point of use. They are therefore a COMPILE-TARGET table
keyed on `__CUDA_ARCH__` — and because a table can be wrong or stale in a way a
driver query cannot, `consumer_check_arch_table()` VALIDATES IT AGAINST
`cudaDeviceProp` on the running device and fails loudly on a mismatch.

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
its message names both places a new row must land: `sm_resources()` and
`consumer_check_arch_table()`.

`kConsumerArch` is the arch this translation pass is compiling FOR. In the HOST
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

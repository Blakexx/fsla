#pragma once

// THE CLOSED-WORLD ARCH TABLE -- the compile-time SM-resource rows every
// `__launch_bounds__` in this extension is derived from, and the gate that refuses an
// architecture nobody measured.
//
// **WHY THIS IS ITS OWN HEADER.** It was inline in `consumer_kernel.cuh`, a
// 1900-line kernel body. The closed-world codegen policy's rule 4 (runtime refusal of
// unratified architectures) and rule 3 (no arch in the fatbin without a ratified
// manifest) both key on exactly this table: `sm_arch_tabulated()` is what fails the
// build for an untabulated compile target, and `rola_consumer_check_arch_table()`
// (`consumer.cu`) is what fails the first launch on a device the table disagrees
// with. A supported-architecture list buried in a kernel header is invisible to
// review and to the `tools/manifests/` cross-check; the list of architectures this
// binary claims to support must be a file one can open.
//
// It is header-only and constexpr on purpose: `__launch_bounds__` arguments must be
// compile-time constants, so there is nothing here to link against.

namespace rola {

//: ---------------------------------------------------------------------------
//: THE SM's RESOURCES, PER ARCHITECTURE -- A TABLE, NEVER A BARE LITERAL.
//:
//: `__launch_bounds__` needs compile-time constants, so these cannot be read from
//: `cudaDeviceProp` at the point of use. They are therefore a COMPILE-TARGET table
//: keyed on `__CUDA_ARCH__` -- and because a table can be wrong or stale in a way a
//: driver query cannot, `consumer_check_arch_table()` below VALIDATES IT AGAINST
//: `cudaDeviceProp` on the running device and fails loudly on a mismatch. The two
//: halves are the point: the table is what the compiler can use, the check is what
//: makes the table trustworthy.
//:
//: **THIS HELPER IS A HINT GENERATOR ONLY.** Spec §6.5: the DRIVER's occupancy API
//: (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) remains the sole planning
//: authority, and every residency number this batch reports is read from it. What
//: this table decides is one thing: which `__launch_bounds__` the compiler is told
//: to hit.
//:
//: An UNTABULATED architecture FAILS THE BUILD rather than guessing -- a wrong
//: launch bound is silently discarded (see below), so a guess here would be a silent
//: occupancy loss on the arch nobody checked.
struct SmResources {
  int smem_per_sm;         // `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
  int max_threads_per_sm;  // `cudaDeviceProp::maxThreadsPerMultiProcessor`
  int regs_per_sm;         // `cudaDeviceProp::regsPerMultiprocessor`
};

//: CUDA C Programming Guide, "Compute Capabilities" table. Every row is checked
//: against the driver at runtime by `consumer_check_arch_table()`.
constexpr bool sm_arch_tabulated(int arch) {
  return arch == 800 || arch == 860 || arch == 870 || arch == 890;
}
constexpr SmResources sm_resources(int arch) {
  return (arch == 800) ? SmResources{167936, 2048, 65536}   // 8.0  GA100
                       : SmResources{102400, 1536, 65536};  // 8.6 / 8.7 / 8.9
}

//: The arch this translation pass is compiling FOR. In the HOST pass `__CUDA_ARCH__`
//: is undefined and no launch bound is emitted, so the value is inert there; it is
//: pinned to the SMALLEST tabulated configuration so that if it ever stopped being
//: inert it would under-promise rather than over-promise.
#ifdef __CUDA_ARCH__
constexpr int kConsumerArch = __CUDA_ARCH__;
#else
constexpr int kConsumerArch = 860;
#endif
static_assert(sm_arch_tabulated(kConsumerArch),
              "untabulated compute capability: add its row to sm_resources() and to "
              "consumer_check_arch_table() rather than letting the launch bound guess");
constexpr SmResources kSm = sm_resources(kConsumerArch);

//: THE DRIVER'S PER-CTA SHARED-MEMORY RESERVE -- measured, and not an arch table
//: entry because no attribute reports it. The driver takes ~1 KiB of shared memory
//: per resident CTA for its own use, so the condition is
//: `ctas * (smem_per_cta + 1024) <= smem_per_sm`, NOT `ctas * smem <= smem_per_sm`.
//: A build at 32.94 KiB/CTA satisfies the naive form for 3 CTAs (98.8 < 100 KiB) and
//: the driver reported 2. §10i.14's ledger and this batch's own first desk-check were
//: both optimistic by exactly this term.
constexpr int kDriverSmemReservePerCta = 1024;

}  // namespace rola

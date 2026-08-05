#pragma once

// THE CLOSED-WORLD ARCH TABLE -- the compile-time SM-resource rows every
// `__launch_bounds__` here is derived from, and the gate that refuses an architecture
// nobody measured. Header-only and constexpr because `__launch_bounds__` arguments
// must be compile-time constants.
//
// Two enforcement points key on this table and both must stay: `sm_arch_tabulated()`
// fails the BUILD on an untabulated compile target; `rola_consumer_check_arch_table()`
// (`consumer.cu`) fails the FIRST LAUNCH on a device the table disagrees with.
//
// Why it is its own file, the §6.5 hint-generator ruling, and the measured driver
// SMEM reserve: docs/internals/arch_table.md

namespace rola {

//: THE SM's RESOURCES, PER ARCHITECTURE -- A TABLE, NEVER A BARE LITERAL, because
//: `__launch_bounds__` cannot read `cudaDeviceProp` at the point of use. A HINT
//: GENERATOR ONLY: spec §6.5 keeps the driver's occupancy API the sole planning
//: authority -- docs/internals/arch_table.md#sm-resources
struct SmResources {
  int smem_per_sm;         // `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
  int max_threads_per_sm;  // `cudaDeviceProp::maxThreadsPerMultiProcessor`
  int regs_per_sm;         // `cudaDeviceProp::regsPerMultiprocessor`
};

//: The RATIFIED SET. Rows come from the CUDA C Programming Guide's "Compute
//: Capabilities" table and are checked against the driver at runtime by
//: `consumer_check_arch_table()`. Adding an arch means adding it in BOTH places.
constexpr bool sm_arch_tabulated(int arch) {
  return arch == 800 || arch == 860 || arch == 870 || arch == 890;
}
constexpr SmResources sm_resources(int arch) {
  return (arch == 800) ? SmResources{167936, 2048, 65536}   // 8.0  GA100
                       : SmResources{102400, 1536, 65536};  // 8.6 / 8.7 / 8.9
}

//: The arch this translation pass is compiling FOR. Inert in the HOST pass, where it
//: is pinned to the SMALLEST tabulated configuration so that if it ever stopped being
//: inert it would under-promise -- docs/internals/arch_table.md#untabulated
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
//: entry because no attribute reports it. The residency condition is
//: `ctas * (smem_per_cta + 1024) <= smem_per_sm`, NOT `ctas * smem <= smem_per_sm`;
//: dropping the term over-reports residency, as it did once
//: -- docs/internals/arch_table.md#driver-reserve
constexpr int kDriverSmemReservePerCta = 1024;

}  // namespace rola

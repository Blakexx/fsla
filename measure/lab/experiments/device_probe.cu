// measure/lab/experiments/device_probe.cu -- QUESTION: what does THIS device report for every fact the contract registry's
// schema reads from the device -- its compute capability, registers and shared memory per SM, shared memory per CTA (opt-in),
// threads per SM, SM count and L2 bytes -- and do the runtime's two routes to each (cudaGetDeviceProperties and
// cudaDeviceGetAttribute) and the device's own registers (`%nsmid`, `%total_smem_size`) agree?
// METHOD: no timing. The host reads each fact by its property and by its attribute, one CTA of one warp reads the two special
// registers and stores them, and each fact is recorded as a row of its own name (`cc` is major * 100 + minor * 10, as the schema
// says) with the model's other identity (the bus, slot and domain the driver reports); a check holds the two routes equal, the
// registers read nonzero, the device's SM count within the register's SM identifiers, and every fact positive.
// TIMELINE: one warp, one CTA, one read of each register; nothing is timed.
// see docs/internals/tools/lab.md#device-probe
#include "lab.cuh"

namespace device_probe {

//: the two special registers a CTA reads: the SM identifiers the device has (`%nsmid`) and the shared memory a CTA's window holds
//: (`%total_smem_size`)
__global__ void __launch_bounds__(32, 1) registers(uint32_t* out) {
  uint32_t nsmid, total_smem;
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
  asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(total_smem));
  if (threadIdx.x == 0) {
    out[0] = nsmid;
    out[1] = total_smem;
  }
}

//: a fact's row: its value as the one sample every repetition repeats
void fact(const lab::Bench& b, const char* name, const char* unit, long long value) {
  lab::Samples s;
  s.v.assign(static_cast<size_t>(b.reps), static_cast<double>(value));
  b.record(name, unit, s);
}

}  // namespace device_probe

int main(int argc, char** argv) {
  using namespace device_probe;
  lab::Bench b(argc, argv);
  //: the one kernel reads two special registers (a special-register read each) and holds no MMA
  b.expect("device_probe::registers\\(", "S2R", ">=", 1);
  if (b.expect_only) return 0;

  cudaDeviceProp prop{};
  LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
  const auto attr = [](cudaDeviceAttr a) {
    int v = 0;
    LAB_CHECK(cudaDeviceGetAttribute(&v, a, 0));
    return static_cast<long long>(v);
  };
  const long long cc = 100ll * attr(cudaDevAttrComputeCapabilityMajor) + 10ll * attr(cudaDevAttrComputeCapabilityMinor);
  const long long regs = attr(cudaDevAttrMaxRegistersPerMultiprocessor);
  const long long smem_sm = attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
  const long long smem_optin = attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
  const long long threads_sm = attr(cudaDevAttrMaxThreadsPerMultiProcessor);
  const long long sm_count = attr(cudaDevAttrMultiProcessorCount);
  const long long l2 = attr(cudaDevAttrL2CacheSize);

  uint32_t* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, 2 * sizeof(uint32_t)));
  LAB_CHECK(cudaMemset(out, 0, 2 * sizeof(uint32_t)));
  registers<<<1, 32>>>(out);
  LAB_CHECK(cudaDeviceSynchronize());
  uint32_t host[2] = {0, 0};
  LAB_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaFree(out));

  fact(b, "cc", "compute capability x10", cc);
  fact(b, "regs_per_sm", "32-bit registers", regs);
  fact(b, "smem_per_sm", "bytes", smem_sm);
  fact(b, "smem_per_cta_optin", "bytes", smem_optin);
  fact(b, "max_threads_per_sm", "threads", threads_sm);
  fact(b, "sm_count", "SMs", sm_count);
  fact(b, "l2_bytes", "bytes", l2);
  //: the model's other identity, kept in the record beside the facts (the slot varies by machine: not a key)
  fact(b, "pci_domain_id", "id", prop.pciDomainID);
  fact(b, "pci_bus_id", "id", prop.pciBusID);
  fact(b, "pci_device_id_slot", "id", prop.pciDeviceID);
  fact(b, "nsmid_register", "SM identifiers", host[0]);
  fact(b, "total_smem_size_register", "bytes", host[1]);

  b.check("the two routes agree", prop.multiProcessorCount == sm_count && prop.l2CacheSize == l2 && prop.regsPerMultiprocessor == regs
                                      && static_cast<long long>(prop.sharedMemPerBlockOptin) == smem_optin
                                      && static_cast<long long>(prop.maxThreadsPerMultiProcessor) == threads_sm,
          "cudaGetDeviceProperties's multiProcessorCount, l2CacheSize, regsPerMultiprocessor, sharedMemPerBlockOptin and "
          "maxThreadsPerMultiProcessor equal cudaDeviceGetAttribute's");
  b.check("the registers were read", host[0] > 0 && host[1] > 0, "the CTA stored a nonzero %nsmid and %total_smem_size");
  b.check("the SM count is within the register's identifiers", sm_count > 0 && sm_count <= static_cast<long long>(host[0]),
          "the attribute's SM count " + std::to_string(sm_count) + " is at most %nsmid " + std::to_string(host[0]));
  b.check("the facts are positive", cc > 0 && regs > 0 && smem_sm > 0 && smem_optin > 0 && threads_sm > 0 && l2 > 0,
          "every fact the schema reads from the device is nonzero");
  return 0;
}

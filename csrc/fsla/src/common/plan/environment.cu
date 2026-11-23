// csrc/fsla/src/common/plan/environment.cu -- a device's facts read once, and the descriptors made
// from them, a kernel's contract at a time. see docs/internals/common/plan/environment.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <string>

#include "plan/decode/contract.h"
#include "plan/environment.h"
#include "plan/plan_api.cuh"
#include "plan/prefill/contract.h"

namespace fsla::plan {

namespace {

template <typename K>
typename K::Descriptor settle_device(const DeviceFacts& facts);

//: THE ONE READ of a device: the properties and attributes the schema's DEVICE facts name, refused unless every kernel's contract holds the device
DeviceFacts read_device(int device) {
  cudaDeviceProp prop{};
  FSLA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int smem_per_sm = 0, smem_per_cta_optin = 0;
  FSLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  FSLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_cta_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  const int cc = prop.major * 100 + prop.minor * 10;
  char pci[16];
  std::snprintf(pci, sizeof pci, "0x%x", prop.pciDeviceID);
  const std::string model = "cc=" + std::to_string(cc) + " pci=" + pci + " name=" + prop.name;
  Named read = {{"cc", cc},
                {"regs_per_sm", prop.regsPerMultiprocessor},
                {"smem_per_sm", smem_per_sm},
                {"smem_per_cta_optin", smem_per_cta_optin},
                {"max_threads_per_sm", prop.maxThreadsPerMultiProcessor},
                {"sm_count", prop.multiProcessorCount}};
  DeviceFacts facts{std::move(read), model, cc, prop.multiProcessorCount};
  settle_device<decode::Kernel>(facts);
  settle_device<prefill::Kernel>(facts);
  return facts;
}

//: THE DEVICE'S DESCRIPTOR for kernel `K`: the read settled against `K`'s contract, refused when no context holds it
template <typename K>
typename K::Descriptor settle_device(const DeviceFacts& facts) {
  const Planned<K> planned = device_descriptor<K>(facts.read, facts.model);
  STD_TORCH_CHECK(planned.ok(), planned.refusal,
                  ". This binary is closed-world: it runs the devices the contract holds and "
                  "refuses the rest.");
  return planned.descriptor;
}

//: the device's descriptor for kernel `K`, settled once a device
template <typename K>
const typename K::Descriptor& kernel_device(int device) {
  static std::mutex mu;
  static std::map<int, typename K::Descriptor> settled;
  const std::lock_guard<std::mutex> hold(mu);
  const auto at = settled.find(device);
  if (at != settled.end()) return at->second;
  return settled.emplace(device, settle_device<K>(device_facts(device))).first->second;
}

//: THE KERNELS the planner serves: `run` is called with the definition of the one named `kernel`
template <typename Run>
auto with_kernel(const std::string& kernel, Run run) {
  if (kernel == decode::Kernel::name) return run(decode::Kernel{});
  STD_TORCH_CHECK(kernel == prefill::Kernel::name, "no kernel '", kernel,
                  "' is planned: the kernels are ", decode::Kernel::name, ", ",
                  prefill::Kernel::name);
  return run(prefill::Kernel{});
}

Named named_call(const std::string& kernel, const std::vector<int64_t>& call,
                 const std::string_view* names, std::size_t count) {
  STD_TORCH_CHECK(call.size() == count, "a ", kernel, " call supplies ", count, " facts, got ",
                  call.size());
  Named named;
  for (std::size_t i = 0; i < call.size(); ++i) named.emplace_back(names[i], call[i]);
  return named;
}

}  // namespace

const DeviceFacts& device_facts(int device) {
  static std::mutex mu;
  static std::map<int, DeviceFacts> read;
  const std::lock_guard<std::mutex> hold(mu);
  const auto at = read.find(device);
  if (at != read.end()) return at->second;
  return read.emplace(device, read_device(device)).first->second;
}

template <typename K>
typename K::Descriptor descriptor_of(int device, const Named& call) {
  const DeviceFacts& facts = device_facts(device);
  const Planned<K> planned = describe<K>(kernel_device<K>(device), call, facts.model);
  STD_TORCH_CHECK(planned.ok(), planned.refusal);
  return planned.descriptor;
}

template <typename K>
typename K::Descriptor checked_descriptor(const std::vector<std::int64_t>& words_, int device) {
  const Planned<K> planned = from_words<K>(words_);
  STD_TORCH_CHECK(planned.ok(), planned.refusal);
  const std::vector<std::string> differs =
      device_differs<K>(planned.descriptor, kernel_device<K>(device));
  if (!differs.empty()) {
    std::string text;
    for (const std::string& fact : differs) text += " " + fact + ";";
    STD_TORCH_CHECK(false, "the descriptor was made for another device than ", device, ":", text,
                    " the device: ", device_facts(device).model);
  }
  return planned.descriptor;
}

template decode::Kernel::Descriptor descriptor_of<decode::Kernel>(int, const Named&);
template decode::Kernel::Descriptor checked_descriptor<decode::Kernel>(
    const std::vector<std::int64_t>&, int);
template prefill::Kernel::Descriptor descriptor_of<prefill::Kernel>(int, const Named&);
template prefill::Kernel::Descriptor checked_descriptor<prefill::Kernel>(
    const std::vector<std::int64_t>&, int);

std::string fsla_call_facts(std::string kernel) {
  return with_kernel(kernel, [](auto k) {
    std::string out;
    for (std::string_view name : decltype(k)::call_fact_names)
      out += (out.empty() ? "" : ",") + std::string(name);
    return out;
  });
}

std::vector<int64_t> fsla_descriptor(std::string kernel, int64_t device,
                                     std::vector<int64_t> call) {
  return with_kernel(kernel, [&](auto k) {
    using K = decltype(k);
    const Named named =
        named_call(kernel, call, std::begin(K::call_fact_names), std::size(K::call_fact_names));
    return words<K>(descriptor_of<K>(static_cast<int>(device), named));
  });
}

std::string fsla_owed_kernels(std::string kernel, int64_t device, std::vector<int64_t> call) {
  return with_kernel(kernel, [&](auto k) {
    using K = decltype(k);
    const DeviceFacts& facts = device_facts(static_cast<int>(device));
    const Named named =
        named_call(kernel, call, std::begin(K::call_fact_names), std::size(K::call_fact_names));
    const Planned<K> planned =
        describe<K>(kernel_device<K>(static_cast<int>(device)), named, facts.model);
    std::string out;
    for (const std::string& owed : planned.owed) out += (out.empty() ? "" : ",") + owed;
    return out;
  });
}

}  // namespace fsla::plan

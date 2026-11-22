// csrc/fsla/src/common/plan/environment.cu -- a device's facts read once, and the descriptors made
// from them. see docs/internals/common/plan/environment.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <string>

#include "plan/environment.h"
#include "plan/plan_api.cuh"

namespace fsla::plan {

namespace {

//: THE ONE READ of a device: the properties and attributes the schema's DEVICE facts name
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
  const Named read = {{"cc", cc},
                      {"regs_per_sm", prop.regsPerMultiprocessor},
                      {"smem_per_sm", smem_per_sm},
                      {"smem_per_cta_optin", smem_per_cta_optin},
                      {"max_threads_per_sm", prop.maxThreadsPerMultiProcessor},
                      {"sm_count", prop.multiProcessorCount}};
  const Planned planned = device_descriptor(read, model);
  STD_TORCH_CHECK(planned.ok(), planned.refusal,
                  ". This binary is closed-world: it runs the devices the contract holds and "
                  "refuses the rest.");
  return DeviceFacts{planned.descriptor, model, cc, prop.multiProcessorCount};
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

Descriptor descriptor_of(int device, const Named& call) {
  const DeviceFacts& facts = device_facts(device);
  const Planned planned = describe(facts.descriptor, call, facts.model);
  STD_TORCH_CHECK(planned.ok(), planned.refusal);
  return planned.descriptor;
}

Descriptor checked_descriptor(const std::vector<std::int64_t>& words_, int device) {
  const Planned planned = from_words(words_);
  STD_TORCH_CHECK(planned.ok(), planned.refusal);
  const DeviceFacts& facts = device_facts(device);
  const std::vector<std::string> differs = device_differs(planned.descriptor, facts.descriptor);
  if (!differs.empty()) {
    std::string text;
    for (const std::string& fact : differs) text += " " + fact + ";";
    STD_TORCH_CHECK(false, "the descriptor was made for another device than ", device, ":", text,
                    " the device: ", facts.model);
  }
  return planned.descriptor;
}

std::string fsla_call_facts() {
  std::string out;
  for (std::string_view name : kCallFactNames) out += (out.empty() ? "" : ",") + std::string(name);
  return out;
}

namespace {

Named named_call(const std::vector<int64_t>& call) {
  STD_TORCH_CHECK(call.size() == std::size(kCallFactNames), "a call supplies ",
                  std::size(kCallFactNames), " facts (", fsla_call_facts(), "), got ", call.size());
  Named named;
  for (std::size_t i = 0; i < call.size(); ++i) named.emplace_back(kCallFactNames[i], call[i]);
  return named;
}

}  // namespace

std::vector<int64_t> fsla_descriptor(int64_t device, std::vector<int64_t> call) {
  return words(descriptor_of(static_cast<int>(device), named_call(call)));
}

std::string fsla_owed_kernels(int64_t device, std::vector<int64_t> call) {
  const DeviceFacts& facts = device_facts(static_cast<int>(device));
  const Planned planned = describe(facts.descriptor, named_call(call), facts.model);
  std::string out;
  for (const std::string& kernel : planned.owed) out += (out.empty() ? "" : ",") + kernel;
  return out;
}

}  // namespace fsla::plan

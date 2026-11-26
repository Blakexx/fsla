// csrc/fsla/src/common/plan/environment.cu -- a device's facts read once, and the descriptors made
// from them, a kernel's contract at a time. see docs/internals/common/plan/environment.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <string>

#include "plan/decode/contract.h"
#include "plan/environment.h"
#include "plan/factor_backward/contract.h"
#include "plan/factor_forward/contract.h"
#include "plan/levels.h"
#include "plan/mask_prepare/contract.h"
#include "plan/plan_api.cuh"
#include "plan/prefill/contract.h"
#include "plan/segments.h"
#include "plan/softmax_backward/contract.h"
#include "plan/softmax_forward/contract.h"

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
  settle_device<factor_forward::Kernel>(facts);
  settle_device<factor_backward::Kernel>(facts);
  settle_device<softmax_forward::Kernel>(facts);
  settle_device<softmax_backward::Kernel>(facts);
  settle_device<mask_prepare::Kernel>(facts);
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
template factor_forward::Kernel::Descriptor descriptor_of<factor_forward::Kernel>(int,
                                                                                  const Named&);
template factor_backward::Kernel::Descriptor descriptor_of<factor_backward::Kernel>(int,
                                                                                    const Named&);
template softmax_forward::Kernel::Descriptor descriptor_of<softmax_forward::Kernel>(int,
                                                                                    const Named&);
template softmax_backward::Kernel::Descriptor descriptor_of<softmax_backward::Kernel>(int,
                                                                                      const Named&);
template mask_prepare::Kernel::Descriptor descriptor_of<mask_prepare::Kernel>(int, const Named&);

namespace {

//: the window of the card `device`, in tokens; a card with none is refused
int window_of_device(int64_t device) {
  const int window = window_of_cc(device_facts(static_cast<int>(device)).cc);
  STD_TORCH_CHECK(window > 0, "no window is derived for this card");
  return window;
}

//: A STEP'S SHAPE from an entry's arguments, a part outside the op's refused by name
StepShape step_shape(const std::vector<int64_t>& widths, const std::vector<int64_t>& masked,
                     const std::vector<int64_t>& heads, int64_t value_width) {
  STD_TORCH_CHECK(!widths.empty() && widths.size() <= kCarveMaxLevels, "a step has 1 to ",
                  kCarveMaxLevels, " levels");
  STD_TORCH_CHECK(masked.size() == kCarveSides && heads.size() == kCarveSides,
                  "a step has each side's masked levels and batch-heads");
  STD_TORCH_CHECK(heads[0] >= 1 && heads[1] >= 1 && value_width >= 1,
                  "a step has a batch-head or more a side and a value channel or more");
  StepShape s{};
  s.levels = static_cast<int>(widths.size());
  for (int l = 0; l < s.levels; ++l) {
    s.width_bits[l] = carve_width_bits(widths[static_cast<std::size_t>(l)]);
    STD_TORCH_CHECK(s.width_bits[l] > 0, "a level's width is a power of two from 2 to 256");
  }
  for (int side = 0; side < kCarveSides; ++side) {
    s.masked[side] = static_cast<int>(masked[static_cast<std::size_t>(side)]);
    s.heads[side] = heads[static_cast<std::size_t>(side)];
  }
  s.value_width = static_cast<int>(value_width);
  return s;
}

}  // namespace

std::vector<int64_t> fsla_segment_plan(int64_t length, int64_t maximum, int64_t device) {
  STD_TORCH_CHECK(length >= 1 && maximum >= kNoMaximum,
                  "a segment plan takes a token or more and a maximum of 0 (no limit) or more");
  const int window = window_of_device(device);
  const SegmentPlan plan = segment_plan(length, maximum, window);
  STD_TORCH_CHECK(plan.steps > 0, "a call of ", length, " tokens under a maximum of ", maximum,
                  " must be split, and a split sits on a window boundary: the window is ", window,
                  " tokens, so no step of at most ", maximum,
                  " exists. The least maximum that serves a call longer than itself is ", window);
  return {plan.steps, plan.tokens};
}

std::vector<int64_t> fsla_step_bytes(std::vector<int64_t> widths, std::vector<int64_t> masked,
                                     std::vector<int64_t> heads, int64_t value_width,
                                     int64_t tokens, int64_t device) {
  STD_TORCH_CHECK(tokens >= 1, "a step is a token or more");
  const StepShape shape = step_shape(widths, masked, heads, value_width);
  const int window = window_of_device(device);
  const int64_t windows = (tokens + window - 1) / window;
  return {workspace_bytes(shape, windows, window), operand_bytes(shape, windows, window)};
}

int64_t fsla_step_tokens(std::vector<int64_t> widths, std::vector<int64_t> masked,
                         std::vector<int64_t> heads, int64_t value_width, int64_t limit,
                         int64_t device) {
  STD_TORCH_CHECK(limit >= 1, "a memory limit is a byte or more");
  const StepShape shape = step_shape(widths, masked, heads, value_width);
  const int window = window_of_device(device);
  const int64_t tokens = step_tokens_within(shape, limit, window);
  if (tokens == 0) {
    const int64_t held = workspace_bytes(shape, 1, window), given = operand_bytes(shape, 1, window);
    std::string why = "no segmentation serves a memory limit of " + std::to_string(limit);
    why += " bytes on this call: one window of " + std::to_string(window) + " tokens needs ";
    why += std::to_string(held + given) + " bytes, the op's workspace at its worst case ";
    why += std::to_string(held) + " and its projected inputs " + std::to_string(given);
    STD_TORCH_CHECK(false, why);
  }
  return tokens;
}

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

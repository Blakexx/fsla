// csrc/fsla/src/common/select.cu -- a device's card: the device the contract holds, the build it runs, refused by name where this
// binary carries none. see docs/internals/common/select.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <string>

#include "common/select.cuh"
#include "plan/environment.h"

namespace fsla {

namespace {

constexpr bool carried(Build b) {
#define FSLA_BUILD_CARRIED(CC_, V_) (b.cc == (CC_) && b.variant == (V_)) ||
  return FSLA_BUILDS(FSLA_BUILD_CARRIED) false;
#undef FSLA_BUILD_CARRIED
}

std::string carried_list() {
  std::string out;
#define FSLA_BUILD_NAME(CC_, V_) out += (out.empty() ? "" : ", ") + std::string("sm_") + std::to_string((CC_) / 10) \
                                        + (V_ ? " variant " + std::to_string(V_) : "");
  FSLA_BUILDS(FSLA_BUILD_NAME)
#undef FSLA_BUILD_NAME
  return out;
}

//: THE ADMISSION of one device: the planner's read of it (outside the contract, it is refused there), its build carried.
Card admit(int device) {
  const plan::DeviceFacts& facts = plan::device_facts(device);
  const Build build{facts.cc, 0};
  STD_TORCH_CHECK(carried(build), "this binary carries no build for ", facts.model, " (sm_",
                  facts.cc / 10, "); it carries ", carried_list(),
                  ". Build it with its architecture in --//:archs.");
  return Card{device, facts.cc, facts.sms, build, traits_of(build)};
}

}  // namespace

//: -- see docs/internals/common/select.md#card-of
const Card& card_of(int device) {
  static std::mutex mu;
  static std::map<int, Card> cards;
  const std::lock_guard<std::mutex> hold(mu);
  const auto at = cards.find(device);
  if (at != cards.end()) return at->second;
  return cards.emplace(device, admit(device)).first->second;
}

const Card& current_card() {
  int dev = 0;
  FSLA_CUDA_CHECK(cudaGetDevice(&dev));
  return card_of(dev);
}

}  // namespace fsla

// A DELIBERATELY NON-COMPLIANT fixture for tools/lint/drift_guards.py's facts-in-the-template rules.
// Every construct here is one they must report. Not compiled.
#pragma once

namespace fixture {

__global__ void staged(float* out) {
  __shared__ float scratch[256];
  out[0] = scratch[0];
}

using Sized = SmemLayout<Bt, 2, Fixed<Table, std::int32_t, 100>,
                         Share<Ring, std::int64_t,
                               0x40u, 3>>;

constexpr int kWindow = 64;
inline constexpr std::int64_t kStride = 128ul;

//: the window the card decides
//: fixed by nothing in particular
constexpr unsigned kLowercase = 8;

//: FIXED
constexpr long long kNoReason = 1'024;

struct Plan {
  static constexpr unsigned kLeaves = 0x20u;
};

}  // namespace fixture

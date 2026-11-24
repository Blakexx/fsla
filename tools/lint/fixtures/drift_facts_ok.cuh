// The compliant twin of drift_facts_bad.cuh, for tools/lint/facts_in_template.py. Not compiled.
#pragma once

namespace fixture {

// a comment may name __shared__ and Fixed<Table, T, 100>
/* so may a block comment: __shared__ float x[4]; */
using Sized = SmemLayout<Bt, 2, Fixed<Table, std::int32_t, kEntries>,
                         Share<Ring, std::int64_t, 2 * kRing, 3>, Share<Panel, std::int16_t, Bt::rows(8), 1>,
                         Share<Spare, std::int16_t, (4 > 3 ? kRing : 1), 1>>;
using Ratio = Share<Ring, std::int64_t, kRing, 3>;

//: FIXED by the format: a mask word is 32 tokens
constexpr int kWordTokens = 32;

//: the word, then
//: FIXED by the format: a lane is 32 threads
inline constexpr unsigned kLane = 32u;

constexpr int kWindow = kLane * 2;
constexpr float kScale = 0.5f;
constexpr bool kOn = true;

struct Plan {
  //: FIXED by the format: a leaf is 16 tokens
  static constexpr unsigned kLeaves = 0x10u;
  static constexpr unsigned kDerived = kLeaves + 1;
};

__device__ int body() {
  constexpr int kInFunction = 7;
  auto f = [] { constexpr int kInLambda = 9; return kInLambda; };
  return kInFunction + f();
}

}  // namespace fixture

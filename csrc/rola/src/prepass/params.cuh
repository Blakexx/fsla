// csrc/rola/src/prepass/params.cuh -- THE PRE-PASS'S PARAMETER BLOCK, handed to the kernel as one value.
// The pre-pass is the int8 carry's skeleton at two columns a leaf (mass, bound): it consumes the bundle's
// u8 rows and their scales, the write gain and the value scale, and writes two floats a token.
// See docs/internals/prepass/prepass.md#params
#pragma once

#include <cstdint>

#include "common/geom.cuh"

namespace rola::prepass {

struct PrepassParams {
  carry::CarryGeomRT g;
  uint32_t sparse_mask[2];  //: bit l: level l sparse on the side (0 read, 1 write) -- params.md

  const uint8_t* rows_read;  //: [BH, L, wtot] the read side's u8 unit-max rows, levels concatenated
  const uint8_t* rows_write;  //: [BH, L, wtot] the write side's
  const float* scale_read;    //: [BH, L, D] each read level's row maximum
  const float* scale_write;   //: [BH, L, D] each write level's
  const float* gain;          //: [BH, L] g_t, the write gain
  const float* vscale;        //: [BH, L] s_t, the token's value scale
  float* den;                 //: [BH, L] the inter denominator, REDUCED into
  float* scale;               //: [BH, L] the packed panel's per-token bound, REDUCED into
  const int32_t* liveness;    //: [BH, 2, wtot, words] the class-1 words

  int L;               //: tokens in the call
  int liveness_words;  //: ceil(L / 32)
  long long* ledger;
  long long* trace;
  int trace_ctas;
  int trace_cap;
};

//: the ledger's rows, one a component -- params.md
enum PrepassPhase : int {
  kPpFrame = 0,
  kPpWords = 1,
  kPpLists = 2,
  kPpFill = 3,
  kPpFold = 4,
  kPpPublish = 5,
  kPpRescale = 6,
  kPpReadout = 7,
  kPpDrain = 8,
  kPpPhases = 9
};

}  // namespace rola::prepass

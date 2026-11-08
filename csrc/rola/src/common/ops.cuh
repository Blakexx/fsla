#pragma once

// LAYER 2b -- THE BASELINE OPERATIONS, their atoms (`common/atoms/`) and the layers' consistency.
// See docs/internals/common/ops.md

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"
#include "design.cuh"

#include "common/atoms/address.cuh"
#include "common/atoms/copy.cuh"
#include "common/atoms/barrier.cuh"
#include "common/atoms/warp.cuh"
#include "common/atoms/clock.cuh"
#include "common/atoms/mma.cuh"
#include "common/atoms/math.cuh"
#include "common/atoms/reduce.cuh"
#include "common/atoms/memory.cuh"

namespace denseref {
namespace ops {

// 2a/2b CONSISTENCY: a selection no available operation can serve is invalid -- see docs/internals/common/ops.md#consistency

//: `design.cuh` claims REGISTERS is a residency this baseline builds. -- see docs/internals/common/ops.md#static-assert
static_assert(!(design::kBuiltResidencies
                & design::residency_bit(design::StateResidency::REGISTERS))
                  || (kMmaAcceptsRegisterA && arch::kCaps.bf16_cvt_packed),
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency "
              "REGISTERS is selected, but no built `mma` accepts an operand from the "
              "issuing unit's registers (or no packed accumulator->operand conversion "
              "exists to feed it). Either build one or stop claiming the residency.");

static_assert(!(design::kBuiltResidencies & design::residency_bit(design::StateResidency::TMEM))
                  || kMmaAcceptsTmemA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency TMEM "
              "is claimed as built, but no built `mma` sources an operand from tensor "
              "memory.");

//: THE CARVE'S UNIT. Not a capability and not an architecture clause: `mma`,
//: `load_frag` and the d_v carve above are written for a 32-thread unit owning
//: a 16-row slice of the state ... -- see docs/internals/common/ops.md#static-assert-2
static_assert(arch::kCaps.mma_unit_threads == kMmaUnitThreads,
              "NOT BUILT FOR THIS ARCHITECTURE: no built implementation of "
              "`mma` and `load_frag` serves this architecture's MMA unit. The "
              "operations here are written for a 32-thread unit owning a 16-row "
              "slice of the state; a 128-thread warpgroup unit owns 64. Every "
              "structural clause holds -- this is an unwritten implementation, "
              "not a refused structure.");

//: THE ANNOUNCEMENT SITES. Each is a case where the capability row records a -- see docs/internals/common/ops.md#note-l562
inline constexpr design::Announce<arch::kCaps.async_bulk_copy && !kStagingIsBulk>
    kStagingAnnouncement{};

}  // namespace ops
}  // namespace denseref

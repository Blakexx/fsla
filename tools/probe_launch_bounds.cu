// LAUNCH-BOUND PROBE -- registers and spills for EVERY instantiation, read off
// `ptxas -v`. Consumed by `gate_v3_launch_bounds.py`, which turns the reading into an
// asserted invariant; readable on its own when one number needs explaining.
//
// **WHY THIS IS A COMMITTED INSTRUMENT AND NOT A SCRATCH FILE.** The launch bound is
// a STRUCTURAL PLAN QUANTITY (`workflows/20260801-rewrite/rewrite.md` correction (d)):
// left at the default, or stated OUT OF RANGE, `ptxas` silently reclaims the
// occupancy the whole consumer redesign exists to buy. That claim is only checkable
// by compiling and reading the allocator's own report, and the standing process rule
// is that a measurement harness is a commit rather than an artifact of the session
// that happened to run it.
//
// It is NOT part of the extension build: it instantiates kernels, defines no entry
// point, and its only output is the compiler's stderr.
//
// **THE MATRIX IS THE WHOLE MATRIX** -- `BC(2) x D(4) x PREC(2) x DECAY(2) x
// GLOBAL(2) x COMPACT(2)` = 128, the same 128 `launch_consumer` dispatches, compiled for
// the same two `-gencode` targets the extension builds. A probe over a SUBSET leaves
// the un-probed instantiations exactly as unchecked as they were before it existed,
// and the one spill this batch accepted was found on one specific arm out of six.
//
// READ THREE THINGS out of the result, and all three are load-bearing:
//   1. any `Value of threads per SM ... is out of range` warning -- that entry
//      compiled with NO launch bound at all, correction (d)'s exact failure mode;
//   2. `Used N registers`;
//   3. `bytes spill stores / bytes spill loads` -- a bound tight enough to spill is a
//      worse defect than the one it fixes.
//
// Run it through the gate rather than by hand:
//     python benchmarks/ops/gate_v3_launch_bounds.py
#include "v3_consumer_kernel.cuh"

namespace f = fla_rola::v3;
using f::MmaPrec;

#define ROLA_PROBE_INST(BT, DV, BC, D, PREC, DECAY, GLOBAL, COMPACT)              \
  template __global__ void f::consumer::v3_consumer_forward_kernel<                     \
      BT, BC, DV, D, PREC, PREC, DECAY, GLOBAL, COMPACT>(f::V3ConsumerParams);

#define ROLA_PROBE_COMPACT(BT, DV, BC, D, PREC, DECAY, GLOBAL)                    \
  ROLA_PROBE_INST(BT, DV, BC, D, PREC, DECAY, GLOBAL, false)                      \
  ROLA_PROBE_INST(BT, DV, BC, D, PREC, DECAY, GLOBAL, true)
#define ROLA_PROBE_GLOBAL(BT, DV, BC, D, PREC, DECAY)                             \
  ROLA_PROBE_COMPACT(BT, DV, BC, D, PREC, DECAY, false)                           \
  ROLA_PROBE_COMPACT(BT, DV, BC, D, PREC, DECAY, true)
#define ROLA_PROBE_DECAY(BT, DV, BC, D, PREC)                                     \
  ROLA_PROBE_GLOBAL(BT, DV, BC, D, PREC, false)                                   \
  ROLA_PROBE_GLOBAL(BT, DV, BC, D, PREC, true)
#define ROLA_PROBE_PREC(BT, DV, BC, D)                                            \
  ROLA_PROBE_DECAY(BT, DV, BC, D, MmaPrec::kBf16)                                 \
  ROLA_PROBE_DECAY(BT, DV, BC, D, MmaPrec::kTf32x3)
#define ROLA_PROBE_D(BT, DV, BC)                                                  \
  ROLA_PROBE_PREC(BT, DV, BC, 1)                                                  \
  ROLA_PROBE_PREC(BT, DV, BC, 2)                                                  \
  ROLA_PROBE_PREC(BT, DV, BC, 3)                                                  \
  ROLA_PROBE_PREC(BT, DV, BC, 4)
//: THE PROBE COVERS THE WHOLE DISPATCH MATRIX, and that is the property that makes
//: the manifest a closure rather than a sample: 512 instantiations = the exact set
//: `launch_consumer` can reach (`BT(2) x d_v(2) x BC(2) x D(4) x PREC(2) x DECAY(2) x
//: GLOBAL(2) x COMPACT(2)`). An instantiation the dispatch can launch but the probe
//: does not compile would have an UNMEASURED launch bound, which is the state the
//: gate exists to make impossible.
#define ROLA_PROBE_BC(BT, DV)                                                     \
  ROLA_PROBE_D(BT, DV, 16)                                                        \
  ROLA_PROBE_D(BT, DV, 64)
#define ROLA_PROBE_DV(BT)                                                         \
  ROLA_PROBE_BC(BT, 32)                                                           \
  ROLA_PROBE_BC(BT, 64)

ROLA_PROBE_DV(16)
ROLA_PROBE_DV(32)

#undef ROLA_PROBE_DV
#undef ROLA_PROBE_BC
#undef ROLA_PROBE_D
#undef ROLA_PROBE_PREC
#undef ROLA_PROBE_DECAY
#undef ROLA_PROBE_GLOBAL
#undef ROLA_PROBE_COMPACT
#undef ROLA_PROBE_INST

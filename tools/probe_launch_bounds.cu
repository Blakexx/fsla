// LAUNCH-BOUND PROBE -- registers and spills for EVERY instantiation, read off
// `ptxas -v`. Consumed by `ratify.py`, which turns the reading into an
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
// **THE MATRIX IS THE WHOLE MATRIX** -- for the consumer, `BT(2) x d_v(2) x BC(2) x
// D(4) x PREC(2) x DECAY(2) x GLOBAL(2) x COMPACT(2)` = 512, the exact set
// `launch_consumer` can reach, compiled for the same `-gencode` targets the extension
// builds; plus (census extension, below) the decode GEMV's 32 and the union-entmax
// kernels' 64. A probe over a SUBSET leaves the un-probed instantiations exactly as
// unchecked as they were before it existed -- historically, the spill §2.21 accepted
// was found on one specific arm the narrower probes never compiled.
//
// READ THREE THINGS out of the result, and all three are load-bearing:
//   1. any `Value of threads per SM ... is out of range` warning -- that entry
//      compiled with NO launch bound at all, correction (d)'s exact failure mode;
//   2. `Used N registers`;
//   3. `bytes spill stores / bytes spill loads` -- a bound tight enough to spill is a
//      worse defect than the one it fixes.
//
// Run it through the gate rather than by hand:
//     python tools/ratify.py
#include "consumer_kernel.cuh"

namespace f = rola;
using f::MmaPrec;

#define ROLA_PROBE_INST(BT, DV, BC, D, PREC, DECAY, GLOBAL, COMPACT)              \
  template __global__ void f::consumer::consumer_forward_kernel<                     \
      BT, BC, DV, D, PREC, PREC, DECAY, GLOBAL, COMPACT>(f::ConsumerParams);

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

// ---------------------------------------------------------------------------
// CENSUS EXTENSION (coverage-closure, audit item 11): the DECODE GEMV and the
// two UNION-ENTMAX kernels, whose `__launch_bounds__` had never been read off a
// `ptxas -v` report -- their bounds were asserted in comments and honored (or
// not) invisibly. Including the two translation units makes their OWN dispatch
// closures implicitly instantiate the exact launchable sets:
//
//   * `decode_gemv_kernel<DV, D, DECAY, GLOBAL>` -- `DV(2) x D(4) x DECAY(2) x
//     GLOBAL(2)` = 32, emitted by `launch_decode_gemv`'s switch tree;
//   * `union_forward_kernel` / `union_backward_kernel`
//     `<LW, IPT, ALPHA15, OutT>` -- 8 `(LW, IPT)` width rungs x 2 alphas x
//     2 output types (`float`, `__nv_bfloat16`) = 32 per kernel, 64 total,
//     emitted by `launch_forward` / `launch_backward`'s switch trees.
//
// The kernels live in anonymous namespaces inside their `.cu` files, so
// inclusion into this TU is the only way a probe can reach them -- and it is
// also the property that keeps the probe honest: the instantiation set is the
// DISPATCH'S OWN, not a restatement that could drift from it.
//
// KNOWN PROPERTY, stated rather than hidden: nvcc mangles these anonymous-
// namespace kernels with a discriminator derived from THIS file's identity
// (`_GLOBAL__N__<hash>_probe_launch_bounds_cu_<hash>`), so ANY edit to this
// probe changes those 96 manifest keys and the gate reports missing+added
// entries until `--write` re-measures. That is the correct failure direction: a
// measurement-harness change invalidates the measurements it produced.
#include "decode.cu"
#include "entmax.cu"

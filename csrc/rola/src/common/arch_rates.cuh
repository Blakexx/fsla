// csrc/rola/src/common/arch_rates.cuh -- LAYER 1, THE MACHINE'S RATES: what an operation costs on an architecture,
// each value with its source (a lab record that measured it, a document, or an announced gap), and a form's cost as
// the max of its components. see docs/internals/common/arch_rates.md
#pragma once

namespace denseref {
namespace arch {

//: the MMA instructions a cost names: m16n8k16 HMMA by operand and accumulate type, m16n8k32 IMMA
enum Mma : int { kHmmaF16F32, kHmmaF16F16, kHmmaBf16F32, kImmaU8S8K32, kMmaKinds };

//: the instruction classes whose issue beside a running MMA chain was measured (the tax)
enum Op : int {
  kFp32,
  kHalf2,
  kHmax2,
  kAddXor,
  kImad,
  kLop3,
  kPrmt,
  kFmnmx,
  kLds,
  kCvtF32F16x2,
  kCvtU32F32,
  kCvtF16F32,
  kOps
};

//: see docs/internals/common/arch_rates.md#sources
struct Source {
  const char* kind;
  const char* where;
  const char* row;
};

struct Rate {
  double cycles;
  Source source;
};

constexpr Rate measured(double cycles, const char* lab, const char* row) {
  return {cycles, {"measured", lab, row}};
}

constexpr Rate documented(double cycles, const char* where) {
  return {cycles, {"documented", where, ""}};
}

constexpr Rate announced(double cycles, const char* why) {
  return {cycles, {"announced", why, ""}};
}

//: AN ARCHITECTURE'S RATES, cycles: `issue` an instruction a scheduler; `wavefront` a shared-memory load wavefront an
//: SM; `atomic` a shared atomic instruction's least, `atomic_wavefront` each of its wavefronts; `mma` an MMA a
//: scheduler; `tax` what an instruction of a class adds to an MMA chain it issues beside. see
//: docs/internals/common/arch_rates.md#rates
struct Rates {
  bool known;
  Rate schedulers, issue, wavefront, atomic, atomic_wavefront;
  Rate mma[kMmaKinds];
  Rate tax[kMmaKinds][kOps];
};

constexpr Rates rates_of(int cc);

//: an architecture whose taxes are not measured carries another's, each announced
constexpr void taxes_announced_from(Rates& r, int cc) {
  const Rates from = rates_of(cc);
  for (int k = 0; k < kMmaKinds; ++k)
    for (int c = 0; c < kOps; ++c)
      r.tax[k][c] =
          announced(from.tax[k][c].cycles, "sm_86's measured tax, unmeasured on this architecture");
}

//: THE RATED SET: an architecture without a row is not `known`, and a kernel reading rates refuses it at compile time
constexpr Rates rates_of(int cc) {
  Rates r{};
  if (cc == 860) {
    r.known = true;
    r.schedulers = documented(
        4, "CUDA C++ Programming Guide, compute capability 8.x: four warp schedulers an SM");
    r.issue = documented(
        1,
        "CUDA C++ Programming Guide, compute capability 8.x: a scheduler issues one instruction a "
        "clock");
    r.wavefront = measured(1.001, "smem_rates", "ldmatrix.x4, 4 wavefronts a warp / 8 warps");
    r.atomic = measured(2.003, "smem_rates",
                        "red.shared.add, a lane a bank, 1 wavefront a warp / 8 warps");
    r.atomic_wavefront = measured(
        1.000, "smem_rates", "red.shared.add, four lanes a bank, 4 wavefronts a warp / 8 warps");

    r.mma[kHmmaF16F32] = measured(32.132, "tensor_tax", "hmma f16->f32 / alone");
    r.mma[kHmmaF16F16] = measured(16.131, "tensor_tax", "hmma f16->f16 / alone");
    r.mma[kHmmaBf16F32] = measured(32.132, "tensor_tax", "hmma bf16->f32 / alone");
    r.mma[kImmaU8S8K32] = measured(16.134, "tensor_tax", "imma u8.s8 k32 / alone");

    r.tax[kHmmaF16F32][kFp32] = measured(0.506, "tensor_tax", "hmma f16->f32 / ffma");
    r.tax[kHmmaF16F32][kHalf2] = measured(0.213, "tensor_tax", "hmma f16->f32 / hfma2");
    r.tax[kHmmaF16F32][kHmax2] = measured(0.161, "tensor_tax", "hmma f16->f32 / hmax2");
    r.tax[kHmmaF16F32][kAddXor] =
        measured(0.715, "tensor_tax", "hmma f16->f32 / add + xor (iadd3 or imad.iadd, lop3)");
    r.tax[kHmmaF16F32][kImad] = measured(0.103, "tensor_tax", "hmma f16->f32 / imad");
    r.tax[kHmmaF16F32][kLop3] = measured(0.056, "tensor_tax", "hmma f16->f32 / lop3");
    r.tax[kHmmaF16F32][kPrmt] = measured(0.052, "tensor_tax", "hmma f16->f32 / prmt");
    r.tax[kHmmaF16F32][kFmnmx] = measured(0.058, "tensor_tax", "hmma f16->f32 / fmnmx");
    r.tax[kHmmaF16F32][kLds] = measured(0.535, "tensor_tax", "hmma f16->f32 / lds");
    r.tax[kHmmaF16F32][kCvtF32F16x2] = measured(0.057, "tensor_tax", "hmma f16->f32 / f32->f16x2");
    r.tax[kHmmaF16F32][kCvtU32F32] = measured(0.046, "tensor_tax", "hmma f16->f32 / u32->f32");
    r.tax[kHmmaF16F32][kCvtF16F32] = measured(0.077, "tensor_tax", "hmma f16->f32 / f16->f32");

    r.tax[kHmmaF16F16][kFp32] = measured(1.008, "tensor_tax", "hmma f16->f16 / ffma");
    r.tax[kHmmaF16F16][kHalf2] = measured(0.482, "tensor_tax", "hmma f16->f16 / hfma2");
    r.tax[kHmmaF16F16][kHmax2] = measured(0.437, "tensor_tax", "hmma f16->f16 / hmax2");
    r.tax[kHmmaF16F16][kAddXor] =
        measured(1.203, "tensor_tax", "hmma f16->f16 / add + xor (iadd3 or imad.iadd, lop3)");
    r.tax[kHmmaF16F16][kImad] = measured(0.188, "tensor_tax", "hmma f16->f16 / imad");
    r.tax[kHmmaF16F16][kLop3] = measured(0.219, "tensor_tax", "hmma f16->f16 / lop3");
    r.tax[kHmmaF16F16][kPrmt] = measured(0.219, "tensor_tax", "hmma f16->f16 / prmt");
    r.tax[kHmmaF16F16][kFmnmx] = measured(0.219, "tensor_tax", "hmma f16->f16 / fmnmx");
    r.tax[kHmmaF16F16][kLds] = measured(0.845, "tensor_tax", "hmma f16->f16 / lds");
    r.tax[kHmmaF16F16][kCvtF32F16x2] = measured(0.219, "tensor_tax", "hmma f16->f16 / f32->f16x2");
    r.tax[kHmmaF16F16][kCvtU32F32] = measured(0.168, "tensor_tax", "hmma f16->f16 / u32->f32");
    r.tax[kHmmaF16F16][kCvtF16F32] = measured(0.127, "tensor_tax", "hmma f16->f16 / f16->f32");

    r.tax[kHmmaBf16F32][kFp32] = measured(0.506, "tensor_tax", "hmma bf16->f32 / ffma");
    r.tax[kHmmaBf16F32][kHalf2] = measured(0.218, "tensor_tax", "hmma bf16->f32 / hfma2");
    r.tax[kHmmaBf16F32][kHmax2] = measured(0.161, "tensor_tax", "hmma bf16->f32 / hmax2");
    r.tax[kHmmaBf16F32][kAddXor] =
        measured(0.713, "tensor_tax", "hmma bf16->f32 / add + xor (iadd3 or imad.iadd, lop3)");
    r.tax[kHmmaBf16F32][kImad] = measured(0.103, "tensor_tax", "hmma bf16->f32 / imad");
    r.tax[kHmmaBf16F32][kLop3] = measured(0.056, "tensor_tax", "hmma bf16->f32 / lop3");
    r.tax[kHmmaBf16F32][kPrmt] = measured(0.052, "tensor_tax", "hmma bf16->f32 / prmt");
    r.tax[kHmmaBf16F32][kFmnmx] = measured(0.058, "tensor_tax", "hmma bf16->f32 / fmnmx");
    r.tax[kHmmaBf16F32][kLds] = measured(0.534, "tensor_tax", "hmma bf16->f32 / lds");
    r.tax[kHmmaBf16F32][kCvtF32F16x2] =
        measured(0.057, "tensor_tax", "hmma bf16->f32 / f32->f16x2");
    r.tax[kHmmaBf16F32][kCvtU32F32] = measured(0.046, "tensor_tax", "hmma bf16->f32 / u32->f32");
    r.tax[kHmmaBf16F32][kCvtF16F32] = measured(0.079, "tensor_tax", "hmma bf16->f32 / f16->f32");

    r.tax[kImmaU8S8K32][kFp32] = measured(0.976, "tensor_tax", "imma u8.s8 k32 / ffma");
    r.tax[kImmaU8S8K32][kHalf2] = measured(0.411, "tensor_tax", "imma u8.s8 k32 / hfma2");
    r.tax[kImmaU8S8K32][kHmax2] = measured(0.381, "tensor_tax", "imma u8.s8 k32 / hmax2");
    r.tax[kImmaU8S8K32][kAddXor] =
        measured(1.593, "tensor_tax", "imma u8.s8 k32 / add + xor (iadd3 or imad.iadd, lop3)");
    r.tax[kImmaU8S8K32][kImad] = measured(0.123, "tensor_tax", "imma u8.s8 k32 / imad");
    r.tax[kImmaU8S8K32][kLop3] = measured(0.150, "tensor_tax", "imma u8.s8 k32 / lop3");
    r.tax[kImmaU8S8K32][kPrmt] = measured(0.150, "tensor_tax", "imma u8.s8 k32 / prmt");
    r.tax[kImmaU8S8K32][kFmnmx] = measured(0.150, "tensor_tax", "imma u8.s8 k32 / fmnmx");
    r.tax[kImmaU8S8K32][kLds] = measured(0.967, "tensor_tax", "imma u8.s8 k32 / lds");
    r.tax[kImmaU8S8K32][kCvtF32F16x2] =
        measured(0.150, "tensor_tax", "imma u8.s8 k32 / f32->f16x2");
    r.tax[kImmaU8S8K32][kCvtU32F32] = measured(0.168, "tensor_tax", "imma u8.s8 k32 / u32->f32");
    r.tax[kImmaU8S8K32][kCvtF16F32] = measured(0.104, "tensor_tax", "imma u8.s8 k32 / f16->f32");
  }

  if (cc == 800) {
    r.known = true;
    r.schedulers = documented(
        4, "CUDA C++ Programming Guide, compute capability 8.x: four warp schedulers an SM");
    r.issue = documented(
        1,
        "CUDA C++ Programming Guide, compute capability 8.x: a scheduler issues one instruction a "
        "clock");
    r.wavefront =
        documented(1, "CUDA C++ Programming Guide, shared memory: 32 banks of 32 bits a clock");
    r.atomic = announced(2.003, "sm_86's measured value, unmeasured on sm_80");
    r.atomic_wavefront = announced(1.000, "sm_86's measured value, unmeasured on sm_80");

    r.mma[kHmmaF16F32] =
        documented(8, "A100 whitepaper: 312 dense FP16 tensor TFLOPS, 108 SMs, 1410 MHz");
    r.mma[kHmmaF16F16] =
        documented(8, "A100 whitepaper: 312 dense FP16 tensor TFLOPS, 108 SMs, 1410 MHz");
    r.mma[kHmmaBf16F32] =
        documented(8, "A100 whitepaper: 312 dense BF16 tensor TFLOPS, 108 SMs, 1410 MHz");
    r.mma[kImmaU8S8K32] =
        documented(8, "A100 whitepaper: 624 dense INT8 tensor TOPS, 108 SMs, 1410 MHz");
    taxes_announced_from(r, 860);
  }
  return r;
}

//: a rate in 1/256 cycles, the unit the cost works in
constexpr long long quantum(double cycles) { return static_cast<long long>(cycles * 256.0 + 0.5); }

//: AN ARCHITECTURE'S RATES IN QUANTA, the strings left behind, for code that prices at run time
struct Quanta {
  long long issue, wavefront, atomic, atomic_wavefront;
  long long mma[kMmaKinds];
  long long tax[kMmaKinds][kOps];
};

constexpr Quanta quanta_of(const Rates& r) {
  Quanta q{};
  q.issue = quantum(r.issue.cycles);
  q.wavefront = quantum(r.wavefront.cycles);
  q.atomic = quantum(r.atomic.cycles);
  q.atomic_wavefront = quantum(r.atomic_wavefront.cycles);
  for (int k = 0; k < kMmaKinds; ++k) {
    q.mma[k] = quantum(r.mma[k].cycles);
    for (int c = 0; c < kOps; ++c) q.tax[k][c] = quantum(r.tax[k][c].cycles);
  }
  return q;
}

//: A FORM'S LOAD: on its busiest scheduler, the MMAs by kind, the taxed instructions by class and the rest; on its
//: SM, the shared-memory load wavefronts and the atomic instructions with their wavefronts
struct SchedulerLoad {
  long long mma[kMmaKinds];
  long long op[kOps];
  long long other;
};

struct SmLoad {
  long long wavefronts, atomics, atomic_wavefronts;
};

//: A FORM'S COST in quanta: the max of tensor (the MMAs plus the taxes of the instructions beside them, at the kind
//: with the most MMA cycles), issue and shared memory. see docs/internals/common/arch_rates.md#cost
constexpr long long cost(const Quanta& q, const SchedulerLoad& s, const SmLoad& m) {
  long long tensor = 0, instructions = s.other, top = -1;
  int kind = 0;
  for (int k = 0; k < kMmaKinds; ++k) {
    const long long t = s.mma[k] * q.mma[k];
    tensor += t;
    instructions += s.mma[k];
    if (t > top) {
      top = t;
      kind = k;
    }
  }
  for (int c = 0; c < kOps; ++c) {
    tensor += s.op[c] * q.tax[kind][c];
    instructions += s.op[c];
  }
  const long long atomics = m.atomics * q.atomic > m.atomic_wavefronts * q.atomic_wavefront
                                ? m.atomics * q.atomic
                                : m.atomic_wavefronts * q.atomic_wavefront;
  const long long smem = m.wavefronts * q.wavefront + atomics;
  const long long issue = instructions * q.issue;
  return tensor > issue ? (tensor > smem ? tensor : smem) : (issue > smem ? issue : smem);
}

//: THE PARITY CASE `measure/arch_rates.py` asserts too: one load priced on sm_86
static_assert(cost(quanta_of(rates_of(860)),
                   SchedulerLoad{{16, 0, 0, 0}, {8, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0}, 40},
                   SmLoad{64, 16, 24})
              == 133204);

}  // namespace arch
}  // namespace denseref

// THE EXTENSION'S OPERATOR REGISTRATION -- the only TU that registers with torch (host-only, so a `.cpp` rather than
// recompiled per `-gencode` in a `.cu`). Through torch's STABLE ABI: every entry is an operator in the `fsla` namespace
// (`torch.ops.fsla.<entry>`), declared by schema and implemented by a boxed function, so the binary depends on torch's
// stable C shim and never on its C++ ABI. The module object `fsla_cu13._C` is only the handle an import loads the library by.
// Full entry inventory, the schemas and the VMM owner's handle protocol: see docs/internals/fsla_api.md

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/build_stamp.cuh"
#include "common/select.cuh"
#include "common/torch_seam.cuh"
#include "plan/prefill_launch.h"
#include "src/common/geom_api.cuh"
#include "src/common/plan/plan_api.cuh"
#include "src/decode/decode_api.cuh"
#include "src/entmax/entmax.cuh"
#include "src/entmax/factor.cuh"
#include "src/mask_prepare/mask_prepare.cuh"

using fsla::check_arch_table;
using fsla::Tensor;

namespace {

//: A DEVICE'S CARD (`common/select.cuh`): its compute capability, its family's name (`kFamilyNames`, generated from
//: the one list the `Family` enum is) and W, the window the planner's matcher derives from the device's facts (`plan::prefill_window`)
std::tuple<int64_t, std::string, int64_t> device_card_op(int64_t device) {
  const fsla::Card& c = fsla::card_of(static_cast<int>(device));
  fsla::plan::PrefillDescriptor facts;
  facts.assign("cc", fsla::plan::Interval{c.cc, c.cc});
  return {c.cc, denseref::arch::kFamilyNames[static_cast<int>(c.traits.family)],
          fsla::plan::prefill_window(facts)};
}

}  // namespace

STABLE_TORCH_LIBRARY(fsla, m) {
  // ---- the device's card and the stamps -------------------
  m.def("device_card(int device) -> (int, str, int)");
  m.def("csrc_build_stamp() -> int");
  m.def("sm_clock_ghz(int spin_cycles=20000000) -> float");

  // ---- the addressing block ------------------------------------------------
  m.def(
      "carry_geometry(int[] widths, int level_modes, int bc, int nsr, int nsw, int dv, int interleave, int atom_floor) -> int[]");
  m.def("carry_sub_boxes(int depth, int box_leaves, int workers, int atom_floor) -> int[][]");

  // ---- the producer's solves -----------------------------------------------
  m.def(
      "entmax_factor_forward(Tensor logits, Tensor(a!) values, Tensor(b!)? values_second, "
      "Tensor(c!) support_words, int[] logit_offsets, int[] value_offsets, int[] support_offsets, int[] widths, "
      "int heads, float alpha) -> ()");
  m.def(
      "entmax_factor_backward(Tensor values, Tensor support_words, Tensor d_values, "
      "Tensor? d_values_second, Tensor(a!) d_logits, int[] logit_offsets, int[] value_offsets, "
      "int[] support_offsets, int[] widths, int heads, float alpha) -> ()");
  m.def(
      "routing_softmax_forward(Tensor logits, Tensor(a!) values, Tensor(b!)? values_second, int[] logit_offsets, "
      "int[] value_offsets, int[] widths, int heads) -> ()");
  m.def(
      "routing_softmax_backward(Tensor values, Tensor d_values, Tensor? d_values_second, Tensor(a!) d_logits, "
      "int[] logit_offsets, int[] value_offsets, int[] widths, int heads) -> ()");

  // ---- the prefill's mask-preparation pass ---------------------------------
  m.def(
      "mask_prepare(Tensor[] masks, int[] slots, int[] widths, Tensor density, "
      "Tensor(a!) workspace) -> ()");
  m.def("mask_prepare_bytes(int[] width_bits, int[] heads, int words, int device) -> int");

  m.def("entmax_max_levels() -> int");
  m.def("entmax_width_class(int width) -> int");
  m.def("fsla_factor_forward_arms() -> int[][]");
  m.def("fsla_factor_backward_arms() -> int[][]");
  m.def("fsla_softmax_forward_arms() -> int[][]");
  m.def("fsla_softmax_backward_arms() -> int[][]");

  // ---- the T=1 decode path -------------------------------------------------
  m.def(
      "fsla_decode_forward(Tensor[] read, Tensor[] write, Tensor[] read_scale, Tensor[] write_scale, "
      "Tensor[] read_mask, Tensor[] write_mask, Tensor? dials, Tensor v, Tensor v_scale, "
      "Tensor(a!) state, Tensor(b!) yacc, Tensor(c!) queue, int ctas, float eps, int[] descriptor) -> "
      "Tensor");
  m.def("fsla_call_facts(str kernel) -> str");
  m.def("fsla_descriptor(str kernel, int device, int[] call) -> int[]");
  m.def("fsla_owed_kernels(str kernel, int device, int[] call) -> str");
  m.def("fsla_decode_build_stamp() -> int");
  m.def("fsla_decode_launch(int[] descriptor, int device) -> int[]");
  m.def("fsla_decode_lattice(int[] widths) -> int[]");
  m.def("fsla_decode_lattice_box(int[] widths, int k, int m) -> int[]");
  m.def("fsla_decode_arms() -> int[][]");
  m.def("fsla_decode_ledger_bind(Tensor? ledger) -> ()");
  m.def("fsla_decode_ledger_warps() -> int");
  m.def("fsla_decode_capacity() -> int");
  m.def("fsla_decode_producer_width_mirror() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(fsla, CompositeExplicitAutograd, m) {
  m.impl("device_card", TORCH_BOX(&device_card_op));
  m.impl("fsla_call_facts", TORCH_BOX(&fsla::plan::fsla_call_facts));
  m.impl("fsla_descriptor", TORCH_BOX(&fsla::plan::fsla_descriptor));
  m.impl("fsla_owed_kernels", TORCH_BOX(&fsla::plan::fsla_owed_kernels));
  m.impl("fsla_decode_launch", TORCH_BOX(&fsla::decode::fsla_decode_launch));
  m.impl("fsla_decode_lattice", TORCH_BOX(&fsla::decode::fsla_decode_lattice));
  m.impl("fsla_decode_lattice_box", TORCH_BOX(&fsla::decode::fsla_decode_lattice_box));

  m.impl("csrc_build_stamp", TORCH_BOX(&fsla::csrc_build_stamp));
  m.impl("sm_clock_ghz", TORCH_BOX(&fsla::sm_clock_ghz));
  m.impl("carry_geometry", TORCH_BOX(&fsla::carry::geometry));
  m.impl("carry_sub_boxes", TORCH_BOX(&fsla::carry::sub_box_set));
  m.impl("entmax_factor_forward", TORCH_BOX(&fsla::entmax::factor_forward));
  m.impl("entmax_factor_backward", TORCH_BOX(&fsla::entmax::factor_backward));
  m.impl("routing_softmax_forward", TORCH_BOX(&fsla::entmax::softmax_forward));
  m.impl("routing_softmax_backward", TORCH_BOX(&fsla::entmax::softmax_backward));
  m.impl("mask_prepare", TORCH_BOX(&fsla::mask_prepare::mask_prepare));
  m.impl("mask_prepare_bytes", TORCH_BOX(&fsla::mask_prepare::mask_prepare_bytes));
  m.impl("entmax_max_levels", TORCH_BOX(&fsla::entmax::max_levels));
  m.impl("entmax_width_class", TORCH_BOX(&fsla::entmax::width_class));
  m.impl("fsla_factor_forward_arms", TORCH_BOX(&fsla::entmax::factor_forward_arms));
  m.impl("fsla_factor_backward_arms", TORCH_BOX(&fsla::entmax::factor_backward_arms));
  m.impl("fsla_softmax_forward_arms", TORCH_BOX(&fsla::entmax::softmax_forward_arms));
  m.impl("fsla_softmax_backward_arms", TORCH_BOX(&fsla::entmax::softmax_backward_arms));
  m.impl("fsla_decode_forward", TORCH_BOX(&fsla::decode::fsla_decode_forward));
  m.impl("fsla_decode_ledger_bind", TORCH_BOX(&fsla::decode::fsla_decode_ledger_bind));
  m.impl("fsla_decode_ledger_warps", TORCH_BOX(&fsla::decode::fsla_decode_ledger_warps));
  m.impl("fsla_decode_build_stamp", TORCH_BOX(&fsla::decode::fsla_decode_build_stamp));
  m.impl("fsla_decode_arms", TORCH_BOX(&fsla::decode::fsla_decode_arms));
  m.impl("fsla_decode_capacity", TORCH_BOX(&fsla::decode::fsla_decode_capacity));
  m.impl("fsla_decode_producer_width_mirror",
         TORCH_BOX(&fsla::decode::fsla_decode_producer_width_mirror));
}

// THE MODULE OBJECT: importing `fsla_cu13._C` loads this library, whose static registrations above run on load. The module
// itself carries nothing, so it is built against Python's limited API and one binary serves every CPython from the
// limited API's floor (3.10, torch's build flag) on.
extern "C" PyObject* PyInit__C(void) {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT,
                               "_C",
                               "fsla's operators, registered as torch.ops.fsla",
                               -1,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
  return PyModule_Create(&module);
}

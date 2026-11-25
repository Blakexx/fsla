// THE EXTENSION'S OPERATOR REGISTRATION -- the only TU that registers with torch (host-only, so a `.cpp` rather than
// recompiled per `-gencode` in a `.cu`). Through torch's STABLE ABI: every entry is an operator in the `fsla` namespace
// (`torch.ops.fsla.<entry>`), declared by schema and implemented by a boxed function, so the binary depends on torch's
// stable C shim and never on its C++ ABI. The module object `fsla_cu13._C` is only the handle an import loads the library by.
// Full entry inventory, the schemas and the VMM owner's handle protocol: see docs/internals/fsla_api.md

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
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
#include "src/paging/vmm_owner.cuh"

using fsla::check_arch_table;
using fsla::Tensor;

namespace {

// THE VMM OWNERS, by handle: an operator cannot return a native object, so the owner lives here and the caller holds
// an integer. A tensor view of an owner keeps the owner alive past its release (VmmOwner::base).
std::mutex g_owners_mutex;
std::unordered_map<int64_t, std::shared_ptr<fsla::paging::VmmOwner>> g_owners;
int64_t g_next_owner = 1;

std::shared_ptr<fsla::paging::VmmOwner> owner_of(int64_t handle) {
  const std::lock_guard<std::mutex> lock(g_owners_mutex);
  const auto it = g_owners.find(handle);
  STD_TORCH_CHECK(it != g_owners.end(), "no FSLA VMM owner holds handle ", handle,
                  " (released, or never created)");
  return it->second;
}

int64_t vmm_create(int64_t device, int64_t dense_limit_pages, int64_t page_rows, int64_t page_cols,
                   int64_t target_chunk_bytes) {
  auto owner = fsla::paging::create_vmm_owner((int)device, dense_limit_pages, page_rows, page_cols,
                                              target_chunk_bytes);
  const std::lock_guard<std::mutex> lock(g_owners_mutex);
  g_owners.emplace(g_next_owner, std::move(owner));
  return g_next_owner++;
}

void vmm_release(int64_t handle) {
  const std::lock_guard<std::mutex> lock(g_owners_mutex);
  g_owners.erase(handle);
}

Tensor vmm_base(int64_t handle) { return owner_of(handle)->base(); }

void vmm_grow(int64_t handle, int64_t required_pages) { owner_of(handle)->grow(required_pages); }

void vmm_rollback_to(int64_t handle, int64_t mapped_capacity_pages) {
  owner_of(handle)->rollback_to(mapped_capacity_pages);
}

void vmm_reset(int64_t handle) { owner_of(handle)->reset(); }

void vmm_close(int64_t handle) { owner_of(handle)->close(); }

//: closed, mapped_capacity_pages, dense_limit_pages, committed_bytes, virtual_bytes, allocation_granularity, chunk_pages
std::vector<int64_t> vmm_facts(int64_t handle) {
  const auto owner = owner_of(handle);
  return {(int64_t)owner->closed(),
          owner->mapped_capacity_pages(),
          owner->dense_limit_pages(),
          (int64_t)owner->committed_bytes(),
          (int64_t)owner->virtual_bytes(),
          (int64_t)owner->allocation_granularity(),
          owner->chunk_pages()};
}

std::tuple<int64_t, bool, int64_t, std::string> vmm_probe(int64_t device) {
  const auto probe = fsla::paging::VmmOwner::probe((int)device);
  return {probe.device, probe.supported, (int64_t)probe.allocation_granularity, probe.reason};
}

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
      "Tensor(a!) state, Tensor(b!) yacc, Tensor(c!) queue, Tensor(d!) growth, Tensor(e!) growth_any, "
      "Tensor(f!) growth_ctr, Tensor(g!) done, Tensor(h!) atom_bits, "
      "int ctas, float eps, int[] descriptor, Tensor(i!)? page_table=None, "
      "Tensor? pool_slots=None, Tensor(j!)? pool_cursor=None) -> Tensor");
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

  // ---- the VMM owner, by handle --------------------------------------------
  m.def("vmm_probe(int device) -> (int, bool, int, str)");
  m.def(
      "vmm_create(int device, int dense_limit_pages, int page_rows, int page_cols, "
      "int target_chunk_bytes=67108864) -> int");
  m.def("vmm_release(int handle) -> ()");
  m.def("vmm_base(int handle) -> Tensor");
  m.def("vmm_grow(int handle, int required_pages) -> ()");
  m.def("vmm_rollback_to(int handle, int mapped_capacity_pages) -> ()");
  m.def("vmm_reset(int handle) -> ()");
  m.def("vmm_close(int handle) -> ()");
  m.def("vmm_facts(int handle) -> int[]");
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
  m.impl("vmm_probe", TORCH_BOX(&vmm_probe));
  m.impl("vmm_create", TORCH_BOX(&vmm_create));
  m.impl("vmm_release", TORCH_BOX(&vmm_release));
  m.impl("vmm_base", TORCH_BOX(&vmm_base));
  m.impl("vmm_grow", TORCH_BOX(&vmm_grow));
  m.impl("vmm_rollback_to", TORCH_BOX(&vmm_rollback_to));
  m.impl("vmm_reset", TORCH_BOX(&vmm_reset));
  m.impl("vmm_close", TORCH_BOX(&vmm_close));
  m.impl("vmm_facts", TORCH_BOX(&vmm_facts));
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

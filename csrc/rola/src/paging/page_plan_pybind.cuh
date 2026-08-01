#pragma once

#include "p3_page_plan.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace fla_rola::p3 {

namespace py = pybind11;

std::vector<torch::Tensor> rola_p3_page_plan_prepare_cuda(
    std::vector<torch::Tensor> write_amps, std::vector<int64_t> level_widths,
    std::vector<int64_t> level_pitches, std::vector<int64_t> level_write_can_zero,
    std::vector<int64_t> owner_stride, int64_t token_count, int64_t token_tile,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices,
    std::vector<torch::Tensor> plan_tensors, std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base, int64_t page_states,
    int64_t pages_per_bh, int64_t max_suffix_combinations,
    int64_t generation, int64_t epoch, bool identity_physical_ids);

void rola_p3_page_admit_mask_cuda(torch::Tensor page_table, torch::Tensor admit);

std::vector<torch::Tensor> rola_p3_page_plan_finalize_cuda(
    std::vector<torch::Tensor> plan_tensors, std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base, int64_t generation, int64_t epoch,
    int64_t page_states, int64_t dqk, int64_t dv, bool identity_physical_ids);

std::vector<torch::Tensor> rola_p3_page_plan_publish_cuda(
    std::vector<torch::Tensor> plan_tensors, std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base, int64_t generation, int64_t epoch,
    int64_t page_states, int64_t dqk, int64_t dv, bool identity_physical_ids);

void rola_p3_page_cache_reset_cuda(
    std::vector<torch::Tensor> cache_tensors);

void launch_paged_state_readiness_assert(
    torch::Tensor page_table, torch::Tensor earliest_epoch_tile,
    torch::Tensor generation_tensor,
    int64_t physical_pages, int64_t generation, int64_t plan_epoch,
    cudaStream_t stream);

// The parent extension owns construction of the tensor-backed view structs and
// calls the typed launch functions. Registration exposes only ABI identity and
// status constants here, avoiding a second Python-side page allocator.
inline void register_rola_page_plan(py::module_& module) {
  //: Bumped for the §7.1 re-key: `page_states := BC`, `dqk := 1`, the V3 §4
  //: amplitude stream in place of V2's RouteDescriptor + PrefixRunSchedule, and
  //: the `CAPPED` status with its admit mask.
  module.def("p3_page_plan_abi_version", []() { return 3; });
  module.def("p3_page_plan_status_codes", []() {
    py::dict result;
    result["READY"] = kPagePlanReady;
    result["NEED_GROWTH"] = kPagePlanNeedGrowth;
    result["OVERFLOW"] = kPagePlanOverflow;
    result["CAPPED"] = kPagePlanCapped;
    return result;
  });
  module.def("p3_page_plan_workspace_bytes", &page_plan_workspace_bytes);
  module.def("rola_p3_page_plan_prepare", &rola_p3_page_plan_prepare_cuda,
            py::arg("write_amps"), py::arg("level_widths"),
            py::arg("level_pitches"), py::arg("level_write_can_zero"),
            py::arg("owner_stride"), py::arg("token_count"), py::arg("token_tile"),
            py::arg("first_spanned_level"), py::arg("slice_digits"),
            py::arg("n_slices"), py::arg("plan_tensors"),
            py::arg("cache_tensors"), py::arg("mass_base"), py::arg("page_states"),
            py::arg("pages_per_bh"), py::arg("max_suffix_combinations"),
            py::arg("generation"), py::arg("epoch"),
            py::arg("identity_physical_ids"));
  module.def("rola_p3_page_admit_mask", &rola_p3_page_admit_mask_cuda,
             py::arg("page_table"), py::arg("admit"));
  module.def("rola_p3_page_plan_finalize", &rola_p3_page_plan_finalize_cuda,
            py::arg("plan_tensors"), py::arg("cache_tensors"),
            py::arg("mass_base"), py::arg("generation"), py::arg("epoch"),
            py::arg("page_states"), py::arg("dqk"), py::arg("dv"),
            py::arg("identity_physical_ids"));
  module.def("rola_p3_page_plan_publish", &rola_p3_page_plan_publish_cuda,
            py::arg("plan_tensors"), py::arg("cache_tensors"),
            py::arg("mass_base"), py::arg("generation"), py::arg("epoch"),
            py::arg("page_states"), py::arg("dqk"), py::arg("dv"),
            py::arg("identity_physical_ids"));
  module.def("rola_p3_page_cache_reset", &rola_p3_page_cache_reset_cuda,
             py::arg("cache_tensors"));
}

}  // namespace fla_rola::p3

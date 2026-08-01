// THE EXTENSION'S PYTHON BINDING -- and the only translation unit that knows
// pybind exists.
//
// **WHY IT IS A `.cpp` AND NOT PART OF `consumer.cu`.** Binding code is host-only.
// Left inside a `.cu` it is recompiled once per `-gencode` target for no benefit,
// and every future architecture added to the ratified set multiplies that cost.
// `flash_api.cpp` is the convention; this is the same split.
//
// **ONE MODULE, NOT TWO.** The consumer and the §7.1 paging arena register into the
// same extension: the consumer reads the page table the plan publishes, so splitting
// them would put a cross-module ABI between two halves of one contract. The two
// JIT-era `*_pybind.cuh` fragment headers that used to do this registration are
// gone -- their `m.def` bodies are here, and their declarations are in
// `src/host_api.cuh`.
//
// The module name is supplied by the build (`TORCH_EXTENSION_NAME`), which
// `setup.py` sets to `rola_cuda`. `rola/ops/_ext.py` imports exactly that name and
// never builds it.

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "src/host_api.cuh"
#include "src/paging/page_plan.cuh"
#include "src/paging/vmm_owner.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ---- the consumer -------------------------------------------------------
  m.def("rola_consumer_forward", &rola::rola_consumer_forward,
        "RoLA consumer forward (spec §3/§5)");
  m.def("rola_consumer_occupancy", &rola::rola_consumer_occupancy,
        "the driver's CTAs/SM for the consumer kernel at its real SMEM size");
  m.def("rola_consumer_check_arch_table", &rola::rola_consumer_check_arch_table,
        "validate the consumer's compile-time SM-resource table against the driver "
        "(the launch bounds in this binary were derived from it)");
  m.def("rola_consumer_smem_bytes", &rola::rola_consumer_smem_bytes,
        "the consumer kernel's dynamic shared-memory footprint per CTA");
  m.def("rola_sm_shared_bytes", &rola::rola_sm_shared_bytes,
        "cudaDevAttrMaxSharedMemoryPerMultiprocessor -- the per-SM SMEM budget the "
        "plan records (§10 ruling)");

  // ---- spec §7.1 page commitment ------------------------------------------
  //
  // Registration exposes ABI identity and the status constants only; the parent
  // extension owns construction of the tensor-backed view structs and calls the
  // typed launch functions, so there is no second Python-side page allocator.
  //
  //: ABI 3 is the §7.1 re-key: `page_states := BC`, `dqk := 1`, the §4 amplitude
  //: stream in place of V2's RouteDescriptor + PrefixRunSchedule, and the `CAPPED`
  //: status with its admit mask.
  m.def("page_plan_abi_version", []() { return 3; });
  m.def("page_plan_status_codes", []() {
    py::dict result;
    result["READY"] = rola::paging::kPagePlanReady;
    result["NEED_GROWTH"] = rola::paging::kPagePlanNeedGrowth;
    result["OVERFLOW"] = rola::paging::kPagePlanOverflow;
    result["CAPPED"] = rola::paging::kPagePlanCapped;
    return result;
  });
  m.def("page_plan_workspace_bytes", &rola::paging::page_plan_workspace_bytes);
  m.def("rola_page_plan_prepare", &rola::paging::rola_page_plan_prepare_cuda,
        py::arg("write_amps"), py::arg("level_widths"),
        py::arg("level_pitches"), py::arg("level_write_can_zero"),
        py::arg("owner_stride"), py::arg("token_count"), py::arg("token_tile"),
        py::arg("first_spanned_level"), py::arg("slice_digits"),
        py::arg("n_slices"), py::arg("plan_tensors"),
        py::arg("cache_tensors"), py::arg("mass_base"), py::arg("page_states"),
        py::arg("pages_per_bh"), py::arg("max_suffix_combinations"),
        py::arg("generation"), py::arg("epoch"),
        py::arg("identity_physical_ids"));
  m.def("rola_page_admit_mask", &rola::paging::rola_page_admit_mask_cuda,
        py::arg("page_table"), py::arg("admit"));
  m.def("rola_page_plan_finalize", &rola::paging::rola_page_plan_finalize_cuda,
        py::arg("plan_tensors"), py::arg("cache_tensors"),
        py::arg("mass_base"), py::arg("generation"), py::arg("epoch"),
        py::arg("page_states"), py::arg("dqk"), py::arg("dv"),
        py::arg("identity_physical_ids"));
  m.def("rola_page_plan_publish", &rola::paging::rola_page_plan_publish_cuda,
        py::arg("plan_tensors"), py::arg("cache_tensors"),
        py::arg("mass_base"), py::arg("generation"), py::arg("epoch"),
        py::arg("page_states"), py::arg("dqk"), py::arg("dv"),
        py::arg("identity_physical_ids"));
  m.def("rola_page_cache_reset", &rola::paging::rola_page_cache_reset_cuda,
        py::arg("cache_tensors"));

  // ---- the VMM owner ------------------------------------------------------
  m.def("rola_vmm_probe", [](int device) {
    const auto probe = rola::paging::VmmOwner::probe(device);
    py::dict result;
    result["device"] = probe.device;
    result["supported"] = probe.supported;
    result["allocation_granularity"] = probe.allocation_granularity;
    result["reason"] = probe.reason;
    return result;
  }, py::arg("device"));

  py::class_<rola::paging::VmmOwner, std::shared_ptr<rola::paging::VmmOwner>>(
      m, "RoLAVmmOwner")
      .def("value_base", &rola::paging::VmmOwner::value_base)
      .def("mass_base", &rola::paging::VmmOwner::mass_base)
      .def("grow", &rola::paging::VmmOwner::grow, py::arg("required_pages"))
      .def("rollback_to", &rola::paging::VmmOwner::rollback_to,
           py::arg("mapped_capacity_pages"))
      .def("reset", &rola::paging::VmmOwner::reset)
      .def("close", &rola::paging::VmmOwner::close)
      .def_property_readonly("closed", &rola::paging::VmmOwner::closed)
      .def_property_readonly("mapped_capacity_pages",
                             &rola::paging::VmmOwner::mapped_capacity_pages)
      .def_property_readonly("max_physical_pages",
                             &rola::paging::VmmOwner::max_physical_pages)
      .def_property_readonly("committed_bytes",
                             &rola::paging::VmmOwner::committed_bytes)
      .def_property_readonly("virtual_bytes", &rola::paging::VmmOwner::virtual_bytes)
      .def_property_readonly("allocation_granularity",
                             &rola::paging::VmmOwner::allocation_granularity)
      .def_property_readonly("chunk_pages", &rola::paging::VmmOwner::chunk_pages)
      .def_property_readonly("backing_kind", &rola::paging::VmmOwner::backing_kind);

  m.def("rola_vmm_create", &rola::paging::create_vmm_owner,
        py::arg("device"), py::arg("max_physical_pages"),
        py::arg("bp"), py::arg("dqk"), py::arg("dv"), py::arg("global_mass"),
        py::arg("target_chunk_bytes") = 64ll * 1024ll * 1024ll);
}

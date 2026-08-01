#pragma once

#include "p3_vmm_owner.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fla_rola::p3 {

// The parent extension owns module construction. It should include this header,
// add p3_vmm_owner.cu to its source list, and call this helper exactly once from
// its existing PYBIND11_MODULE body. Keeping registration separate avoids a
// second extension and keeps the VMM owner behind the existing CUDA ABI.
inline void register_rola_vmm(pybind11::module_& module) {
  module.def("rola_vmm_probe", [](int device) {
    const auto probe = VmmOwner::probe(device);
    pybind11::dict result;
    result["device"] = probe.device;
    result["supported"] = probe.supported;
    result["allocation_granularity"] = probe.allocation_granularity;
    result["reason"] = probe.reason;
    return result;
  }, pybind11::arg("device"));

  pybind11::class_<VmmOwner, std::shared_ptr<VmmOwner>>(module, "RoLAVmmOwner")
      .def("value_base", &VmmOwner::value_base)
      .def("mass_base", &VmmOwner::mass_base)
      .def("grow", &VmmOwner::grow, pybind11::arg("required_pages"))
      .def("rollback_to", &VmmOwner::rollback_to,
           pybind11::arg("mapped_capacity_pages"))
      .def("reset", &VmmOwner::reset)
      .def("close", &VmmOwner::close)
      .def_property_readonly("closed", &VmmOwner::closed)
      .def_property_readonly("mapped_capacity_pages", &VmmOwner::mapped_capacity_pages)
      .def_property_readonly("max_physical_pages", &VmmOwner::max_physical_pages)
      .def_property_readonly("committed_bytes", &VmmOwner::committed_bytes)
      .def_property_readonly("virtual_bytes", &VmmOwner::virtual_bytes)
      .def_property_readonly("allocation_granularity", &VmmOwner::allocation_granularity)
      .def_property_readonly("chunk_pages", &VmmOwner::chunk_pages)
      .def_property_readonly("backing_kind", &VmmOwner::backing_kind);

  module.def(
      "rola_vmm_create", &create_vmm_owner,
      pybind11::arg("device"), pybind11::arg("max_physical_pages"),
      pybind11::arg("bp"), pybind11::arg("dqk"), pybind11::arg("dv"),
      pybind11::arg("global_mass"),
      pybind11::arg("target_chunk_bytes") = 64ll * 1024ll * 1024ll);
}

}  // namespace fla_rola::p3

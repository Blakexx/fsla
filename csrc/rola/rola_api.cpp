// THE EXTENSION'S PYTHON BINDING -- and the only translation unit that knows
// pybind exists.
//
// Binding code is host-only, so it lives in a `.cpp`: inside a `.cu` it would be
// recompiled once per `-gencode` target for no benefit. `flash_api.cpp` is the
// convention this follows.
//
// ONE MODULE, NOT TWO. The consumer, the decode sibling and the §7.1 paging arena all
// register here, because they share one contract and a split would put a cross-module
// ABI through the middle of it.
//
// The module name is supplied by the build (`TORCH_EXTENSION_NAME`). NOTHING in this
// file or under `src/` spells it, so the packaging layout can move without touching a
// kernel symbol.
//
// The full entry inventory and the summary pass's no-allocation contract:
// docs/internals/rola_api.md

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "src/decode_api.cuh"
#include "src/entmax.cuh"
#include "src/host_api.cuh"
#include "src/summaries.cuh"
#include "src/paging/vmm_owner.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ---- the producer's union entmax solve ----------------------------------
  m.def("entmax_union_forward", &rola::entmax::union_forward,
        "union entmax forward: midpoint values, read/write routes, packed support");
  m.def("entmax_union_backward", &rola::entmax::union_backward,
        "union entmax VJP into read/write logits");

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
  m.def("rola_dispatch_table", &rola::rola_dispatch_table,
        "every row of the consumer's dispatch table: (BC, d_v, D, decay, "
        "group_tokens, smem_bytes, shard, src_key). Touches no device.");
  m.def("rola_max_dynamic_smem_bytes", &rola::rola_max_dynamic_smem_bytes,
        "cudaDevAttrMaxSharedMemoryPerBlockOptin -- the device fact a row's footprint "
        "is admitted against");
  m.def("rola_sm_shared_bytes", &rola::rola_sm_shared_bytes,
        "cudaDevAttrMaxSharedMemoryPerMultiprocessor -- the per-SM SMEM budget the "
        "plan records (§10 ruling)");

  // ---- the T=1 decode path (thunder §6b) ----------------------------------
  //: A SIBLING of the consumer, not a specialization of it: shared produce contract,
  //: own consumer, frozen config. Two of its three entries are capacity GATES rather
  //: than operations -- docs/internals/rola_api.md#registered-entries
  m.def("rola_decode_forward", &rola::decode::rola_decode_forward,
        "RoLA T=1 decode: bitmap support expansion + token-stationary gather-GEMV",
        py::arg("amp_read"), py::arg("amp_write"), py::arg("amp_clock"), py::arg("dials"),
        py::arg("g_write"), py::arg("v"), py::arg("state"), py::arg("bits"), py::arg("pc_coarse"),
        py::arg("ws"), py::arg("ctr"), py::arg("level_widths"), py::arg("level_strides"),
        py::arg("level_spans"), py::arg("level_row_offsets"), py::arg("heads"), py::arg("n_split"),
        py::arg("eps"));
  m.def("rola_decode_capacity", &rola::decode::rola_decode_capacity,
        "the decode path's maximum level width; must be >= the producer's "
        "MAX_BRANCH_WIDTH (tests/unit/test_decode_capacity.py pins the pair)");
  m.def("rola_decode_producer_width_mirror", &rola::decode::rola_decode_producer_width_mirror,
        "the C++ mirror of the producer's MAX_BRANCH_WIDTH; pinned EQUAL to the Python "
        "constant so a stale mirror cannot silently satisfy the capacity static_assert");

  // ---- spec §4.1/§4.2 producer-epilogue summaries ---------------------------
  //: THE PRODUCER-EPILOGUE ROUTING SUMMARY PASS (THUNDER_RUN_SPEC.md §4.1/§4.2).
  //: One launch, no allocation, no host synchronization: every output is a caller-owned
  //: tensor the Python side zeroes beforehand and reads back in its ONE D2H transfer,
  //: which is why nothing here returns -- docs/internals/rola_api.md#summaries-contract
  m.def("rola_routing_summaries", &rola::summaries::rola_routing_summaries_cuda,
        "Producer-epilogue routing summaries: occupancy, the quantum histogram, the "
        "candidate count, and §4.1's canonical per-leaf write bitmap with its owner "
        "condensation");

  // ---- the VMM owner ------------------------------------------------------
  m.def(
      "rola_vmm_probe",
      [](int device) {
        const auto probe = rola::paging::VmmOwner::probe(device);
        py::dict result;
        result["device"] = probe.device;
        result["supported"] = probe.supported;
        result["allocation_granularity"] = probe.allocation_granularity;
        result["reason"] = probe.reason;
        return result;
      },
      py::arg("device"));

  py::class_<rola::paging::VmmOwner, std::shared_ptr<rola::paging::VmmOwner>>(m, "RoLAVmmOwner")
      .def("value_base", &rola::paging::VmmOwner::value_base)
      .def("mass_base", &rola::paging::VmmOwner::mass_base)
      .def("grow", &rola::paging::VmmOwner::grow, py::arg("required_pages"))
      .def("rollback_to", &rola::paging::VmmOwner::rollback_to, py::arg("mapped_capacity_pages"))
      .def("reset", &rola::paging::VmmOwner::reset)
      .def("close", &rola::paging::VmmOwner::close)
      .def_property_readonly("closed", &rola::paging::VmmOwner::closed)
      .def_property_readonly("mapped_capacity_pages",
                             &rola::paging::VmmOwner::mapped_capacity_pages)
      .def_property_readonly("max_physical_pages", &rola::paging::VmmOwner::max_physical_pages)
      .def_property_readonly("committed_bytes", &rola::paging::VmmOwner::committed_bytes)
      .def_property_readonly("virtual_bytes", &rola::paging::VmmOwner::virtual_bytes)
      .def_property_readonly("allocation_granularity",
                             &rola::paging::VmmOwner::allocation_granularity)
      .def_property_readonly("chunk_pages", &rola::paging::VmmOwner::chunk_pages)
      .def_property_readonly("backing_kind", &rola::paging::VmmOwner::backing_kind);

  m.def("rola_vmm_create", &rola::paging::create_vmm_owner, py::arg("device"),
        py::arg("max_physical_pages"), py::arg("bp"), py::arg("dqk"), py::arg("dv"),
        py::arg("global_mass"), py::arg("target_chunk_bytes") = 64ll * 1024ll * 1024ll);
}

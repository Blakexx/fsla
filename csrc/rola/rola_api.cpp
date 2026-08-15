// THE EXTENSION'S PYTHON BINDING -- and the only translation unit that knows
// pybind exists.
//
// Binding code is host-only, so it lives in a `.cpp`: inside a `.cu` it would be
// recompiled once per `-gencode` target for no benefit. `flash_api.cpp` is the
// convention this follows.
//
// ONE MODULE, NOT TWO. The chunk consumer, the decode sibling and the paging
// arena all register here, because they share one contract and a split would put a
// cross-module ABI through the middle of it.
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

#include "src/decode/decode_api.cuh"
#include "src/entmax/entmax.cuh"
#include "src/paging/vmm_owner.cuh"
#include "src/chunk/chunk_api.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ---- the chunk consumer, the adopted operator ---------------------------
  m.def("chunk_union_table", &rola::chunk::union_table,
        "chunk consumer: the precomputed membership union table");
  //: the state plane is the CALLER'S, on both sides; absent on both is a fresh
  //: sequence whose carried state nobody asked for -- what a stateless forward
  //: passes, and the null the kernel's epilogue is skipped by.  A page table
  //: makes that plane PAGED (`[slots, 16, d_v+1]`); without one it is the dense
  //: `[BH, N/16, 16, d_v+1]` plane, where a slot IS its atom.
  m.def("chunk_forward_raw", &rola::chunk::forward_raw,
        "chunk consumer: the readout's d_v+1 columns, undivided", py::arg("pr"), py::arg("pw"),
        py::arg("gw"), py::arg("v"), py::arg("tbl"), py::arg("chunk"), py::arg("bc"),
        py::arg("widths"), py::arg("s_in") = py::none(), py::arg("s_out") = py::none(),
        py::arg("page_tbl") = py::none(), py::arg("block_bits") = py::none());
  m.def("chunk_forward", &rola::chunk::forward, "chunk consumer: table pass, kernel, divide pass",
        py::arg("pr"), py::arg("pw"), py::arg("gw"), py::arg("v"), py::arg("chunk"), py::arg("bc"),
        py::arg("widths"), py::arg("eps"), py::arg("read_mass") = py::none(),
        py::arg("out") = torch::kFloat32, py::arg("s_in") = py::none(),
        py::arg("s_out") = py::none(), py::arg("page_tbl") = py::none(),
        py::arg("block_bits") = py::none());
  m.def("chunk_stats", &rola::chunk::stats,
        "chunk consumer: the exact atom-grain write bitmap the page plan admits from");
  //: the sparse grid's predicate INPUT: three liveness facts per (bh, block),
  //: reduced from the union table this call already built. The predicate OVER
  //: them is the dispatch's, because it differs by call class and by backing.
  m.def("chunk_block_bits", &rola::chunk::block_bits,
        "chunk consumer: the block-grain read/write/state liveness bitmap", py::arg("tbl"),
        py::arg("bc"), py::arg("widths"), py::arg("L"), py::arg("page_tbl") = py::none());
  //: the read mass is the scale the read levels still carry; `eps * mass` is what
  //: keeps the floor on the ratio's own scale (docs/internals/chunk/chunk.md#readout).
  m.def("chunk_divide_pass", &rola::chunk::divide_pass,
        "chunk consumer: the ratio, one elementwise pass, in the operand's layout", py::arg("num"),
        py::arg("den"), py::arg("mass"), py::arg("eps"), py::arg("out") = torch::kFloat32);
  m.def("chunk_build_stamp", &rola::chunk::build_stamp,
        "chunk consumer: device-side build identity + the census of both state "
        "variants, stateless first");
  m.def("chunk_arms", &rola::chunk::arms,
        "chunk consumer: the built arms with derived plan and geometry");

  // ---- the producer's union entmax solve ----------------------------------
  m.def("entmax_union_forward", &rola::entmax::union_forward,
        "union entmax forward: midpoint values, read/write routes, packed support");
  m.def("entmax_union_backward", &rola::entmax::union_backward,
        "union entmax VJP into read/write logits");

  // ---- the T=1 decode path -------------------------------------------------
  //: A SIBLING of the chunk consumer, not a specialization of it: shared produce
  //: contract, own consumer, frozen config. Two of its three entries are capacity GATES rather
  //: than operations -- docs/internals/rola_api.md#registered-entries
  m.def("rola_decode_forward", &rola::decode::rola_decode_forward,
        "RoLA T=1 decode: bitmap support expansion + token-stationary gather-GEMV",
        py::arg("amp_read"), py::arg("amp_write"), py::arg("amp_clock"), py::arg("dials"),
        py::arg("g_write"), py::arg("v"), py::arg("state"), py::arg("bits"), py::arg("pc_coarse"),
        py::arg("ws"), py::arg("ctr"), py::arg("level_widths"), py::arg("level_strides"),
        py::arg("level_spans"), py::arg("level_row_offsets"), py::arg("heads"), py::arg("n_split"),
        py::arg("eps"),
        //: `page_table` resolves each touched ATOM to its slot; `None` is the dense
        //: backing, where the slot IS the atom and the bytes are the same.
        py::arg("page_table") = c10::optional<at::Tensor>());
  //: The step's PRECONDITION, not a mode of it: one CTA per batch-head expands the write
  //: side and ANDs it against residency, so the host learns whether the step needs the
  //: exact admission preamble at all. It publishes the condensation it judged
  //: (`atom_bits`), which is what a growth step admits from --
  //: docs/internals/rola_api.md#growth-probe
  m.def("rola_decode_growth_probe", &rola::decode::rola_decode_growth_probe,
        "RoLA T=1 decode growth probe: does this step write an atom the page table does "
        "not map? Publishes the step's write-atom set beside the verdict.",
        py::arg("amp_write"), py::arg("page_table"), py::arg("bits"), py::arg("growth"),
        py::arg("atom_bits"), py::arg("level_widths"), py::arg("level_strides"),
        py::arg("level_spans"));
  m.def("rola_decode_capacity", &rola::decode::rola_decode_capacity,
        "the decode path's maximum level width; must be >= the producer's "
        "MAX_BRANCH_WIDTH (tests/unit/test_decode_capacity.py pins the pair)");
  m.def("rola_decode_producer_width_mirror", &rola::decode::rola_decode_producer_width_mirror,
        "the C++ mirror of the producer's MAX_BRANCH_WIDTH; pinned EQUAL to the Python "
        "constant so a stale mirror cannot silently satisfy the capacity static_assert");

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
      .def("base", &rola::paging::VmmOwner::base)
      .def("grow", &rola::paging::VmmOwner::grow, py::arg("required_pages"))
      .def("rollback_to", &rola::paging::VmmOwner::rollback_to, py::arg("mapped_capacity_pages"))
      .def("reset", &rola::paging::VmmOwner::reset)
      .def("close", &rola::paging::VmmOwner::close)
      .def_property_readonly("closed", &rola::paging::VmmOwner::closed)
      .def_property_readonly("mapped_capacity_pages",
                             &rola::paging::VmmOwner::mapped_capacity_pages)
      .def_property_readonly("dense_limit_pages", &rola::paging::VmmOwner::dense_limit_pages)
      .def_property_readonly("committed_bytes", &rola::paging::VmmOwner::committed_bytes)
      .def_property_readonly("virtual_bytes", &rola::paging::VmmOwner::virtual_bytes)
      .def_property_readonly("allocation_granularity",
                             &rola::paging::VmmOwner::allocation_granularity)
      .def_property_readonly("chunk_pages", &rola::paging::VmmOwner::chunk_pages)
      .def_property_readonly("backing_kind", &rola::paging::VmmOwner::backing_kind);

  //: The page geometry is the ARENA's -- `page_rows x page_cols` fp32, which is
  //: `MMA_K_QUANTUM x (d_v + 1)` with the mass column fused. The owner takes it as two
  //: extents rather than naming them, because it owns no meaning for either:
  //: docs/internals/rola_api.md#vmm-binding
  m.def("rola_vmm_create", &rola::paging::create_vmm_owner, py::arg("device"),
        py::arg("dense_limit_pages"), py::arg("page_rows"), py::arg("page_cols"),
        py::arg("target_chunk_bytes") = 64ll * 1024ll * 1024ll);
}

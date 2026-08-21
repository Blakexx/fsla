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
#include "src/entmax/factor.cuh"
#include "src/paging/vmm_owner.cuh"
#include "src/facts/facts_api.cuh"
//: K31 R1 -- the box-native CARRY family, built ALONGSIDE the shipped facts
//: producers and reachable from no shipped dispatch (development/plans/
//: K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md section 6).
#include "src/carry/carry_api.cuh"
#include "src/intra/intra.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ---- the facts family (the chunk consumer's surviving fact producers) ---
  //: ONE PASS, BOTH ROW SETS: the membership union table always, and the atom-grain
  //: write bitmap when this call has residency to plan.  `mask` is the CALL CLASS
  //: (`rola/engine/types.py`'s M1/M2), which is compiled in; `atoms` is the
  //: runtime-null output skip within the stateful class.
  // ---- K31 R2: the standalone intra kernel -------------------------------
  m.def("intra_forward", &rola::intra::intra_forward,
        "K31 intra: the within-window causal term, REDuced into o/den", py::arg("pread"),
        py::arg("pwrite"), py::arg("gwrite"), py::arg("v"), py::arg("o"), py::arg("den"),
        py::arg("levels"), py::arg("level_width"), py::arg("level_modes"), py::arg("window"));
  m.def("intra_build_stamp", &rola::intra::intra_build_stamp,
        "K31 intra: a fact only the loaded binary can produce");
  m.def("intra_arms", &rola::intra::intra_arms, "K31 intra: the built (levels, window) arms");

  m.def("chunk_facts", &rola::chunk::facts,
        "facts: the union table and the atom-grain write bitmap, one pass", py::arg("pr"),
        py::arg("pw"), py::arg("chunk"), py::arg("bc"), py::arg("widths"), py::arg("mask"),
        py::arg("atoms"));
  //: the sparse grid's predicate INPUT: three liveness facts per (bh, block),
  //: reduced from the union table this call already built. The predicate OVER
  //: them is the dispatch's, because it differs by call class and by backing.
  m.def("chunk_block_bits", &rola::chunk::block_bits,
        "facts: the block-grain read/write/state liveness bitmap", py::arg("tbl"), py::arg("bc"),
        py::arg("widths"), py::arg("L"), py::arg("page_tbl") = py::none());
  // ---- K31 R1: the box-native carry family (NOT on any shipped path) ------
  m.def("carry_forward_inter", &rola::carry::forward_inter,
        "K31 carry kernel: the INTER (carried-state) term, undivided, plus the "
        "final state plane in lattice leaf order",
        py::arg("pr"), py::arg("pw"), py::arg("gw"), py::arg("v"), py::arg("widths"), py::arg("k"),
        py::arg("m"), py::arg("window"), py::arg("chunk"), py::arg("carve"),
        py::arg("s_in") = py::none());
  m.def("carry_build_stamp", &rola::carry::build_stamp,
        "K31 carry kernel: device-side build identity + the arm's census", py::arg("widths"),
        py::arg("k"), py::arg("m"), py::arg("window"), py::arg("chunk"), py::arg("carve"));
  m.def("carry_owner_rows", &rola::carry::owner_rows,
        "K31 carry kernel: the owner ballot rows at the lattice's spans");
  m.def("carry_arms", &rola::carry::arms, "K31 carry kernel: the built arms");
  // ---- K31 R4: the reverse carry family, two concurrent scans ------------
  m.def("carry_backward_inter", &rola::carry::backward_inter,
        "K31 R4: the cross-window reverse pass -- (d_pr, d_pw, d_gw, d_v, d_state_in, "
        "replay), with zero state snapshots",
        py::arg("pr"), py::arg("pw"), py::arg("gw"), py::arg("vt"), py::arg("dn"),
        py::arg("widths"), py::arg("k"), py::arg("m"), py::arg("window"), py::arg("chunk"),
        py::arg("s_in") = py::none(), py::arg("ds_out") = py::none(),
        py::arg("want_replay") = false);
  m.def("intra_backward", &rola::intra::intra_backward,
        "K31 R4: the within-window half of the four cotangents, REDUCED into the "
        "planes the carry's two scans also write",
        py::arg("pread"), py::arg("pwrite"), py::arg("gwrite"), py::arg("v"), py::arg("dnum"),
        py::arg("dden"), py::arg("d_pr"), py::arg("d_pw"), py::arg("d_gw"), py::arg("d_v"),
        py::arg("levels"), py::arg("level_width"), py::arg("level_modes"), py::arg("window"));
  m.def("intra_backward_arms", &rola::intra::intra_backward_arms,
        "K31 R4: the built reverse intra arms and their census");
  m.def("intra_backward_stamp", &rola::intra::intra_backward_stamp,
        "K31 R4: the reverse intra's device build stamp");
  m.def("carry_backward_stamp", &rola::carry::backward_build_stamp,
        "K31 R4: device-side build identity + the per-kernel census", py::arg("widths"),
        py::arg("k"), py::arg("m"), py::arg("window"), py::arg("chunk"));
  m.def("chunk_arms", &rola::chunk::arms,
        "facts: the built (C, BC, D, B) dispatch vocabulary with plan quantities");

  // ---- the producer's union entmax solve ----------------------------------
  m.def("entmax_union_forward", &rola::entmax::union_forward,
        "union entmax forward: midpoint values, read/write routes, packed support");
  m.def("entmax_union_backward", &rola::entmax::union_backward,
        "union entmax VJP into read/write logits");

  // ---- the producer's independent (per-side) solves ------------------------
  //: One entry per direction per family, each covering as many LEVELS as the caller
  //: groups into one width class -- docs/internals/entmax/factor.md#batched-levels
  m.def("entmax_factor_forward", &rola::entmax::factor_forward,
        "independent entmax forward over one or more levels: routes and packed support");
  m.def("entmax_factor_backward", &rola::entmax::factor_backward,
        "independent entmax VJP into the packed logits");
  m.def("routing_softmax_forward", &rola::entmax::softmax_forward,
        "softmax routing forward over one or more levels");
  m.def("routing_softmax_backward", &rola::entmax::softmax_backward,
        "softmax routing VJP into the packed logits");

  // ---- the T=1 decode path -------------------------------------------------
  //: A SIBLING of the chunk consumer, not a specialization of it: shared produce
  //: contract, own consumer, frozen config. Two of its three entries are capacity GATES
  //: rather than operations -- docs/internals/rola_api.md#registered-entries
  //: THE WHOLE STEP: the operand fold, the lattice factor tables, the gather-GEMV and the
  //: step's OWN admission question in ONE launch, over the caller's raw `[B, 1, H, *]`
  //: views -- docs/internals/rola_api.md#the-step
  m.def("rola_decode_forward", &rola::decode::rola_decode_forward,
        "RoLA T=1 decode step: operand fold + lattice factor walk + token-stationary "
        "gather-GEMV + the per-batch-head residency verdict, in one kernel.",
        py::arg("read"), py::arg("write"), py::arg("normalize"), py::arg("dials"),
        py::arg("g_write"), py::arg("v"), py::arg("state"), py::arg("ws"), py::arg("ctr"),
        //: The verdict's buffers: the per-batch-head answer, its reduction, the reducing
        //: CTA's arrival counter, the done flags a replay reads, and the write-atom set a
        //: host admission plans from.
        py::arg("growth"), py::arg("growth_any"), py::arg("growth_ctr"), py::arg("done"),
        py::arg("atom_bits"), py::arg("level_widths"), py::arg("level_row_offsets"),
        //: THE LEAF ORDER, as data: the `(k, m)` box the state plane is written in.
        py::arg("lattice_k"), py::arg("lattice_m"), py::arg("n_split"), py::arg("eps"),
        //: `page_table` resolves each touched ATOM to its slot; `None` is the dense
        //: backing, where the slot IS the atom and the bytes are the same.
        py::arg("page_table") = c10::optional<at::Tensor>(),
        //: THE SLACK POOL, all three or none: the slots the host pre-committed and
        //: pre-zeroed, how many of each batch-head's are spent, and the claim the walk
        //: addresses through until the table is folded.
        py::arg("pool_slots") = c10::optional<at::Tensor>(),
        py::arg("pool_cursor") = c10::optional<at::Tensor>(),
        py::arg("pool_map") = c10::optional<at::Tensor>());
  //: THE DEVICE-SIDE BUILD STAMP: a fact only the loaded binary can produce, which is the
  //: only kind of check an extension trap cannot survive --
  //: docs/internals/rola_api.md#decode-build-stamp
  m.def("rola_decode_residency", &rola::decode::rola_decode_residency,
        "the decode step's declared CTAs per SM for this arm, read off the launch bound",
        py::arg("decay"), py::arg("levels"));
  m.def("rola_decode_build_stamp", &rola::decode::rola_decode_build_stamp,
        "the decode step's own footprint, read back off a kernel this translation unit "
        "compiles");
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

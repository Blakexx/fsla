#pragma once

// THE HOST ENTRY SURFACE the pybind translation unit binds.
//
// **WHY THIS HEADER EXISTS.** The JIT-era shape put `PYBIND11_MODULE` inside
// `consumer.cu` and registered the paging entries from two `*_pybind.cuh` fragment
// headers that the same `.cu` `#include`d. Under an ahead-of-time build that is a
// per-`-gencode` recompile of the binding code, which is host-only and has nothing
// to gain from it; `flash_api.cpp` is the convention this follows instead. The
// bindings live in ONE host TU (`csrc/rola/rola_api.cpp`) and this header is the
// only thing it needs to see: the declarations of the entries the device TUs define.
//
// Nothing here is a public API. `rola/ops/_ext.py` is the only Python importer and
// `rola/interface.py` is the only thing that should call through it.

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rola {

//: `consumer.cu` -- the fused forward, its residency queries, and the closed-world
//: rule-4 arch check (validated once, on the first launch, from `launch_consumer`).
std::vector<at::Tensor> rola_consumer_forward(
    std::vector<at::Tensor> amp_read, std::vector<at::Tensor> amp_write,
    c10::optional<std::vector<at::Tensor>> amp_clock, c10::optional<at::Tensor> dials,
    at::Tensor read_bits, at::Tensor write_bits, at::Tensor read_tok_bits,
    at::Tensor write_tok_bits, at::Tensor read_union, at::Tensor write_union, at::Tensor owners,
    c10::optional<at::Tensor> g_read, at::Tensor g_write, at::Tensor v, at::Tensor state,
    c10::optional<at::Tensor> page_table, std::vector<int64_t> level_stride,
    std::vector<int64_t> level_modulus, std::vector<int64_t> level_row_offset,
    std::vector<int64_t> level_width, std::vector<int64_t> owner_stride,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices, int64_t owners_total,
    int64_t n_heads, int64_t BT, int64_t BC, bool global_norm, std::string mma_precision,
    bool compact, double eps,
    // ---- §6.4 sequence parallelism. All-or-nothing: either every one of these is
    // present (a split plan) or none is (`P = 1`, the primary configuration, which
    // takes the single-launch path below unchanged).
    c10::optional<at::Tensor> work_seg,        // [n_owners] int32, keyed like `owners`
    c10::optional<at::Tensor> work_nseg,       // [n_owners] int32
    c10::optional<at::Tensor> work_slot,       // [n_owners] int32
    c10::optional<at::Tensor> summary_owners,  // [n_summary, 2] int32
    c10::optional<at::Tensor> summary_seg,     // [n_summary] int32
    c10::optional<at::Tensor> summary_nseg,    // [n_summary] int32
    c10::optional<at::Tensor> summary_slot,    // [n_summary] int32
    c10::optional<at::Tensor> scan_owners,     // [n_split, 2] int32
    c10::optional<at::Tensor> scan_nseg,       // [n_split] int32
    c10::optional<at::Tensor> scan_slot,       // [n_split] int32
    c10::optional<at::Tensor> ws_state,        // [n_slots, BC, cols] float32
    c10::optional<at::Tensor> ws_scale,        // [n_slots, BC] float32
    c10::optional<at::Tensor> ws_leaf,         // [n_slots, BC] int32
    c10::optional<at::Tensor> ws_count);

int64_t rola_consumer_occupancy(bool decay, bool compact, int64_t BC, int64_t BT, int64_t DV);

int64_t rola_consumer_smem_bytes(bool decay, bool compact, int64_t BC, int64_t BT, int64_t DV);

void rola_consumer_check_arch_table();

int64_t rola_sm_shared_bytes();

}  // namespace rola

namespace rola::paging {}  // namespace rola::paging

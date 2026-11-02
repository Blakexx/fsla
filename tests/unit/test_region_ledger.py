"""The region ledger's function map (`tools/region_ledger.py`, docs/internals/tools/stall_census.md): every definition
form the kernel sources use starts a function -- a `__host__ __device__ constexpr` helper missed books its lines to the
definition before it -- and no statement inside a body does.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import region_ledger  # noqa: E402

SOURCE = """\
__device__ __forceinline__ void plain(int x) {
  constexpr int kStep = 2;
  constexpr uint32_t kMask = mask_of(3);
  static_assert(sizeof(int) == 4, "");
  static_for<4>([&](auto I) { x += I; });
}
__host__ __device__ __forceinline__ constexpr uint32_t unit(int l) {
  return l;
}
__host__ __device__ constexpr int slot(int d) { return d; }
__device__ __forceinline__ long long wide(int x) {
}
__device__ __forceinline__ const uint16_t* row(const uint16_t* p) {
}
constexpr bool holds() {
}
constexpr int kTable = table_of(4);
template <class P>
struct Sums {
  __device__ __forceinline__ int operator[](int j) const {
  }
  __host__ __device__ __forceinline__ int total() const {
  }
};
__global__ __launch_bounds__(256, 1) void kernel(int* y) {
}
"""


def test_every_definition_form_starts_a_function():
    got = region_ledger.function_map(SOURCE.splitlines())
    assert got == [(1, "plain"), (7, "unit"), (10, "slot"), (11, "wide"), (13, "row"), (15, "holds"), (19, "Sums"),
                   (20, "Sums::operator[]"), (22, "Sums::total"), (25, "kernel")]


def test_a_helper_line_is_its_own_function():
    fmap = region_ledger.function_map(SOURCE.splitlines())
    assert region_ledger.fn_of(fmap, 8) == "unit"
    assert region_ledger.fn_of(fmap, 3) == "plain"

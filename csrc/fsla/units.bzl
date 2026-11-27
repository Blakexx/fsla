"""THE RUNTIME UNITS: the hand units under `src/common/` that define no kernel -- the card's facts and the
selection that every kernel's host side calls. The library links them, and so does an instrument's library
(bazel/defs.bzl's `fsla_instrument`), whose host side calls them too. `tests/unit/test_units_list.py` holds this list to
`tools/kernels.py`'s `runtime_units`, which derives it from the source."""

RUNTIME_UNITS = [
    "src/common/arch_runtime.cu",
    "src/common/plan/environment.cu",
    "src/common/select.cu",
]

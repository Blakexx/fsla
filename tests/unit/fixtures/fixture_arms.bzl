"""THE FIXTURE ARM FAMILY: three rows on the decode family's real key, which `bazel/arms.bzl`'s `arm_family` instantiates in this
package (tests/BUILD.bazel) so the macro is exercised on a family that is not a shipped one. A build carries the first and the
last of them in the arm switch's probe; the middle row is what makes "declared but not built" a distinct case from "not
declared"."""

FIXTURE_ROWS = [
    [32, 1, 0],
    [32, 1, 1],
    [64, 2, 1],
]

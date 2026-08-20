// csrc/rola/src/facts/facts_ops.cuh -- the facts kernels' device operations:
// since the K31 merge-prep the family-neutral operation catalogue has ONE
// home, common/ops.cuh (the full baseline catalogue, restored verbatim from
// tag baseline/pre-k31), and this header is the facts family's stable
// include seam onto it.  The namespace stays `denseref::ops`, so the
// kernels' `using` directive does not move; every body the facts TU
// instantiates is the baseline body, and the post-build ratify gate is the
// referee -- the source digest moves, no ratified body does.
#pragma once

#include "common/ops.cuh"

# `carry/carry_arm_abi.cuh` — the dispatch/arm seam

<a id="pointers-not-tensors"></a>
## Why the seam speaks pointers and not tensors

This is the smallest header in the family — `cuda_runtime`, `cuda_bf16`, `cstdint`
— and that is a measured build requirement, not a preference.

K36 gave every carry arm its own translation unit. The obvious seam would have
been the one the single TU already used: pass `torch::Tensor` around, let each arm
allocate what it needs. Measured on this tree (sm_86, one `nvcc --threads=1`):

| translation unit | wall | `cicc` peak RSS |
|---|---|---|
| a bare CUDA TU (`cuda_runtime` + `cuda_bf16`, one trivial kernel) | 0.86 s | 0.19 GB |
| the same TU plus `#include <torch/extension.h>` | 33.7 s | 2.86 GB |
| one carry arm, with the tensor library | 39.8 s | 3.11 GB |
| **one carry arm, on this ABI** | **4.7 s** | **0.33 GB** |

The arm's own codegen is about 6 s. The extension header is about 34 s. With one
TU per arm that fixed cost is paid 22 times instead of once — roughly 12 minutes
of redundant host-side header parsing — and, worse, it sets the per-TU resident
set at 2.9 GB, which is the number `MAX_JOBS` is derived from. A tensor-typed seam
would have handed back most of the win the split exists to get, and capped the
parallelism that is the other half of it.

So **every allocation is the dispatch's**. `carry.cu` sizes the ballot table from
`carry_arm_rows_entries_<i>` and the probe's four planes from `CarryProbeShape`,
and the arm TU only launches into memory it was handed. `carry.cu` pays the
extension header once, because it is the one TU that must.

## The seven entry points

`CARRY_ARM_DECLARE` declares them and `CARRY_ARM_DEFINE` (`carry_arm.cuh`)
defines exactly these:

* `carry_arm_forward_<i>` — launch the body. Its signature *is* `launch<A>`'s.
* `carry_arm_rows_<i>` / `carry_arm_rows_entries_<i>` — fill the ballot table, and
  say how many rows it needs so the caller can size it.
* `carry_arm_probe_shape_<i>` / `carry_arm_probe_<i>` — describe the segment
  layout's round trip, and run it. `CarryProbeShape::carved` is zero for an arm
  that carves nothing: the per-stream segment belongs to the carved body, and the
  dispatch refuses on that flag rather than in the kernel.
* `carry_arm_census_<i>` / `carry_arm_row_<i>` — the device census and the
  `arms()` row, written into a caller-owned buffer of `kCarryCensusFields` /
  `kCarryRowFields`.

An arm the build did not compile has no declaration, because `CARRY_ARMS_X`
expands only over `ROLA_CARRY_BUILD_<i>`. That is what makes the seam's "unbuilt
carry arm" refusal the only thing a caller asking for it can reach — a link error
is not a diagnostic.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/carry/carry_arm_abi.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="carry-arm-declare"></a>
### `CARRY_ARM_DECLARE`

THE STATE PLANES AND THE TABLE ARE THE CALLER'S, which is the shipped contract
(docs/internals/state.md section 4): a continuation hands the SAME plane as `s_in` and
`s_out` and advances it in place, and `page_tbl` null is the dense backing -- the slot
is then the atom.  BOTH PLANES NULL IS THE NULL-STATE CALL and reaches an arm variant
nothing is read, written, or allocated at either end (state.md section 1) -- a
RUNTIME null check inside the one compiled body (K41; no STATE template axis).

WHAT ONE ARM'S TRANSLATION UNIT EXPORTS.  Five entry points: launch the body,
launch and size its ballot table, and state its census and its row.
`CARRY_ARM_DEFINE` (`carry_arm.cuh`) defines exactly these.  The SEGMENT LAYOUT
PROBE left with the per-stream segments and the exchange it gated (K35 M1P).

<a id="kcarryrowfields"></a>
### `kCarryRowFields`

the `arms()` row's width: the nine declaration fields plus BC, owners, smem and the
read side's carve level.

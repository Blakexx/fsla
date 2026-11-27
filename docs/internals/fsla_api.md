<!-- ABSENT: flash_api carry_owner_rows carry_backward_stamp fsla_consumer_forward fsla_consumer_occupancy fsla_consumer_check_arch_table fsla_consumer_smem_bytes fsla_sm_shared_bytes page_plan_abi_version page_plan_status_codes page_plan_workspace_bytes fsla_page_plan_prepare fsla_page_plan_finalize fsla_page_plan_publish fsla_page_admit_mask fsla_page_cache_reset -->
# `csrc/fsla/fsla_api.cpp` — the extension's operator registration

The only translation unit that registers with torch, and it does so through torch's **stable ABI**: every entry is an
operator in the `fsla` namespace, `torch.ops.fsla.<entry>`, declared by a schema (`STABLE_TORCH_LIBRARY`) and
implemented by a boxed function (`STABLE_TORCH_LIBRARY_IMPL`, `TORCH_BOX`). The binary depends on torch's stable C shim
and header-only types, never on torch's C++ ABI, so one build loads under every torch from the targeted minimum on
(`tools/build_flags.py`'s `TORCH_MIN`; [`docs/build.md`](../build.md#stable-abi)). The kernels meet torch only through
[`common/torch_seam.cuh`](common/torch_seam.md).

Symbols: a TU is a translation unit; `-gencode` is the nvcc flag that emits one device code path per target
architecture; a schema is torch's operator signature string (`int[] widths`, `Tensor(a!) num` for a tensor the call
writes).

---

## <a id="why-cpp"></a>1. Why it is a `.cpp` and not part of a kernel `.cu`

Registration is host-only. Left inside a `.cu` it is recompiled once per `-gencode` target for no benefit, and every
architecture added to the built set multiplies that cost. Every declaration it names is a host header
(`src/decode/decode_api.cuh`, `src/common/plan/plan_api.cuh`, `src/entmax/entmax.cuh`, `src/entmax/factor.cuh`, `src/mask_prepare/mask_prepare.cuh`, `src/common/geom_api.cuh`,
`common/arm_switch.cuh`), so no device body is in scope.

## <a id="one-library"></a>2. One library, registered here only

Every family registers into the one `fsla` namespace, because they are halves of one contract: the carry and the decode step read and write one state page format, and the decode step is a sibling of the prefill, not a specialization of it. Every `m.def`
and `m.impl` is in this file, so there is one place to read to know what the extension offers;
`tests/unit/test_closed_world_arch_gate.py` scans exactly these registrations.

## <a id="module-name"></a>3. The module object is only a handle

`fsla_cu13/_C.abi3.so` is what an import names, in the binary plugin the build ships as ([build.md#wheels](../build.md#wheels)):
`PyInit__C` builds an empty module against Python's limited API (3.10 and on), and loading the library is what runs the
static registrations. `fsla/ops/_ext.py` imports it after checking the plugin's build record is this `fsla`'s version, and returns
`torch.ops.fsla` under the names the package calls, so one binary serves every CPython from 3.10 and nothing in Python
reaches into the module itself.

## <a id="registered-entries"></a>4. What is registered

| group | operators |
|---|---|
| build, card and clock facts | `csrc_build_stamp`, `device_card`, `sm_clock_ghz` |
| the addressing block | `carry_geometry`, `carry_sub_boxes` |
| the producer's solves | `entmax_factor_forward`, `entmax_factor_backward`, `routing_softmax_forward`, `routing_softmax_backward` |
| the prefill's mask-preparation pass | `mask_prepare`, `mask_prepare_bytes` (`mask_prepare(Tensor[] masks, int[] slots, int[] widths, int[] heads, int tokens, Tensor density, Tensor(a!) workspace) -> ()`; `mask_prepare_bytes(int[] widths, int[] masked, int[] heads, int tokens, int device) -> int`, `device` the card's index: the op derives the card's window) |
| the planner's descriptor | `fsla_call_facts`, `fsla_descriptor`, `fsla_owed_kernels` |
| the segments of a call | `fsla_segment_plan` (`fsla_segment_plan(int length, int max_tokens, int max_heads, int batch, int heads, int device) -> int[]`: the steps and the tokens a step, and a head block's sequences and write heads, a maximum 0 no limit), `fsla_step_bytes` (`fsla_step_bytes(int[] widths, int[] masked, int[] heads, int value_width, int tokens, int device) -> int[]`: a step's workspace and operand bytes), `fsla_step_policy` (`fsla_step_policy(int[] widths, int[] masked, int[] heads, int value_width, int limit, int device) -> int[]`: a step's most tokens and most write heads under `limit` bytes, 0 for no limit); [`common/plan/segments.md`](common/plan/segments.md#entries) |
| the T=1 decode path | `fsla_decode_forward`, `fsla_decode_launch`, `fsla_decode_lattice`, `fsla_decode_lattice_box`, `fsla_decode_build_stamp`, `fsla_decode_arms`, `fsla_decode_ledger_bind`, `fsla_decode_ledger_warps`, `fsla_decode_capacity`, `fsla_decode_producer_width_mirror` |

Every operator that launches a kernel opens with `check_arch_table()`, the closed-world refusal of an architecture the
binary was never measured on ([`common/arch_runtime.md`](common/arch_runtime.md)). A schema names every argument, so
`torch.ops.fsla` takes the same keywords the pybind binding did; the default (`sm_clock_ghz`'s spin) is the schema's.

Three of the decode entries are gates rather than operations: `fsla_decode_capacity` reports the decode path's maximum
level width, which must be `>=` the producer's `MAX_BRANCH_WIDTH` (`tests/unit/test_decode_capacity.py` pins the pair);
`fsla_decode_producer_width_mirror` is the C++ mirror of that producer constant, pinned equal to the Python value so a
stale mirror cannot satisfy the capacity `static_assert`; and <a id="decode-build-stamp"></a>`fsla_decode_build_stamp`
is the device-side fact a stale extension cannot fake ([`decode/decode_api.md`
§build-stamp](decode/decode_api.md#build-stamp)).

## <a id="the-step"></a>5. `fsla_decode_forward` — the whole step, one launch

The decode step is ONE kernel: the operand fold, the `(k, m)` lattice's factor tables and the token-stationary walk are
phases of it, so what the boundary passes is the caller's raw `[B, 1, H, *]` views and not a widened plane. There is no
separate fold entry and no `heads` argument -- `v`'s own shape carries `H`
([`decode/decode_fold.md`](decode/decode_fold.md)).

`state` is the dense `[BH, N / 16, page bytes]` plane ([`decode/decode_api.md` §3](decode/decode_api.md#dense-state)).
`yacc` and `queue` are the work queue's sums and counters
([`decode/decode.md#queue`](decode/decode.md#queue)); `queue` is `[3 BH + 2]`.

# The contract, kernel by kernel

Every run-time context each kernel accepts, a fact a column, as the contract (`hardware/kernels/<kernel>/descriptors`)
states it and the build guarantees an arm for (`arms_gen`): a fact's values are points or a `lo..hi` range, a level's
width a `width_bits_<l>` log2 (0 where the call has no such level), a code its schema's unit
(`hardware/kernels/<kernel>/schema.json`). The README's table is the summary of these in a reader's terms. Both are
**generated** by `tools/supported.py` (`bazel run //:support_update` writes this page's block; the gate's `//:support`
fails if it drifts): change the contract, not the table.

<!-- BEGIN GENERATED: tools/supported.py --contract -->
**`decode`**

| Compute capability | GPUs | d_v | levels | decay | batch_heads | width_bits_1 | width_bits_2 | width_bits_3 | width_bits_4 | write_listed_1 | write_listed_2 | write_listed_3 | write_listed_4 | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..64 | 1, 2, 3, 4 | 0, 1 | 1..536870911 | 4..8 | 0, 4..8 | 0, 4..8 | 0, 4..8 | 0..1 | 0, 0..1 | 0, 0..1 | 0, 0..1 | `cu13` | ptxas 13.5.37 | built |

**`factor_backward`**

| Compute capability | GPUs | width | levels | alpha_tenths | logit_dtype | value_dtype | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..256 | 1..16 | 15, 20 | 1 | 1 | `cu13` | ptxas 13.5.37 | built |

**`factor_forward`**

| Compute capability | GPUs | width | levels | alpha_tenths | logit_dtype | value_dtype | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..256 | 1..16 | 15, 20 | 1 | 1 | `cu13` | ptxas 13.5.37 | built |

**`mask_prepare`**

| Compute capability | GPUs | width | levels | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..256 | 1..4 | `cu13` | ptxas 13.5.37 | built |

**`softmax_backward`**

| Compute capability | GPUs | width | levels | logit_dtype | value_dtype | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..256 | 1..16 | 1 | 1 | `cu13` | ptxas 13.5.37 | built |

**`softmax_forward`**

| Compute capability | GPUs | width | levels | logit_dtype | value_dtype | Toolchain | Assembler | Status |
|---|---|---|---|---|---|---|---|---|
| `sm_90a` | H100 | 2..256 | 1..16 | 1 | 1 | `cu13` | ptxas 13.5.37 | built |
<!-- END GENERATED: tools/supported.py --contract -->

# `csrc/rola/src/common/arch_rates.cuh` — the machine's rates

What an operation costs on an architecture, in cycles, one row an architecture, and the cost of a form's load
computed from them. It is the one table every cost model reads: a kernel's choice among exact forms (a sparse side's
digit groups, ruled 2026-10-06), a budget line (`measure/carry_model.py`), an offline pricing
(`measure/group_cost.py`). `measure/arch_rates.py` parses it (never restates it), checks its sources and mirrors the
cost in Python with the same integer arithmetic.

It is layer 1 beside [`arch_caps.cuh`](arch_caps.md): facts about the architecture, no judgment. `arch_caps` says what
the machine can do; this file says what each thing costs. The tensor rates here replaced `arch_caps`' percentage of the
fp32-accumulate rate (50 on consumer Ampere and Ada, 100 on GA100), one fact in two tables being one that can disagree.

<a id="sources"></a>

## Every value has a source

A `Rate` is a number of cycles and where it came from, one of three kinds:

- **measured** -- a lab record (`measure/lab/results/<lab>/sm_XX.json`) and the result in it. The check
  (`python -m measure.arch_rates --check`, gating in `tools/lint/lint_standards.py`) refuses a measured value that
  differs from its record's median by more than its rounding. Re-running a lab and re-recording it is how a value
  changes; editing the number alone fails the check.
- **documented** -- a public document and the figure in it.
- **announced** -- a value no source supports on this architecture yet, carried as a stated gap with its reason (sm_80's
  taxes and atomics are sm_86's until the rented A100 runs the labs). An announced value is a known deficiency in plain
  sight, never a silent default.

NVIDIA documents issue (one instruction a clock a scheduler), the shared-memory bank rate (32 banks of 32 bits a clock)
and peak tensor rates (the architecture whitepapers), not the cost of a native instruction by shape and type, a shared
atomic's read-modify-write, or how ALU work interferes with a running MMA chain: those are measured, once an
architecture. Nsight Compute's `*.peak_sustained` values are pipe issue ceilings, not these costs (its HMMA ceiling on
sm_86 is one in 8 cycles; m16n8k16 takes 16 at fp16 accumulate and 32 at fp32).

<a id="rates"></a>

## The rates

| field | unit | sm_86 | sm_80 |
|---|---|---|---|
| `schedulers` | an SM | 4, documented | 4, documented |
| `issue` | an instruction, a scheduler | 1, documented | 1, documented |
| `wavefront` | a shared-memory load wavefront (128 bytes), an SM | 1.001, `smem_rates` | 1, documented |
| `atomic` | a shared atomic instruction's least, an SM | 2.003, `smem_rates` | announced |
| `atomic_wavefront` | each wavefront of a shared atomic, an SM | 1.000, `smem_rates` | announced |
| `mma[kind]` | an MMA, a scheduler | 32 / 16 / 32 / 16, `tensor_tax` | 8 each, documented |
| `tax[kind][class]` | what an instruction of the class adds to the kind's MMA chain | `tensor_tax` | announced |

The kinds are m16n8k16 HMMA f16 x f16 -> fp32, f16 -> f16, bf16 -> fp32, and m16n8k32 IMMA u8 x s8. A shared atomic
add takes the larger of two cycles an instruction and a cycle a wavefront on sm_86: one, two and four wavefronts an
instruction cost 2, 2 and 4 cycles ([`smem_rates`](../tools/lab.md#smem-rates)).

sm_80's MMA rates from the A100 whitepaper: 312 dense FP16 and BF16 tensor TFLOPS (fp32 accumulate) over 108 SMs at
1410 MHz is 1024 FMAs a clock an SM, 256 a scheduler, so an m16n8k16 (2048 FMAs) takes 8 cycles; fp16 accumulate runs at
the same rate there; 624 dense INT8 TOPS gives 512 MACs a clock a scheduler, an m16n8k32 (4096) in 8. The same
arithmetic on the RTX 3090's whitepaper figures gives sm_86's measured 32 for fp32 accumulate (half the fp16-accumulate
rate on that card).

<a id="cost"></a>

## The cost

A form's LOAD is what it issues on its busiest scheduler -- MMAs by kind, taxed instructions by class, the rest -- and
what its SM's shared memory carries -- load wavefronts, atomic instructions and their wavefronts. Its COST is the max
of three components, in 1/256 cycles (`quantum`), integer so the same load always prices the same:

- tensor: each kind's MMAs at its rate, plus each taxed instruction at its class's tax beside the kind with the most
  MMA cycles;
- issue: every instruction at the issue rate;
- shared memory: load wavefronts at the wavefront rate, plus the larger of the atomics at their least and their
  wavefronts at theirs.

The max assumes the components overlap, as a kernel built to the ideal regime makes them; the tax is the one
interaction measured and priced. A built form's distance from its cost is that kernel's to close, and its gate's to
report. `Quanta` holds the rates without their sources for code that prices at run time; the header's `static_assert`
and `tests/unit/test_arch_rates.py` price the same load (16 fp32-accumulate HMMAs, 8 FP32 and 4 LDS beside them, 40
others; 64 wavefronts, 16 atomics with 24 wavefronts) and must agree: 133204.

<a id="rated"></a>

## The rated set

An architecture with no row is not `known`. A kernel that reads rates asserts `rates_of(kArch).known` at compile time,
so it refuses an unrated architecture the way `arch_caps`' `tabulated` refuses an untabulated one; the check refuses a
built architecture (one with a manifest under `tools/manifests/`) that has no row. Adding an architecture is a row and
the labs that measure its undocumented rates (`tensor_tax`, `smem_rates`); adding a rate is a field, a value a row and
the lab that measures it.

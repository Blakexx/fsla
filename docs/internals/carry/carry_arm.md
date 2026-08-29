# `carry/carry_arm.cuh` — one arm, as a type and as four entry points

<a id="one-arm-per-tu"></a>
## One arm per translation unit

This is the family's HEAVY header, and it is included by the per-arm translation
units and by nothing else. `carry.cu` must not include it: that is what stops the
dispatch from instantiating an arm locally and leaving the split doing nothing —
the failure mode `tools/fatbin_entries.py` exists to catch on the facts family, and
the one a `nm` check cannot see, because a `__global__` template's host stub is
weak and the linker keeps one.

Before K36 the carry family was one translation unit holding all 23 arms. One
`cicc` compiled them serially and peaked at several gigabytes, so a build spent
tens of minutes on a file `MAX_JOBS` could not help with, and the ratification
recompiled the whole thing. KERNEL_STANDARDS section 7 is the rule that was being
broken; the arm list in `tools/gen_shards.py` is now the generator of one `.cu`
per arm, and the arms compile in parallel.

## What is here

* `Arm<...>` — the topology, the window shape, the declaration and the two carves,
  as ONE type. The body, the geometry, the row plan and the census all read it, so
  an arm cannot be described two ways. The two carves are DERIVED from `(MO, side,
  stream count)` by the order-by-mode rule and never declared (K35 S1b).
* `launch<A>`, `census<A>`, `run_probe<A>`, `build_arm_rows<A>` — the four
  operations, as templates over the arm.
* `CARRY_ARM_TYPE` — one row of the list, as the arm type.
* `CARRY_ARM_DEFINE` — the entire content of a per-arm translation unit: the seven
  entry points `carry_arm_abi.cuh` declares. The generated file under
  `instantiations/` is an include, a guard and one expansion of this, which is what
  keeps 22 generated files from being 22 places to edit.

This header is FREE OF THE TENSOR LIBRARY, and so is everything it includes. That
is a measured build requirement rather than a preference, and the measurement and
the reasoning are in [`carry_arm_abi.md`](carry_arm_abi.md#pointers-not-tensors):
the extension header costs 34 s and 2.9 GB per translation unit against 6 s for an
arm's own codegen, so a tensor-typed seam would have paid that 22 times and pinned
the per-TU resident set that `MAX_JOBS` is derived from. Allocation is the
dispatch's; this side speaks pointers.

## Why a row and a kernel are not the same thing

An arm's TU is keyed on the KERNEL the arm instantiates, not on the row. A carved
arm takes `carry_kernel` (and the probe kernel keyed identically); an arm that
carves nothing takes `carry_split_kernel`, whose parameter list has no `MO` and no
stream counts. So two rows can be different ARMS and one kernel — rows 6 and 7 are
the deep-3 topology declared DENSE_BOTH and ALT, and the channel-split body ignores
the declaration. Emitting one instantiation from two objects is the state
`tools/ratify.py` refuses by name, so those two rows share a file and each keeps
its own `ROLA_CARRY_BUILD_<i>` guard.

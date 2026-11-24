# A family as the arm generator sees it (`csrc/fsla/src/common/plan/arm_family.h`)

`ArmFamily<K>` is what [`arms_gen.md`](arms_gen.md) reads of one kernel `K` (the generated definition of `plan/<kernel>/contract.h`): its `Facts`, its `Arm` and `Reach`, the key's
`fields` and `key(arm)`, the matcher's `reach` over a context's facts and `arm` over a point's, `matcher` (the file named where a context reaches no arm), `read` (the call's facts as text)
and `launch_reflects` (whether the launch function's arguments for a point are that point's facts). One specialization is written for each kernel whose arms the build generates:

- decode's is its matcher's own names (`decode_reach`, `decode_arm`, `decode_args_reflect`; [`decode_launch.md`](decode_launch.md)), the key `(d_v, levels, decay)`.
- the four entmax kernels' are one template, `EntmaxFamily<K>`, over [`entmax_launch.md`](entmax_launch.md)'s matcher: the key is the arm's (`lw`, `ipt`, `alpha15` where the kernel has an
  alpha, `logit`, `value`, the dtypes as codes).

A family joins the generator by being specialized here and named in `ARM_FAMILIES` (`bazel/arms.bzl`); the generator picks it by the kernel the contexts file names. A key's field is an integer;
a field that is a type in the kernel's name is a code, and `arm_family`'s `spellings` says which type each code spells ([`../../../build-graph.md#arms`](../../../build-graph.md#arms)).

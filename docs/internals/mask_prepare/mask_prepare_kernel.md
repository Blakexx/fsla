# The mask-preparation pass's device code (`csrc/fsla/src/mask_prepare/mask_prepare_kernel.cuh`)

The device half of the pass, in a header of its own so that TWO translation units compile the same text: the library's
(`mask_prepare.cu`, which adds the op's entry, its checks and the launch) and the lab's timed kernel
(`measure/lab/experiments/mask_prepare_time.cu`, which puts a clock around `mask_prepare_cta`). What it holds and why is
[`mask_prepare.md`](mask_prepare.md): the launch's facts (`Args`), the walk (`prepare`), a word finished (`finish`), a
CTA's work (`mask_prepare_cta`) and the kernel that calls it with its block index (`mask_prepare_kernel`).

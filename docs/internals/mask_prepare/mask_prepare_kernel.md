# The mask-preparation pass's device code (`csrc/fsla/src/mask_prepare/mask_prepare_kernel.cuh`)

The device half of the pass, in a header of its own so that TWO translation units compile the same text: the library's
(`mask_prepare.cu`, which adds the op's entry, its checks and the launch) and the lab's timed kernel
(`measure/lab/experiments/mask_prepare_time.cu`, which puts a clock around `mask_prepare_cta`). What it holds and why is
[`mask_prepare.md`](mask_prepare.md): the launch's facts (`Args`), a level-side's group from the call's carve (`carved_group_bits`), the walk (`prepare`), a word finished (`finish`), a
CTA's work (`mask_prepare_cta`) and the kernel that calls it with its block index (`mask_prepare_kernel`).

THE WINDOW IS THE CARD'S, a constant of the kernel's build: `mask_prepare_kernel<Bt>` takes `WORDS = plan::window_of_cc(Bt::build.cc) / 32` and passes it to `mask_prepare_cta<WORDS>` and `prepare<GB, WORDS>`; the lab's timed kernel takes it from the same rule (`plan::window_of_cc(plan::kHopperCc)`).

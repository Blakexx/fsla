# The mask-preparation pass's device code (`csrc/fsla/src/mask_prepare/mask_prepare_kernel.cuh`)

The device half of the pass, in a header of its own so that TWO translation units compile the same text: the library's
(`mask_prepare.cu`, which adds the op's entry, its checks and the launch) and the lab's timed kernel
(`measure/lab/experiments/mask_prepare_time.cu`, which puts a clock around `mask_prepare_cta`). What it holds and why is
[`mask_prepare.md`](mask_prepare.md): the launch's facts (`Args`), its shared memory (`Smem`: `Handed`, `Formed`), a side's leaf groups from the call's carve
(`hand_leaf_groups`), the walk (`form`), a group's word finished (`finish`), the leaf groups' lines (`lay_lines`), a CTA's work (`mask_prepare_cta`) and the kernel that
calls it with its block index (`mask_prepare_kernel`).

THE WINDOW IS THE CARD'S, a constant of the kernel's build: `mask_prepare_kernel<Bt>` takes `kWindowWords<Bt>` = `plan::window_of_cc(Bt::build.cc) / 32` and passes it to
`mask_prepare_cta<WORDS>`, `form<GB, WORDS>` and `lay_lines<WORDS>`; the lab's timed kernel takes it from the same rule (`plan::window_of_cc(plan::kHopperCc)`).

`mask_prepare_cta` takes its two shared tiles as arguments, so the lab's kernel, which runs several rounds of a CTA under one clock, hands each round its own handed
words (no round's are written under another's readers) and one tile of formed words.

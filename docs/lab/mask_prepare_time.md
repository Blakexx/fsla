# `mask_prepare_time`: the library's mask-preparation pass, timed at its seven group sizes

[`measure/lab/experiments/mask_prepare_time.cu`](../../measure/lab/experiments/mask_prepare_time.cu); how an experiment is run and
recorded: [`internals/tools/lab.md`](../internals/tools/lab.md).

**QUESTION.** What does the LIBRARY's mask-preparation pass
([`internals/mask_prepare/mask_prepare.md`](../internals/mask_prepare/mask_prepare.md)) cost a token on a mask it reads
once, at each of the seven group sizes its density byte can select: the three the lab's own kernel measured
([`mask_prep`](mask_prep.md): 1, 4 and 16 digits) and the four no kernel has (32 to 256 digits, whose lanes OR by warp
shuffle)?

**METHOD.** The experiment compiles the library's device code (`mask_prepare/mask_prepare_kernel.cuh`) and puts the
lab's clock around `mask_prepare_cta`, the work of one CTA of the library's grid: 1,024 threads over one 256-digit
level-side of 4 to 5 live digits a token, a window 1,024 tokens. A round is the CTA's own windows (4 at a group of one
digit, 16 at 4, 64 at 16 and above); a call is the rounds that cover 256 windows a CTA, each round fresh windows, so
the mask is read once. The density byte selects the group as it does in the library. Every word, count and total of
the first round is checked against the host's, in the library's layout. The library's CTA dispatches once outside its
own loops; here that dispatch sits inside the lab's loop over rounds, which the experiment declares.

**NOT MEASURED YET.** No record of this experiment is in the tree.

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
the mask is read once. The density byte selects the group as it does in the library. Every word and total of
the first round is checked against the host's, in the library's layout. The library's CTA dispatches once outside its
own loops; here that dispatch sits inside the lab's loop over rounds, which the experiment declares.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM pxfjwwmx, driver 580.196.07; 2026-11-25, the experiment at
9c2a0f30; 7 repetitions; the 7 checked rows pass, every word and total of each group size's
first round exact; the median). SM-cycles a token, the mask read once; the lab's kernel is [`mask_prep`](mask_prep.md)'s row at
the same group size:

| a group, digits | the library's pass | the lab's kernel | bytes a token, read + stored | bytes a cycle an SM |
|---:|---:|---:|---:|---:|
| 1 | 2.74 | 4.93 | 32 + 0.50 | 11.9 |
| 4 | 3.09 | 3.46 | 32 + 8.12 | 13.0 |
| 16 | 2.64 | 2.60 | 32 + 2.03 | 12.9 |
| 32 | 2.67 | not measured | 32 + 1.02 | 12.4 |
| 64 | 2.66 | not measured | 32 + 0.51 | 12.2 |
| 128 | 2.61 | not measured | 32 + 0.25 | 12.4 |
| 256 | 2.53 | not measured | 32 + 0.13 | 12.7 |

WHAT IT SHOWS.
- THE COST BY GROUP SIZE: 2.74 at a group of one digit; the highest 3.09, at a group of 4; the lowest 2.53, at a group of 256.
- THE SHUFFLE FORM (32 digits and above): 2.53 to 2.67, beside 2.64 at 16 digits, where a thread is a group.
- ONE RATE: the bytes a token a row moves, read and stored, over its cycles a token is 11.9 to 13.0 bytes a cycle an SM
  at every group size.

THE PREDICTIONS WRITTEN FOR THIS FORM OF THE KERNEL, AND THEIR BASIS. The pass when it stored a count a word, measured
at the experiment's 0fb5eb86, fit one rate over the bytes a token it read and stored: 11.8 to 12.9 bytes a cycle an SM.
Each row here was predicted as the bytes this form moves a token over that rate; the row at one digit was widened
upward, its held loads being 8 words' where they were 16.
- a group of 1: predicted 2.5 to 3.1, measured 2.74 -- HELD.
- a group of 4: predicted 3 to 3.5, measured 3.09 -- HELD.
- a group of 16: predicted 2.6 to 2.9, measured 2.64 -- HELD.
- a group of 32: predicted 2.6 to 2.9, measured 2.67 -- HELD.
- a group of 64: predicted 2.5 to 2.85, measured 2.66 -- HELD.
- a group of 128: predicted 2.5 to 2.8, measured 2.61 -- HELD.
- a group of 256: predicted 2.4 to 2.7, measured 2.53 -- HELD.

WHAT THE RUN SAYS, AND WHAT IS NOT SHOWN. Every range held. The ranges came from one rate over the bytes a token read
and stored, fitted on the pass when it stored a count a word; this form stores fewer bytes and its rows fall where that
rate puts them. It is two forms on one rate and no more: no counter was read, so that the multiprocessor's memory
traffic is what bounds the pass is LIKELY, NOT SHOWN.

Record: `measure/lab/results/mask_prepare_time/sm_90a.json`; the lab kernel's: `measure/lab/results/mask_prep/sm_90a.json`.

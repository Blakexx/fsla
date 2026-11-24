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

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM nbxcde8m, driver 580.196.07; 2026-11-24, the experiment at
0fb5eb86; 7 repetitions; the 7 checked rows pass, every word, count and total of each group size's
first round exact; the median). SM-cycles a token, the mask read once; the lab's kernel is [`mask_prep`](mask_prep.md)'s row at
the same group size:

| a group, digits | the library's pass | the lab's kernel | bytes a token, read + stored | bytes a cycle an SM |
|---:|---:|---:|---:|---:|
| 1 | 4.04 | 4.93 | 32 + 16.50 | 12.0 |
| 4 | 3.42 | 3.46 | 32 + 12.12 | 12.9 |
| 16 | 2.74 | 2.60 | 32 + 3.03 | 12.8 |
| 32 | 2.82 | not measured | 32 + 1.52 | 11.9 |
| 64 | 2.77 | not measured | 32 + 0.76 | 11.8 |
| 128 | 2.67 | not measured | 32 + 0.38 | 12.1 |
| 256 | 2.55 | not measured | 32 + 0.19 | 12.6 |

WHAT IT SHOWS.
- THE COST BY GROUP SIZE: from 4.04 at a group of one digit to 2.55 at the largest.
- THE SHUFFLE FORM (32 digits and above): 2.55 to 2.82, beside 2.74 at 16 digits, where a thread is a group.
- ONE RATE: the bytes a token a row moves, read and stored, over its cycles a token is 11.8 to 12.9 bytes a cycle an SM
  at every group size.

THE PREDICTIONS WRITTEN FOR THIS FORM OF THE KERNEL, AND THEIR BASIS. The lab's kernel had measured 4.93, 3.46 and
2.60 at groups of 1, 4 and 16 digits, with 8, 32 and 128 bytes of the mask in flight a thread. This kernel holds 16
words' loads at one digit a thread (64 bytes in flight) and 8 at four digits (128): each was predicted between the
lab's rows at the bytes in flight around it. A group of 16 digits was predicted as the lab's row; 32 digits and above,
which no kernel had measured, as the row at 16 digits, a thread's mask traffic being the same.
- a group of 1: predicted 2.9 to 3.9, measured 4.04 -- MISSED.
- a group of 4: predicted 2.6 to 3.2, measured 3.42 -- MISSED.
- a group of 16: predicted 2.5 to 2.9, measured 2.74 -- HELD.
- a group of 32: predicted 2.4 to 3, measured 2.82 -- HELD.
- a group of 64: predicted 2.4 to 3, measured 2.77 -- HELD.
- a group of 128: predicted 2.4 to 3, measured 2.67 -- HELD.
- a group of 256: predicted 2.4 to 3, measured 2.55 -- HELD.

WHAT THE MISSES SAY. The ranges at groups of 1 and 4 digits priced the pass by the mask bytes a thread has in flight,
counting the mask's read traffic alone. What the table above fits is the multiprocessor's TOTAL memory traffic, the
stored counts and words included: every row's bytes a token over its cycles a token is one rate, the column on the
right. At one digit the stored counts are half as many bytes again as the mask read, and no lookahead removes them.

Record: `measure/lab/results/mask_prepare_time/sm_90a.json`; the lab kernel's: `measure/lab/results/mask_prep/sm_90a.json`.

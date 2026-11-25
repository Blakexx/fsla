# `mask_prepare_time`: the library's mask-preparation pass, timed at its group sizes

[`measure/lab/experiments/mask_prepare_time.cu`](../../measure/lab/experiments/mask_prepare_time.cu); how an experiment is run and
recorded: [`internals/tools/lab.md`](../internals/tools/lab.md).

**QUESTION.** What does the LIBRARY's mask-preparation pass
([`internals/mask_prepare/mask_prepare.md`](../internals/mask_prepare/mask_prepare.md)) cost a token on a mask it reads
once, at each of the eight group sizes it prepares on a level of 256 digits (a group of 256 is the level itself,
whole, and nothing is prepared for it): the three the lab's own kernel measured ([`mask_prep`](mask_prep.md): 1, 4 and
16 digits), 2 and 8, and the three whose lanes OR by warp shuffle (32 to 128 digits)?

**METHOD.** The experiment compiles the library's device code (`mask_prepare/mask_prepare_kernel.cuh`) and puts the
lab's clock around `mask_prepare_cta`, the work of one CTA of the library's grid: 1,024 threads over one 256-digit
level-side of 4 to 5 live digits a token, a window 1,024 tokens. A round is the CTA's own windows (4 at a group of one
digit, 8 at 2, 16 at 4, 32 at 8, 64 at 16 and above); a call is the rounds that cover 256 windows a CTA, each round fresh windows, so
the mask is read once. The density byte selects the group as it does in the library. Every word and total of
the first round is checked against the host's, in the library's layout. The library's CTA dispatches once outside its
own loops; here that dispatch sits inside the lab's loop over rounds, which the experiment declares.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM hgrdycm0, driver 580.196.07; 2026-11-25, the experiment at
4dca551b; 7 repetitions; the 8 checked rows pass, every word and total of each group size's
first round exact; the median). SM-cycles a token, the mask read once; the lab's kernel is [`mask_prep`](mask_prep.md)'s row at
the same group size:

| a group, digits | the library's pass | the lab's kernel | bytes a token, read + stored | bytes a cycle an SM |
|---:|---:|---:|---:|---:|
| 1 | 2.77 | 4.93 | 32 + 0.50 | 11.7 |
| 2 | 3.69 | not measured | 32 + 16.25 | 13.1 |
| 4 | 3.12 | 3.46 | 32 + 8.12 | 12.9 |
| 8 | 2.79 | not measured | 32 + 4.06 | 12.9 |
| 16 | 2.65 | 2.60 | 32 + 2.03 | 12.9 |
| 32 | 2.67 | not measured | 32 + 1.02 | 12.4 |
| 64 | 2.67 | not measured | 32 + 0.51 | 12.2 |
| 128 | 2.60 | not measured | 32 + 0.25 | 12.4 |

WHAT IT SHOWS.
- THE COST BY GROUP SIZE: 2.77 at a group of one digit; the highest 3.69, at a group of 2; the lowest 2.60, at a group of 128.
- THE SHUFFLE FORM (32 digits and above): 2.60 to 2.67, beside 2.65 at 16 digits, where a thread is a group.
- ONE RATE: the bytes a token a row moves, read and stored, over its cycles a token is 11.7 to 13.1 bytes a cycle an SM
  at every group size.

THE PREDICTIONS WRITTEN FOR THIS RUN, AND THEIR BASIS. The pass's two runs before (the experiment at 0fb5eb86, when it
stored a count a word, and at 9c2a0f30) fit one rate over the bytes a token it read and stored: 11.8 to 13.0 bytes a
cycle an SM. Each row here was predicted as the bytes its form moves a token over that rate: the six sizes the run
before held kept its ranges, and 2 and 8 digits, forms no run had, took theirs from the same rate.
- a group of 1: predicted 2.5 to 3.1, measured 2.77 -- HELD.
- a group of 2: predicted 3.6 to 4.2, measured 3.69 -- HELD.
- a group of 4: predicted 3 to 3.5, measured 3.12 -- HELD.
- a group of 8: predicted 2.7 to 3.15, measured 2.79 -- HELD.
- a group of 16: predicted 2.6 to 2.9, measured 2.65 -- HELD.
- a group of 32: predicted 2.6 to 2.9, measured 2.67 -- HELD.
- a group of 64: predicted 2.5 to 2.85, measured 2.67 -- HELD.
- a group of 128: predicted 2.5 to 2.8, measured 2.60 -- HELD.

WHAT THE RUN SAYS, AND WHAT IS NOT SHOWN. Every range held, the two new forms' with the rest (a pair a thread at 2
digits, two lines a thread at 8). The ranges came from one rate over the bytes a token read and stored, which the two runs
before had fit, and every row falls where that rate puts it: the pass is dearest at 2 digits, where it stores the most.
It is three runs on one rate and no more: no counter was read, so that the multiprocessor's memory traffic is what
bounds the pass is LIKELY, NOT SHOWN.

Record: `measure/lab/results/mask_prepare_time/sm_90a.json`; the lab kernel's: `measure/lab/results/mask_prep/sm_90a.json`.

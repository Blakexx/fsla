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
level-side of 4 to 5 live digits a token, a window 1,024 tokens. The call has a second level of 256 digits, whole on both sides and unmasked: with it a set holds 16 leaves at every group, so the carve raises none and the density byte selects the group timed. A round is the CTA's own windows (4 at a group of one
digit, 8 at 2, 16 at 4, 32 at 8, 64 at 16 and above); a call is the rounds that cover 256 windows a CTA, each round fresh windows, so
the mask is read once. The density byte selects the group as it does in the library. Every word and total of
the first round is checked against the host's, in the library's layout. The library's CTA dispatches once outside its
own loops; here that dispatch sits inside the lab's loop over rounds, which the experiment declares.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM 02z2f0yn, driver 580.196.07; 2026-11-25, the experiment at
34b9a08d; 7 repetitions; the 8 checked rows pass, every word and total of each group size's
first round exact; the median). SM-cycles a token, the mask read once; the lab's kernel is [`mask_prep`](mask_prep.md)'s row at
the same group size:

| a group, digits | the library's pass | the lab's kernel | bytes a token, read + stored | bytes a cycle an SM |
|---:|---:|---:|---:|---:|
| 1 | 3.63 | 4.93 | 32 + 0.50 | 8.9 |
| 2 | 3.85 | not measured | 32 + 16.25 | 12.5 |
| 4 | 3.14 | 3.46 | 32 + 8.12 | 12.8 |
| 8 | 2.85 | not measured | 32 + 4.06 | 12.6 |
| 16 | 2.73 | 2.60 | 32 + 2.03 | 12.5 |
| 32 | 2.76 | not measured | 32 + 1.02 | 12.0 |
| 64 | 2.74 | not measured | 32 + 0.51 | 11.9 |
| 128 | 2.67 | not measured | 32 + 0.25 | 12.1 |

WHAT IT SHOWS.
- THE COST BY GROUP SIZE: 3.63 at a group of one digit; the highest 3.85, at a group of 2; the lowest 2.67, at a group of 128.
- THE SHUFFLE FORM (32 digits and above): 2.67 to 2.76, beside 2.73 at 16 digits, where a thread is a group.
- ONE RATE FROM 2 DIGITS UP: the bytes a token a row moves, read and stored, over its cycles a token is 11.9 to 12.8 bytes a cycle an SM;
  at one digit, where a CTA is 4 windows and its first warp's carve stands beside so short a walk, 8.9.

THE PREDICTIONS WRITTEN FOR THIS RUN, AND THEIR BASIS. Each row was predicted about an unrecorded reading of it on a
held box, the pass deriving the carve a CTA as it does here; the group of one digit was moved down for the carve's
leaner form since that reading.
- a group of 1: predicted 3.35 to 3.75, measured 3.63 -- HELD.
- a group of 2: predicted 3.8 to 3.9, measured 3.85 -- HELD.
- a group of 4: predicted 3.05 to 3.12, measured 3.14 -- MISSED.
- a group of 8: predicted 2.76 to 2.82, measured 2.85 -- MISSED.
- a group of 16: predicted 2.64 to 2.7, measured 2.73 -- MISSED.
- a group of 32: predicted 2.67 to 2.73, measured 2.76 -- MISSED.
- a group of 64: predicted 2.65 to 2.72, measured 2.74 -- MISSED.
- a group of 128: predicted 2.59 to 2.66, measured 2.67 -- MISSED.

WHAT THE RUN SAYS, AND WHAT IS NOT SHOWN. A CTA's first warp now derives the call's carve and hands its group to the rest
behind a barrier, once a round here. AT ONE DIGIT, where a round is 4 windows, that shows whole: the row is the dearest
but one. THE SIX ROWS FROM 4 DIGITS UP MISSED HIGH, by 0.01 to 0.04, at most 1.3 percent: each range was 0.06 to 0.07 wide
about one box's unrecorded reading, and these rows follow a rate that moves more than that from one rented box to the
next. THE WRONG PRICE: the spread between boxes, taken as nothing. NOT SHOWN: what the carve costs a CTA at each size
(the record holds no row without it), and that the multiprocessor's memory traffic is what bounds the rows from 2
digits up (no counter was read).

Record: `measure/lab/results/mask_prepare_time/sm_90a.json`; the lab kernel's: `measure/lab/results/mask_prep/sm_90a.json`.

# `mask_prep`: a mask turned into its groups' words and counts

[`measure/lab/experiments/mask_prep.cu`](../../measure/lab/experiments/mask_prep.cu); how an experiment is run and
recorded: [`internals/tools/lab.md`](../internals/tools/lab.md).

**QUESTION.** What does it cost a token to compute, from a sparse level-side's mask as the prefill is handed it -- a
word of 32 tokens a digit -- each group's word (its digits' words ORed) and the group's live tokens of the window
before that word? Does the cost move with the group's size, or with how dense the mask is?

**METHOD.** A CTA of 1,024 threads, a thread a (window, group) pair of one 256-digit level-side; a window is 1,024
tokens, 32 words a digit. A thread walks its group's 32 words in order with a running count, holding two words'
loads at fixed registers, each issued two words ahead (at a group of one digit the toolkit reads one of the two in
the pass that issued it, which the experiment declares). A group of 1, 4 or 16 digits; a mask of
4 to 5 live digits a token and one of 32; in cache (the grid over 4 windows a CTA's worth at every group size, its CTAs
sharing them) and once-read. At a group of one digit the group's
word is the mask's own and only the counts are written. Every word, count and total of the first round is checked
against the host's.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM mjh6nsj7, driver 580.196.07; 2026-11-24, the experiment at
f22d40f0; 7 repetitions; the 6 checked rows pass, every word, count and total of each kernel's first
round exact; the median). SM-cycles a token for one side's sparse level:

| a group | 4 to 5 live digits: in cache | once-read | 32 live digits: in cache | once-read |
|---|---:|---:|---:|---:|
| 1 digit | 1.01 | 4.93 | 1.00 | 4.92 |
| 4 digits | 0.96 | 3.46 | 0.94 | 3.44 |
| 16 digits | 1.10 | 2.60 | 1.10 | 2.60 |

WHAT IT SHOWS.
- THE MASK'S DENSITY: the denser mask moves no row by more than 1.7% of the sparser one's figure.
- ONCE-READ, BY THE GROUP'S SIZE: 4.93, 3.46 and 2.60 SM-cycles a token at groups of 1, 4 and 16 digits; a group of 16 is below a group of 1, 0.53 of it.
- IN CACHE: 1.01, 0.96 and 1.10; a group of 16 is above a group of 1, 1.10 of it.
- A GROUP OF 1, ONCE-READ AGAINST IN CACHE: 4.9 times.
- A GROUP OF 4, ONCE-READ AGAINST IN CACHE: 3.6 times.
- A GROUP OF 16, ONCE-READ AGAINST IN CACHE: 2.4 times.

WHAT THE IN-CACHE ROWS ARE AND ARE NOT. They run the grid over one working set at every group size: 4 windows a CTA's
worth, so at groups of 4 and 16 digits, where a CTA's round is 16 and 64 windows, several CTAs own the same windows
and store the same words, counts and totals to the same places. The outputs are the same whichever CTA writes them;
the rows' times include those shared stores. At a group of one digit each CTA owns its own windows.

Record: `measure/lab/results/mask_prep/sm_90a.json`.

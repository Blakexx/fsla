# `mask_prep`: a mask turned into its groups' words and counts

[`measure/lab/experiments/mask_prep.cu`](../../measure/lab/experiments/mask_prep.cu); how an experiment is run and
recorded: [`internals/tools/lab.md`](../internals/tools/lab.md).

**QUESTION.** What does it cost a token to turn a sparse level-side's mask, as the prefill is handed it -- a word of
32 tokens a digit -- into what the gather and the carry read blind: each group's word (its digits' words ORed) and
the group's live tokens of the window before that word? Does the cost move with the group's size, or with how dense
the mask is?

**METHOD.** A CTA of 1,024 threads, a thread a (window, group) pair of one 256-digit level-side; a window is 1,024
tokens, 32 words a digit. A thread walks its group's 32 words in order with a running count, holding two words'
loads at fixed registers so that no load is read in the pass that issued it. A group of 1, 4 or 16 digits; a mask of
4 to 5 live digits a token and one of 32; the windows in cache and once-read. At a group of one digit the group's
word is the mask's own and only the counts are written. Every word, count and total of the first round is checked
against the host's.

**NOT MEASURED YET.** The experiment is built and its kernels checked; no card has run it.

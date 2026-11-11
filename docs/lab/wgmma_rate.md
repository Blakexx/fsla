# `wgmma_rate`

Mirrors `measure/lab/experiments/wgmma_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What is one `m64n64k16` wgmma's own time on the tensor pipe, apart from its commit group's round trip, and do two
warpgroups of a CTA share that pipe?

THE FORM. One or two warpgroups a CTA, a CTA an SM. A step issues K wgmmas into K accumulator sets in one commit group
and waits for it, K = 1, 2, 4 and 6, with A in registers or in shared memory. Every operand descriptor and register
A's words are held in registers, so the step loop is the fence, the K `HGMMA`, the wait and the branch -- read in the
SASS (sm_90a): no descriptor arithmetic inside it. The slope of a step's cycles over K is a wgmma's own time and the
intercept the group's round trip; the K = 1 rows are [`wgmma_regs`](wgmma_regs.md)'s readout-alone forms with the
descriptors held. Every accumulator is checked against the host's.

UNMEASURED: no hopper card has run it (`bazel run //tools:lab -- pack wgmma_rate --arch sm_90a`, then `carried` on the
card, under its kill timeout).

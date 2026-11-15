# `reg_five`

Mirrors `measure/lab/experiments/reg_five.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`reg_trade`](reg_trade.md) traded registers between two warpgroups of a CTA. Can a CTA of FIVE warpgroups, 640
threads, trade them among its roles, and what does the compiler give such a kernel at entry?

THE FORM. A CTA an SM of five warpgroups, its launch bounds one CTA an SM. Two warpgroups raise themselves to 160
registers a thread, one lowers to 80 and two to 40: 480 over the five. Each role's whole body is in its own branch,
its `setmaxnreg` first. A thread holds as many words as its role's registers hold -- 128, 32 or 12 -- through a
loop that rewrites every one each trip, and stores their sum; every sum is checked exactly.

WHAT THE COMPILER SAYS (ptxas of the pinned toolkit, sm_90a; no device). A KERNEL OF 640 THREADS ENTERS WITH 96
REGISTERS A THREAD, 61,440 in all: the SM's 65,536 over 640 is 102.4, and the count is a multiple of 8. A raise
takes from the CTA's pool only what its other warpgroups released from that count, so THE ROLES' COUNTS SUM TO
61,440, NOT TO THE SM'S FILE: here 2 × 160 + 80 + 2 × 40 = 5 × 96. The kernel compiles to two releases (to 40 and
to 80) and one raise (to 160), with no local memory. At 80 a role holding 40 words spills; at 40 one holding 12
does not. THE ENTRY COUNT GOES BY WHOLE WARPGROUPS: kernels of 528, 560, 592 and 624 threads also enter
with 96, and of 656, 672, 704 and 768 with 80 -- the SM's file over the threads rounded up to 128s -- so a partial
fifth warpgroup has no more registers a thread than a whole one.

THE TRADE BY WARP (a second kernel). 16 warps enter at 128 registers a thread; ONE WARP of the last warpgroup
raises itself to 176 and its neighbour lowers itself to 80, the other fourteen -- their own warpgroup's two among
them -- untouched; the raised warp holds 144 words, the lowered 32, the rest 96. WHAT THE COMPILER SAYS: it is
honoured -- one release to 80 and one raise to 176 (`USETMAXREG`), each role's body in its own branch on the
warp, no local memory. ON THE DEVICE: not in the record below, which predates the kernel.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60425304, driver 580.173.02; 2026-11-12, the experiment at
bed41355; 7 repetitions, every one of the 84,480 threads' sums exact):

| the form | cycles a trip a CTA |
|---|---|
| five warpgroups: two at 160, one at 80, two at 40 | 666 (665 to 667) |

WHAT IT SHOWS. FIVE WARPGROUPS TRADE REGISTERS ON THE DEVICE when their counts sum to what the kernel entered
with: the two raised to 160 return from their raise and every role's held words are exact. A CTA of 640 threads
has 61,440 registers for its roles, 4,096 under the SM's file.

Record: `measure/lab/results/reg_five/sm_90a.json`.

# `drain_role`

Mirrors `measure/lab/experiments/drain_role.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`drain_rate`](drain_rate.md) put an SM's rate for fp32 adds into global memory at one 128-byte line every 9.6
cycles, whatever issues them. A carry's CTA holds a token's partial twice, once a warpgroup, where its two boxes
share their read-sparse digits. What does a token cost when a DRAIN ROLE of warps pulls both partials from shared
memory, adds them, and issues ONE line of f16 pairs -- and what does an f16 line cost by itself? Do value loads on
the same path add to it?

THE FORMAT'S SUM ACROSS CTAS IS fp32 ([precision.md](../architecture/precision.md)): an f16 line is priced here as
a cost, and is no adopted form -- what the f16 sum gives up in accuracy is not measured by this experiment.

THE FORM. A CTA an SM.
- `direct`: 4 warps add a token's 64 channels as 32 f16 pairs, one line, from registers.
- `role`: two source warpgroups each store a chunk of 64 tokens x 64 channels fp32 into their panels. 1, 2 or 4
  drain warps, a chunk behind, take the tokens of their residue: a token's two rows loaded (two batches of 8 at
  fixed registers, a batch's rows loaded two batches before its adds), added in fp32, rounded once to f16 pairs,
  and added as one line into the token's row -- the CTA's own rows, or rows drawn from 1,024 that every CTA shares.
  The sources hand a chunk over by an arrival a warp at a barrier; the drain warps free its panels by another.
- `loads`: one thread bulk-loads 32 KB a chunk from global memory, two in flight, nothing else running.
- `role and loads`: both in one CTA.
Every drained sum is checked exactly. A role's `__syncwarp` before its one-lane arrival compiles to a convergence
call, once a chunk; the experiment declares those.

UNMEASURED IN THIS FORM: the record is the experiment at 627e96c0, before a chunk's last trip reloaded its own
last batch and before the role beside the loads ran with four drain warps -- an f16 line 9.9 cycles a token a CTA;
the role 33.4, 20.5 and 11.7 with one, two and four drain warps; 32 KB a chunk of loads 10.5 alone and 3.7 more
beside the two-warp role.

Record: `measure/lab/results/drain_role/sm_90a.json` (the experiment at 627e96c0).

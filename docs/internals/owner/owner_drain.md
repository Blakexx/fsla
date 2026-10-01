# `csrc/rola/src/owner/owner_drain.cuh` — the drain

Mirror doc for the owner carry's drain: a pass of the window's read-live tokens' panel rows reduced into the output
planes, `num` `[BH, L, 64]` and `den` `[BH, L]` fp32, which the intra reduces into beside it. Vocabulary:
[`owner.md`](owner.md).

<a id="pass"></a>

## A drain pass

A token's 32 channels of the CTA's channel group are one warp instruction, a lane a channel: one `red.global.add.f32`
request of 128 bytes, the request the global reduction is priced by (0.15 chip clocks a warp reduction, measured).
Its den is one 4-byte reduction a token, by the first channel group alone (both groups compute the same den). The
panel here is fp32; the design's panel is the normalized s16 one, and a packed s16 reduction (two tokens a request) is
the measured lever for a later stage.

<a id="measured"></a>

## Measured

`--check drain` drains a panel of ones, so each token's `num` and `den` count the boxes it is read-live in, held exactly
against the words (`flagship-alt-k4`, `flagship-dense`, `flagship-alt-k4-L4096`). At `nl64k-alt-k4` the reductions a
head are 2.07M requests at `(16, 64)` and 3.66M at `(32, 32)`: a read-live token is reduced by every box of its
read-dense level, four at `(16, 64)` and eight at `(32, 32)`.

# `carry/plan.cuh` — one arm's plan

Mirrors `csrc/rola/src/carry/plan.cuh`: the box's tiles, registers and shared memory on the compiling card. Every size
is fixed by the plan and the card, never by the data (R22).

<a id="tenants"></a>
## The persistent tenants

`Tenants` holds what lives across the window, 17,920 bytes on every card and arm:

- each side's union tokens (1 KB), count and box lists' pass starts, two windows' each by parity: a window's are copied
  while the window before it is still read;
- the write lists' exponent pairs and the tokens' units, by parity;
- the unit book (a list's running unit, `fold.md#units`), the digit maxima (each warp's per digit) and the digit
  exponents (`parts.md#maxima`);
- the phase clock's rows; the barriers (`kMaxBars`: the readout ring's two, the fold ring's, the staging barrier and the
  fold heads' barrier, `carry_kernel.md#staging`);
- each warp's heads, read and fold (512 bytes each a warp);
- the CTA's addresses (`Addr`, 256 bytes reserved: `parts.md#place`).

<a id="forms"></a>
## The forms

A FORM is how the phase region holds a window: the image in `parts` (the whole state, or half its digits at a time,
`parts.md#store`), a readout pass of `slots` union slots, a fold unit of `unit` slots. `form_bytes` is its region: the
image, the pads and the panel (`parts.md#readout`), and the readout ring's two slots, which the window's passes and then
the fold's first unit take in turn (`ring_slot`: a pass and a unit where one pass covers the window, else both the
larger). `kForms` lists the forms in a fixed order, the fewest image stores a dense window first (parts x passes), then
the larger unit; `form_of` takes the first the card holds.

| card, arm | form (parts, pass, unit) | shared bytes |
|---|---|---|
| sm_86, the even box `(32, 1, 32, 1, 32, 0)` | 2, 256, 128 | 100,096 of 101,376 |
| sm_86, dense-dense `(32, 2, 32, 2, 32, 0)` | 2, 128, 64 | 91,904 |
| sm_80, the even box | 1, 512, 128 | 165,632 of 166,912 |
| sm_80, dense-dense | 1, 256, 128 | 165,632 |

The fold's ring is the image and the panel once the readout is done: `kImageUnits` units over the image, then the
panel's (`kFoldRing`: 3 units at sm_86's even box, 6 at sm_80's, 2 at either's dense-dense).

<a id="plan"></a>
## The plan

`Plan<Arm, Cc>` is an arm's key (`tools/gen_shards.py`'s `ARM_KEY`) on card `Cc`, the compiling one by default
(`arch::kArch`). The key fixes each side's dense rows a warp and its operand's level count; the plan asserts the box's
32 rows a side and 32 channels, and derives:

- each side's tiles: a warp folds lists `w + 8 j` and reads out digits `w + 8 i` (4 each), so a part of the image is
  whole m-tiles of every fold list and whole digits of every reader;
- the registers: `kStateRegs` = 128 fp32 a lane;
- the pool rows: a read row is the read side's operand slices (`kReadRow`: 64 bytes, or 128 with a coefficient level),
  a write row the write side's slices then the values (`kWriteRow`: 128 or 192);
- the form (`kForm`) and its offsets: `kImage`, `kPads`, `kPanel`, `kRing0`, `kRing1`, `kSmemBytes`.

The launch sizes for the device's card, and the kernel refuses any other size (`carry_arm.md#plan`).

# `carry/plan.cuh` — one arm's plan

Mirrors `csrc/rola/src/carry/plan.cuh`: the box's tiles, registers, pool rows and shared memory on the compiling card.
Every compiled size is fixed by the arm and the card, never by the data (R22); what the data sizes -- each window's
chunks and buffers -- the window's plan places in the area at run time (`schedule.md`).

<a id="tenants"></a>
## The persistent tenants

`Tenants` holds what lives across the window, 10,752 bytes on every card and arm:

- each side's union tokens (1 KB), count and group's boundaries in prefix form (528 bytes), two windows' each by
  parity: a
  window's are copied while the window before it is still read;
- the write lists' exponent pairs and the tokens' units, by parity;
- the unit book (a list's running unit, `fold.md#units`) and the window's scale a list (`fold.md#units`), the digit
  maxima (each warp's per digit, by parity) and the
  digit exponents (`parts.md#maxima`);
- the phase clock's rows; the barriers (`#barriers`); the CTA's plans, two windows' by parity (`schedule.md#plan`);
- the CTA's addresses (`Addr`, 256 bytes reserved: `parts.md#place`).

<a id="barriers"></a>
## The barriers

`CarryBarrier`, every card's, the async copies' barriers, each counting all 256 threads (`carry_kernel.md#barriers`):
the staging (a window's inputs landed), the NEXT read (a window's first read chunk, filled by the window before) and an
item's LANDED (its buffer's copies), one for each of a plan's `kPlanItems` items (`schedule.md#items`). Every other
rendezvous is the hardware barrier.

<a id="image"></a>
## The image

`kPlaneBytes` a digit (32 lists x 32 channels, f16), the whole image `kImageWhole` (64 KB) or `kImageRing` (32 KB):
a ring of two batches or one half of two (`schedule.md#buffers`). `kBatchPlanes` = 8 digits a batch, so a warp stores 8
digits of its 4 lists (half an m-tile of each) and reads one digit a batch (`kBatches` = 4).

<a id="plan"></a>
## The plan

`Plan<Arm, Cc>` is an arm's key (`tools/gen_shards.py`'s `ARM_KEY`) on card `Cc`, the compiling one by default
(`arch::kArch`). The key fixes each side's dense rows a warp and its operand's level count; the plan asserts the box's
32 rows a side and 32 channels, and derives:

- each side's tiles: a warp folds lists `w + 8 j` and reads out digits `w + 8 i` (4 each);
- the registers: `kStateRegs` = 128 fp32 a lane;
- the pool rows: a read row is the read side's operand slices (`kReadRow`: 64 bytes, or 128 with a coefficient level),
  a write row the write side's slices then the values (`kWriteRow`: 128 or 192);
- the shared memory: `kSmemBytes`, the card's whole opt-in maximum a CTA, and `kArea`, what the tenants leave -- 90,496
  bytes on sm_86, 156,032 on sm_80 and sm_87, 221,568 on sm_90.

The launch sizes for the device's card, and the kernel refuses any other size (`carry_arm.md#plan`).

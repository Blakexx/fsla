# `carry/plan.cuh` — one arm's plan

Mirrors `csrc/rola/src/carry/plan.cuh`: the box's tiles, registers, pool rows and shared memory on the compiling card.
Every compiled size is fixed by the arm and the card, never by the data (R22); what the data sizes -- each window's
chunks and buffers -- the window's plan places in the area at run time (`schedule.md`).

<a id="tenants"></a>
## The persistent tenants

`Tenants` holds what lives across the window and is the same on every card, arm and call: 13,648 bytes at W = 512,
the area from 13,696:

- both sides' counts; the write side's union tokens (1 KB) and its group's boundaries in prefix form (528 bytes), two
  windows' each by parity: a window's are copied while the window before it is still read;
- the write lists' exponent pairs, by parity;
- the unit book (a list's running unit, `fold.md#units`) and the window's scale a list (`fold.md#units`), the
  digit keys (by parity), the digit exponents in lane order and the scales' bits (`parts.md#maxima`);
- each leaf's exponent byte (1 KB, dead from the window's leaf scales to its next keys) under the window's den rows
  (2 KB, `parts.md#den`), each leaf's f16 scale, and each digit's den factor and largest den (the den state is
  registers, `fold.md#state`);
- the phase clock's rows;
- the CTA's addresses (`Addr`, 256 bytes reserved: `parts.md#place`).

The AREA starts where they end (`kAreaAt`, the same on every call) and ends where the read tables start.

<a id="read-tables"></a>
## The read tables

What scales with the read side's query heads a write head (`g`, grouped queries) is laid out by the host for each
call (`tables_of`, into `CarryParams::tables`) at the top of shared memory, each two windows' by parity: the read union
(`g` KB), its group's boundaries in prefix form (`528 g` bytes: 33 rows of `8 g` columns), each query head's token
units (`512 g` bytes) and the plan records (`plan_words(g)` words, `schedule.md#plan`). One arm serves every `g`; the
area is what the tables leave, 91,136 bytes at one head on sm_86, and the launch refuses a `g` whose area does not hold
the least plan (`least_plan_fits`). At the top, not after the tenants, because a lane's address in the area folds the
area's base into its instruction as an immediate: from a base read at run time each such address held a register,
and at 255 a thread they spilled.

<a id="barriers"></a>
## The barriers

None in shared memory since 2026-10-25: the async copies land on commit groups in a fixed order a window
(`carry_kernel.md#staging`), and every rendezvous is the hardware barrier (`carry_kernel.md#barriers`). The mbarriers
the step machine landed its two slots on, each counting all 256 threads, are retired with it.

<a id="image"></a>
## The image

`kPlaneBytes` a digit (32 lists x 32 channels, f16), the whole image `kImageWhole` (64 KB) or `kImageRing` (32 KB):
one half of two, stored and read by each read chunk (`schedule.md#buffers`). `kBatchPlanes` = 8 digits a batch, so a warp stores 8
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
- the shared memory: `kSmemBytes`, the card's whole opt-in maximum a CTA, `kAreaAt` where the area starts, and
  `kArea`, what the tenants and one head's read tables leave (the census's) -- 91,136 bytes on sm_86, 156,672 on sm_80
  and sm_87, 222,208 on sm_90.

The launch sizes for the device's card, and the kernel refuses any other size (`carry_arm.md#plan`).

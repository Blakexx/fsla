# `carry/carry_coefs.cuh` — the carry's coefficient lines

Mirrors `csrc/fsla/src/carry/carry_coefs.cuh`: a fold warp's expansion of a round's coefficients into its own tile
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the third of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)). It reads the round's tokens where the listing laid them
([`carry_list.md`](carry_list.md#lay)) and its tile is the coefficient operand of the steps that read the round's rows
([`carry_values.md`](carry_values.md#tile)). The fold calls it a round ([`carry_kernel.md`](carry_kernel.md#supply));
the lab checks every coefficient of the calls it runs ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)).

- <a id="rows"></a>A LEVEL'S ROWS on the hopper card are tile-major
  ([`../../ARCHITECTURE.md` section 1](../../ARCHITECTURE.md#1-what-the-kernel-computes)): a tile of 64 tokens, its
  16-digit blocks one after another, a block a token's 16 bytes a line (`kRowTileBits`, `kRowBlockBits`,
  `kRowBlockBytes`). So a token's bytes on up to 16 aligned digits of a level are one load, and on 32 or 64 digits a
  load a block, 1,024 bytes apart.
- <a id="shape"></a>`StepShape<bits a position, four>`: A STEP'S SHAPE IS COMPILE TIME. A step's columns are a set's leaves,
  16 to 64 of them (R25), and a column's leaf is a digit a level: the column's bits are cut into POSITIONS, position 0
  the lowest, each a level's digits inside the set, a count of bits of them. A lane that knows the shape as it compiles is a
  token's line in about 180 instructions at 16 by 4 -- a load a position, a conversion a byte, a multiplication a
  column, 8 stores; with the positions' widths as values every column's bytes would be picked out by shifts the lane
  computes, about three times the instructions. WHICH LEVEL A POSITION IS is a value: a pointer.
- <a id="class"></a>`FoldClass`, `kFoldClasses`: A FOLD'S CLASS is the call's levels, its step's shape and the bits of a set's CHUNKS.
  A set wider than a step is `1 << chunk_bits` steps' columns on the same value lines
  ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), each a chunk, and
  the chunks' bits come off position 0. The fold's windows are compiled a class, behind the fold's one switch a call
  ([`carry_kernel.md`](carry_kernel.md#fold)). THE TABLE HOLDS THE CLASSES OF ONE LEVEL AND OF TWO -- sets of 16, 32 and 64
  leaves, and of 128 and 256 as two and four chunks of 64, 5 and 19 of them -- and a call whose fold has none of them
  traps there. The levels are in the class so that a level the call has not costs nothing: no load, no factor. NOT
  BUILT: the classes of three and four levels ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#the-order-of-the-build), the contract's width).
- <a id="positions"></a>`fold_positions(operands, carve, place, fold, shift, mask)`: A FOLD'S POSITIONS, once a fold.
  The levels are ordered by their extent in a set (the smaller of the write group and the fold's extent), the widest
  first and the lower level first of two alike, so that a class is a set of extents and never their order across
  levels; a position holds its level's rows at the head, a tile's bytes, the fold's first digit, and the bits of a
  set's index that pick its group there (`fold_levels`' `shift` and `mask`:
  [`carry_groups.md`](carry_groups.md#sets)). A position the call has no level for holds nothing and is never read. EACH OF THE CARVE'S ARRAYS IS READ BEFORE THE LEVEL IS COMPARED WITH ANYTHING, as in
  `fold_levels`. It runs on the host as it does in a thread: the lab forms a call's positions with it.
- <a id="source"></a>`CoefSource`, `coef_source<levels, bits a position, chunk bits>(positions, set, chunk)`: where a chunk of
  a set is read -- a position's pointer at token 0's line of the block its first digit lies in, and that digit's place
  inside the block. The digit is the fold's first, the set's group on, the chunk on at position 0.
- <a id="stage"></a>`stage_bytes<levels, bits a position>(source, tokens, stage, live)`, `StageShape`, `kStageMost`: THE
  BYTES ARRIVE AS THE VALUE LINES DO, by asynchronous copies into shared memory (the owner, 2026-11-25: "I thought we
  were losing coefficents to smem async like values, and then expanding them from smem into operand areas? Why not?"
  -- and to the form laid out in answer, "Ok go"). Lane `i` takes place `i` of the round's 32 and copies its token's
  16-digit blocks a position, whole, 16 bytes a copy, into its slot of the stage; the copies are left uncommitted so
  that they join the round's value lines' group. A SLOT'S PITCH IS ODD in 16-byte lines (the blocks' count, made odd),
  so an instruction's 32 lanes write each of the banks' eight 16-byte places four times
  ([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)); the widest stage of the classes built is 5
  lines a token, 2,560 bytes (`kStageMost`, which every class's stage is asserted to fit where its windows are
  compiled). THE EXPANSION'S READS of a slot are a 16-byte load a block, or, of a position of fewer than 16 digits,
  the word or the pair of its block the digits lie in, at the same place in every lane's slot: with the odd pitch the
  32 lanes' words fall on the banks at most four to a bank, as the copies do. WHY NOT REGISTER LOADS, which this was first: a token's bytes loaded from global memory
  into registers took about 700 cycles to land and held the warp there, a set's first round 1,237 cycles in its
  coefficient phase where its later rounds were about 500
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)); a copy's latency rides the wait on the round's landing,
  which the warp has anyway.
- <a id="expand"></a>`expand_round<levels, bits a position>(source, tokens, stage, tile)`, `position_floats`: A LANE A
  TOKEN. Lane `i` reads its token's staged bytes a position -- a block a 16-byte load, fewer digits the word or the
  pair of the block they lie in -- converts them to fp32, multiplies positions 1 on
  into the columns' high bits once, and then a column is one multiplication, rounded to bf16 as the packed pair is
  formed ([`../../architecture/precision.md`](../../architecture/precision.md#components): multiplied in fp32,
  rounded once). A position of one digit is a factor of the whole line. A PLACE WITH NO TOKEN STORES ZEROS, whatever
  its slot of the stage holds: a short step is padded on the coefficient side
  ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), and the tile is used
  again a round, so nothing in it is left from the round before. NOT BUILT: the rows' scales.
- <a id="tile"></a>THE TILE, `coef_step<Shape>(tile, step)`: a round's two steps, a step its bricks one after another
  -- a brick 8 columns of the step's 16 tokens, a token's 8 columns a 16-byte line, tokens 8 to 15 behind tokens 0 to
  7 (`kBrickColumns`, `kBrickBytes`; the operand both-major steps read:
  [`../common/atoms/tensor.md`](../common/atoms/tensor.md#operand)). ITS STORES, by lanes and banks
  ([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)): an instruction is the 32 lanes' 16-byte
  stores of one brick, lane `i` at line `i % 16` of step `i / 16`'s brick, a step's bricks a whole number of the
  banks' 128 bytes -- each of the eight 16-byte places of the banks is written by four lanes, the count of any 512
  contiguous bytes.

# `common/atoms/copy.cuh` — the staged copies

Mirrors `csrc/fsla/src/common/atoms/copy.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): global to shared without the issuing thread, their groups and waits, and the 16-byte quantum moved synchronously.

- `stage_tile(dst, src)`: copy <a id="kstagebytes"></a>`kStageBytes` (16) bytes global -> shared without occupying
  the issuing thread. `kStageBytes` is a fact of the operation, the widest single asynchronous copy the staging
  issues, and staging loops are shaped from it. A group is every `stage_tile` since the last `stage_commit()`;
  `stage_wait<n>()` means all but the newest `n` groups have landed FOR THE ISSUING THREAD, and CTA-wide visibility
  needs a following `rendezvous()` -- a property of the contract that TMA/mbarrier satisfies the same way; the hopper
  family's bulk (TMA) form, one descriptor issue a tile, is not written yet and will be its own body of this contract.

- <a id="stage-tile-if"></a>`stage_tile_if`: the copy with its source predicated -- a dead quantum ZERO-FILLS its
  destination and reads nothing, so a partial tile is the same code as a full one and never reads past an operand's
  end. <a id="stage-tile-when"></a>`stage_tile_when`
  guards the copy itself (`@p LDGSTS`): a copy that must write nothing when dead -- a part a call has none of, whose
  destination another part owns -- where `stage_tile_if`'s zero-fill would land.

- <a id="stage-tile-n"></a>`stage_tile_n`: the copy of a run's first `bytes` (0 to 16), the rest zero-filled and
  nothing past them read, so a row that ends inside a quantum (a level whose width is not a multiple of sixteen) never
  reads the next level's digits (the facts pass's rows; the pass is deleted, `docs/internals/DELETIONS.md`).

- <a id="stage-run"></a>`stage_run`: the same copy at a narrower quantum. A run shorter than 16 bytes is still ONE
  asynchronous access, never a padded 16-byte one that would read a neighbour's bytes. The 4- and 8-byte forms are
  the cache-all variant, because the 16-byte bypass has no narrower member; source and destination are naturally
  aligned, the caller's contract, so no size is ever split here.

- <a id="stage-run-if"></a>`stage_run_if`: `stage_run`'s predicated form -- a dead run reads nothing and zero-fills,
  which makes a partial window arithmetic rather than a tail case (a token past the window lands zero amplitudes).

- <a id="stage-run-lanes"></a>`stage_run_lanes`: `stage_run`'s LANE-predicated form; a dead lane issues nothing and
  its destination keeps what it holds. Measured: dead lanes landing zero-size cost the memory pipe nothing on
  16-byte copies and ~60 cycles a copy on the 4-byte `.ca` ones (no shipped caller).

- <a id="load-vec16"></a>`load_vec16` / `store_vec16` and <a id="storesharedvec16"></a>`store_shared_vec16`: the
  16-byte quantum moved synchronously, the same width as every staged copy, so one bank analysis covers a buffer
  whether its rows were staged or computed. `copy_shared_to_global_16b` composes the two into one coalesced 128-bit
  store.

- <a id="stage-box"></a>`stage_box_2d`, `stage_box_3d`: the tensor copy in (`cp.async.bulk.tensor`, sm_90), one thread's
  issue a box of the tensor a `__grid_constant__` map describes, its bytes completing an `mbarrier` whose arrival stated
  them (`mbar_expect_bytes`, [`barrier.md`](barrier.md#expect-bytes)). MEASURED ([`tma_box`](../../../lab/tma_box.md)):
  a 48 KB stage of 128-byte lines, 128-byte swizzled, 966 cycles a CTA (50.9 bytes a cycle an SM) where one warp's
  `cp.async` takes 5,664 and the same box of 16-byte lines 5,360 -- so a box's lines are 128 bytes (the swizzled form a
  swizzled `wgmma` descriptor reads), never 16.

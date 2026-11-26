# `carry/carry_kernel.cuh` — the carry's device code

Mirrors `csrc/fsla/src/carry/carry_kernel.cuh`. THE SKELETON
([`KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)) of the design
[`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md): every role with its registers, its
hand-offs and its order of work, and TWO COMPONENTS BUILT IN IT, the fold warpgroups' supply and their coefficients
([below](#supply)); every other component is a stub named where it goes. It has no host half and no op: the lab runs it, on a call's real
operands ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)) and on a call of no tokens, where a window is its
hand-offs ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)). The supply is the listing
([`carry_list.md`](carry_list.md)) and the value lines' fetch ([`carry_values.md`](carry_values.md)) on the call's
groups ([`carry_groups.md`](carry_groups.md)); the coefficients are a round's bytes staged and expanded
([`carry_coefs.md`](carry_coefs.md)).

- <a id="roles"></a>THE ROLES. A CTA is 512 threads, four warpgroups of 128 (a warpgroup is the tensor step's unit):
  two FOLD warpgroups (`kFolds`), the READOUT's (`kReadAt`), and the DRAIN's four warps (`kDrainAt`).
- <a id="registers"></a>THE REGISTERS. A fold thread's and a readout thread's counts are the arm's (`FOLD`, `READ`,
  template parameters of `carry_cta`): a split the climb tunes, so the kernel states no number for them. A drain
  thread takes what the multiprocessor's register file leaves (`Registers<Bt, FOLD, READ>::kDrain`, from `regs_per_sm`
  of the build's caps; a split that leaves it under the instruction's least does not compile). A role's count is set
  once, at the head of its branch: two counts in one warpgroup hang the CTA. WHERE A ROLE'S STATE DOES NOT FIT ITS
  COUNT the remedies are taken in the owner's order (2026-11-25: "If we hit register issues we should first try loop re
  ordering, uniform switches, etc  and then try seeing if we can save registers from other phases, before going to
  smem"): a per-call value moves to shared memory last. THE FOLD'S ACCUMULATORS ARE TO BE SET ASIDE FIRST (the
  owner, 2026-11-26: "Should we pre allocate the accumulator registers before writing anything else so we know what
  we are working with"): 128 registers a fold thread (`kStateRegisters`), which at the lab's count of 176 leave its
  code 48. NOT YET SO, AND THE CODE SAYS SO where the fold's count is set: compiled at 48 the supply and the
  coefficients spill 3.5 to 4.1 KB a kernel and run 43 to 67 percent longer; at 72, 1.4 to 2.0 KB and 18 to 25
  percent; at 104, under 80 bytes and 2 to 6 percent (held readings: [`../../lab/carry_supply.md`](../../lab/carry_supply.md)).
  The fold's code wants about 105 to 110 registers; its round loop alone fits 48, and what does not is the state it
  carries from set to set. TRIED OF THE FIRST REMEDY: a switch on which fold a thread is in freed no register (the
  assembler keeps no CTA-wide value in a uniform register), and forming the window's state where it is used freed
  few and cost cycles.
- <a id="handoff"></a>A HAND-OFF between roles is a NAMED BARRIER of the two roles' threads: the producer's warps
  arrive and go on (`hand`), the waiter's sync (`take`). There are twelve -- a tile group's "published" (the folds to
  the readout) and "read" (the readout to the folds), a panel's "filled" (the readout to the drain) and "drained" --
  beside the roles' own three (a fold warpgroup's and the readout's, `kFoldId`, `kReadId`): fifteen of a CTA's sixteen,
  which a static assertion holds. WHY NAMED: on the idle skeleton a window's 16 hand-offs cost 528 cycles so, against
  849 as shared barriers with suspending waits ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)), and the
  owner keeps one form (25 November 2026: "I don't want to maintain two forms, so we need to pick one after the barrier
  result comes in"). A WAIT THAT SPINS was not tried there ("the spinning will look free until it contends with real
  work ... just keep spin in mind as something we can try if we see barriers as a problem in the final kernel"). A
  copy's landing is not a hand-off of this kind: it stays the copy's own wait.
- <a id="trace"></a>THE TRACER (`Tr`, a template parameter of every role). A role tells its tracer each wait's start
  and end, each arrival, and its phases (`Site`: a role's entry and exit, a window's start). The library's is
  `NoTrace`, whose calls are nothing, so the library takes no dependency and pays nothing; the lab passes warpline's
  stamps, and a warp's timeline is then the kernel's own account of where it waited
  ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)). A component adds its phases to `Site` as it is
  built; the supply's are a set's start, its tokens listed, the next set's words asked for, and a round's fetch begun,
  issued and landed. `round(fold, set, window, round, token, rows)` is the supply's: a fold warp calls it when it has issued a
  round's copies, which is how the lab copies a round's tokens and rows out with no test code in the library
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)).
- <a id="fold"></a>`fold_role`. A WINDOW'S HAND-OFFS STAND BEHIND ITS FIRST SET'S LISTING AND FIRST STAGE'S ISSUE,
  so that the stage lands under them (and under the image, when it is built): the listing and the copies' issue
  change no state, so the image is still the state as of the window's start. A held reading: the cell's window 3
  percent shorter, both levels' and the crossed call's 11 and 12; a window whose first set is empty gains nothing and
  its hand-offs stand a set's listing later (one digit 1.3 percent longer, the call of no tokens 8): A FORM THAT WINS
  FOUR CALLS AND LOSES ONE (R10), kept on the coordinator's judgment and put to the owner to rule
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)). THE IMAGE GOES OUT A TILE GROUP AT A TIME so that the readout starts on the first
  group while the folds store the rest, and each group waits on the readout's read of the window before because the
  image is overwritten in place: the state has one image, not a ring of them
  ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#a-window)). The warpgroup meets at its own
  barrier before its steps (every thread's operands are stored) and behind them (every step has read its tile, so the
  next window's copies may land in it: R21's release edge).
- <a id="readout"></a>`read_role`. TWO PANELS so that a pass's sums are stored while the drain adds the pass before:
  a panel's turn waits on the drain's arrival of its turn before, which is what holds the readout back when the
  drain's adds are the slower (R22: no cliff, the readout waits).
- <a id="drain"></a>`drain_role` follows the readout a panel behind and frees each panel as it has added it.
- <a id="channels"></a>A CTA'S VALUE CHANNELS are the tensor step's 64 (`kChannels`), a SLICE of a token's line: a
  wider value is more CTAs over the slices of the same leaves
  ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md)), and a CTA's place carries its slice
  ([`carry_groups.md`](carry_groups.md#place)).
- <a id="smem"></a>THE CTA'S SHARED MEMORY is one layout (`Smem`, [`../common/smem_layout.md`](../common/smem_layout.md)).
  Its tiles so far are the fold warps' token lists (`FoldTokens`, [`carry_list.md`](carry_list.md#lay): a warp 1,056
  places of 2 bytes, the eight 16.5 KB, each list its own shared object) and their rows (`FoldRows`): a warp two rounds of 32 rows (`kHeld`), 8 KB, the eight
  fold warps 64 KB, on a 1,024-byte boundary ([`../common/smem_layout.md`](../common/smem_layout.md#aligned)), which
  is where a step's operand of rows starts ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#rows)). ITS EDGES
  (R21): a round's copies are one committed group, and the warp waits for the group before it, so two are in flight;
  a round's rows go to the slot of its index's turn in its set, so they are written over by the set's round two later,
  or by the next set's first round where the set ends -- and either issue stands behind the tracer's call for the round
  whose rows they were, at its trip's end, where the wait of that round's steps will stand (NOT BUILT: the steps). And the fold
  warps' COEFFICIENT TILES (`FoldCoefs`, [`carry_coefs.md`](carry_coefs.md#tile)): a warp one tile, a round's two
  steps at the widest step's 64 columns, 4 KB, the eight 32 KB, on a boundary of the banks' 128 bytes; and their
  STAGED BYTES (`FoldStaged`, [`carry_coefs.md`](carry_coefs.md#stage)): a warp two stages in flight at the widest
  stage's 2,560 bytes, the eight 40 KB, a slot written by its stage's copies and read by its expansion, and written
  again by the stage two later, which is issued behind that expansion. A coefficient tile's warp
  writes it a chunk of a round and the chunk's steps read it, so the next expansion stands behind those steps' wait
  (NOT BUILT: the steps; the one tile is the form the owner took,
  [`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)).
- <a id="supply"></a>THE FOLD'S SUPPLY, built: a fold warp lists its sets' live tokens and fetches their value lines.
  A SET is one leaf group of the fold's box, a group a level ([`carry_groups.md`](carry_groups.md#sets)); a fold holds
  `fold_sets` of them, a power of two. THE WARPS' SHARES ARE THE CARVE'S, NOT THE DATA'S: with four sets or more a
  warp takes every fourth set, whole; with two, two warps share a set, and with one, all four -- and SHARERS SPLIT A
  SET BY BIT, each a half or a quarter of every word's tokens ([`carry_list.md`](carry_list.md#share)), listing its
  part and running its own rounds -- so a fold's four warps issue copies together whatever its sets. A SET'S WORD a lane
  is one word of the set's line, the write side's leaf group of that index ([`carry_groups.md`](carry_groups.md#lines)): the fold's sets are the write
  side's consecutive leaf groups from `first_leaf`, the leaf group of the fold's first group a level
  (`plan::leaf_group_of` of `plan::leaf_groups(carve, false)`, [`../common/plan/leaf_groups.md`](../common/plan/leaf_groups.md#order)), so the set `i` of a fold is line
  `first_leaf + i`. The line holds the call's own tokens alone, so nothing is masked and nothing is ANDed in the fold. Its address is formed once a fold and a step on a window. THE WORD IS
  LOADED A SET AHEAD AND USED WHERE IT IS NEXT NEEDED: a warp issues in order, so a word used where it is loaded holds the warp there until it lands, whatever
  was meant to run under it. The listing ([`carry_list.md`](carry_list.md#lay)) lays the set's tokens in the warp's list (`FoldTokens`, a tile of the CTA's layout: 8 warps of 1,056 places, 16.5 KB on the hopper card), a round reads its 32 places there, and the fetch
  ([`carry_values.md`](carry_values.md)) lands their lines in the warp's rows; a set with no live token issues
  nothing. A STAGE IS AS MANY OF A ROUND'S CHUNKS AS FIT ITS SLOT, and its copies one committed group: those chunks'
  coefficient bytes ([`carry_coefs.md`](carry_coefs.md#stage)) and, with a round's first stage, the round's value
  lines. The dense call's set of four chunks is one stage a round so: one issue and one wait where there were four
  (its window 23,655 cycles a CTA where it was 29,536, a held reading). A set's first
  stage is issued behind its lay; then each stage ISSUES THE STAGE AFTER IT, WAITS FOR ITS OWN, AND EXPANDS
  ([`carry_coefs.md`](carry_coefs.md#expand)) -- two groups in flight, the newer one's latency under this stage's work.
  A SET'S LAST ROUND ISSUES NOTHING: a branch on a value every lane holds (a vote), and its wait is for its own
  stage alone -- A DEPARTURE from the standards' default of a prefetch issued unconditionally into a pad
  ([`../../KERNEL_STANDARDS.md` section 29](../../KERNEL_STANDARDS.md#s29)), put to the owner with its record -- where it issued a stage of copies that moved nothing, each still wrote its place zero, about a
  twentieth of the cell's window. THE BASES ARE THE CALL'S AND THE WINDOW IS AN OFFSET: the value lines and each
  level's rows are addressed from a pointer that does not change, the window's and the token's place a 32-bit
  offset, so no pointer is stepped a window; a call whose head's value lines pass 4 GB (an unsigned offset), or whose widest level's rows of a head (its tokens times that level's digits, a byte each) pass 2 GB -- the rows' offset is formed in signed 32-bit arithmetic -- traps where the fold starts
  (the op's own refusal of it is owed with the op). A STAGE'S ORDER IS ITS LOADS AND ADDRESSES, THEN THE VALUE LINES'
  COPIES, THEN THE BYTES', under the wait for the stage that is landing. MEASURED: with the bytes' copies first the
  cell's window was 5,487 cycles a CTA and with them last 5,066, the crossed call's 3,274 and 3,079
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)). WHY IS ONE PROFILE'S READING, not on record: its largest
  stalls in a stage's issue were the first uses of tokens loaded from the list behind the two byte copies, whose
  lanes ask for 32 different lines each. The positions' rows are formed once a fold and stepped a window on. ONE SWITCH A CALL: `fold_role` switches on its fold's CLASS
  ([`carry_coefs.md`](carry_coefs.md#class)) -- the call's levels and its step's shape -- and each arm is
  `fold_windows<CLASS>`, the warp's windows COMPILED FOR THAT DEPTH AND THAT STEP SHAPE: a level the call has not has
  no instruction, and a column's digits are picked as the code is compiled. WHICH LEVELS A SIDE LISTS IS NOT A MATTER OF THE FOLD'S CODE: the pass has formed
  the side's leaf groups' lines over them ([`carry_groups.md`](carry_groups.md#lines)), so the fold reads one line a set whatever the levels it lists, and the
  switch is on the class alone. THE OWNER'S, 2026-11-25, when a switch on the 16 listed patterns nested with the class's made 304
  copies of these windows and the compiler ran the host out of memory: "The tensor step is basically coupled to group
  size right, up to a max of 64? I think that's the more important thing to switch on, for the listing should we
  switch on depth and runtime on listed pattern? What's the cost?", answered, to the branch-free form laid out with its estimate, "Ok": the fold's one switch a call is on the
  call's depth and its step's shape, and which levels are listed is read at run time, with no branch. The switch he ruled the day before ("switching uniformly to
  precomputed things defined for those patterns ... We are just templating in the code instead of at the arm") stays, on the pattern that changes the most code. The
  leaf groups' lines carry that listing for the whole fold, so the run-time part of the ruling is held by the pass and the fold has no flag a level. A WINDOW'S OWN ARITHMETIC IS KEPT FROM WINDOW TO WINDOW: where its
  value lines start is a step on, and where its set's line is a step of `leaf_step` words on, an address formed once a fold, with no multiplication and no division a window. The loops are bounded by values every lane holds (a shuffle from lane 0,
  a vote, a warp reduction) ([`../../lab/collective_branch.md`](../../lab/collective_branch.md)).
  NOT BUILT: sides of different head counts (one head index reads both); the steps that consume a round and its
  coefficients. THE COEFFICIENTS' BUDGET, stated before the code: a round a warp, a chunk -- a lane a shared load of
  its token, a copy a block a position and then a shared load of it (two each at two levels of 16 and 4 digits), a
  conversion a byte (20), a multiplication a
  column and its high bits' few (68), a packed rounding a pair of columns (32) and 8 stores of 16 bytes: about 180
  instructions a lane at 64 columns, 2.8 a column; its floor is the columns', 64 multiplications and 32 roundings a
  lane. THE BUDGET, stated before the code
  ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a set a window, a warp -- one load a
  lane of the set's line, the listing's running sums (5 shuffles, about 20 instructions a lane); the lay, about 15 instructions a lane a trip of the fullest word's tokens; a round of 32 places -- 8 shared loads and
  8 copy instructions (about 35). THE FLOOR is the copies' own: 2.1 to 3.0 cycles
  a line a CTA with 16 warps issuing ([`../../lab/carry_fetch.md`](../../lab/carry_fetch.md)), 5.8 with 4
  ([`../../lab/value_fetch.md`](../../lab/value_fetch.md)); a CTA's folds have 8. MEASURED in the composed CTA, stamped: a
  round's 8 loads and 8 copies 180 to 390 cycles of its warp whatever number of its 32 places are live, then a wait of up
  to 580 for the round before's copies; a set's sums and lay 420 to 930
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)).
- WHAT IS NOT BUILT is named in the code where it goes: the image, the readout's lists and coefficients, the steps,
  the sums' stores and the adds, and the tiles they add to the CTA's shared memory.
- <a id="the-cta"></a>`carry_cta<Bt, FOLD, READ, Tr>(operands, tracer)`: each role's whole body runs in its own branch,
  its registers set at the branch's head. A fold thread derives the call's carve and its CTA's place there, once
  ([`carry_groups.md`](carry_groups.md#carve)).

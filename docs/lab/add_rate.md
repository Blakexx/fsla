# `add_rate`

Mirrors `measure/lab/experiments/add_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's output is a line added to global memory a token a pair of boxes
([`carry_window`](carry_window.md)), and four warps adding lines from registers read 9.8 cycles a line a CTA
([`drain_role`](drain_role.md)). What holds an add to global memory: a limit a warp, the lanes an instruction
carries, the lines it touches, the add's type, or its being atomic?

THE FORM. A CTA an SM of 1, 2, 4 or 8 warps. Each warp issues 16 instructions a trip to its own rows of global
memory, from a register that never changes and to addresses that are constants off the warp's base: nothing waits
on anything.
- ONE LINE: an instruction's lanes add f16 pairs into one 128-byte line -- by 32 lanes, by 16 and by 8 (the others
  issue nothing).
- SPREAD: its 32 lanes each add into a different line.
- STORE: the one-line form as a plain 4-byte store a lane, no add.
- INTEGER AND fp32: the one-line form by a 32-bit integer add and by an fp32 add; and 64-bit integer adds, 8 bytes
  a lane, two lines an instruction.
- VECTOR: one add of four f16 pairs a lane, 16 bytes -- 32 lanes an instruction are FOUR lines, 8 lanes one: is an add's
  price its instruction's or its lines'?
- ON A FEW SMs: the one-line form, 4 warps on 8 and on 32 of the card's SMs and 8 warps on 8: is the limit an SM's own,
  or the card's shared among its SMs?
Every row's words are checked exactly.

MEASURED (H100 SXM, 132 SMs, a Nebius preemptible VM, driver 580; 2026-11-15, the experiment at 8de3aa6b; 7
repetitions, every word exact; the cycles in which each warp of the CTA issues one instruction, the median. The PCIe
card's run of 2026-11-15 read the same per line, 8.5 to 10.2, on 114 SMs):

| the form | 1 warp | 2 warps | 4 warps | 8 warps |
|---|---|---|---|---|
| f16 adds into one line, 32 lanes | 12.2 | 22.5 | 40.7 | 80.3 |
| the same on 8 of the card's SMs | | | 20.4 | 41.7 |
| the same on 32 | | | 20.9 | |
| the same, 16 lanes | 8.4 | | | |
| the same, 8 lanes | 7.8 | | | |
| f16 adds, a lane a line of its own | 85.8 | | 375.3 | |
| 32-bit integer adds into one line | 12.5 | | 40.7 | |
| fp32 adds into one line | 12.1 | | 41.1 | |
| 64-bit integer adds, two lines an instruction | 21.9 | | 77.8 | |
| plain stores into one line, 32 lanes | 1.8 | | 1.8 | |
| VECTOR f16 adds, 16 bytes a lane: four lines an instruction | 39.2 | | 155.4 | 304.5 |
| the same, 8 lanes: one line an instruction | 13.9 | | 38.6 | |

WHAT IT SHOWS.
- THE LIMIT IS THE CARD'S, SHARED AMONG ITS SMs: with every SM adding, an SM gets a line's add out about every 10
  cycles whichever of its warps issues it; with 8 or 32 SMs adding, every 5. The card takes some 13 lines a cycle --
  the SXM's 132 SMs at 10.2 and the PCIe card's 114 at 9.0 the same -- so adds spread over time or over fewer SMs buy
  no more of them. The rows are of lines a CTA owns and rewrites, which stay in the cache.
- IT IS THE ATOMIC ADD, WHATEVER ITS TYPE: a line by 32-bit integers, fp32 or f16 pairs costs the same; 64-bit
  integers, two lines an instruction, twice.
- AN ADD IS PRICED BY THE 32-BYTE SECTORS IT TOUCHES, NOT ITS BYTES OR ITS INSTRUCTION: some 2.1 to 2.5 cycles a
  sector an SM whether an instruction holds one line (4 sectors), four (the vector form, 16), or 32 lines of 4 bytes
  each (32 sectors carrying an eighth of their bytes). A vector add saves instructions, not cycles.
- A PLAIN STORE OF THE SAME LINE IS 1.8 CYCLES AND DOES NOT SLOW WITH THE WARPS -- of lines it rewrites, which
  its cache holds. That is no store's cost to a line nothing has touched; [`slot_sum`](slot_sum.md) asks that.

Record: `data:records/fsla/measure/lab/add_rate.run/20261115T223232Z-1d3a5b7b/output.json`.

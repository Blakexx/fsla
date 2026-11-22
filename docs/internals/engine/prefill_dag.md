# The prefill DAG

`fsla/engine/dags/prefill_dag.py`: one SEGMENT of a prefill -- its routes and values -- into its readout,
against the state the call is handed. `fsla.fsla_op` validates the context in the caller's grad mode
(`validate_context`), then walks this DAG inside an autograd node whose backward refuses; the layer
(`fsla.FSLA`) splits a sequence into segments by its declared memory budget and calls the op once a segment
(`docs/api.md` section 4.2).

## The nodes, in order

| Node | Gives | What it does |
|---|---|---|
| `rules.paging_preference` | `paging` | whether a state bound here gets an arena (the expert may say no) |
| `nodes.topology` | `widths`, `d_v` | the level widths and the value width |
| `nodes.classify_call` | `call_class` | the engagement bits: state in, state out, paged |
| `rules.variant` | `variant` | the variant: stateless or stateful |
| `activity` (stateful) | `atom_bits` | the per-atom `WRITTEN | READ` byte, the host fold of the masks: the state's write set (`fsla.engine.facts.planes.atom_bits`) |
| `box` (stateful) | `bc` | the carry's box, the arena's allocation extent |
| `residency` (stateful) | `state_in`, `arena` | the state's entry plane, this call's write atoms committed |
| `nodes.join` (stateful) | | waits for the admission before anything reads the page table |
| `nodes.backing` | `page_table`, `state_out` | the planes the launch is handed; a continuation's are one |
| `launch` | `num`, `den` | the carry at its bytes (`num` and the v+1 channel `den`) and the intra into the same planes, each over the reader factor (`fsla.ops.prefill.prefill`, [`#the-seam`](#the-seam)) |
| `post` | `y` | THE POST PASS, `y = num / (den + eps / r)`, `[B, T, H, d_v]` fp32 (`fsla.ops.prefill.readout`; `r` the reader factor the kernels leave out) |

## The seam

The DAG is seeded with the bundle itself. The carry and the intra (not built) read its rows and values where the
producer stores them, grouped or not (`docs/api.md#grouped-queries`). The `launch` node is one composition
(`fsla.ops.prefill.prefill`): the carry reduces its partials and reads each token's earlier-window denominator out as
its v+1 channel, and the intra reduces its in-window pairs into the same two planes, both sums the recurrence's over
the reader factor (`fsla/ops/prefill.py`'s header). No carry or intra kernel is built on this line: the envelope
refuses every segment for want of a carry arm, and the composition raises by name
([`docs/open-work.md`](../../open-work.md)).

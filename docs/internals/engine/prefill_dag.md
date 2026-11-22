# The prefill DAG

`fsla/engine/dags/prefill_dag.py`: one SEGMENT of a prefill -- its routes and values -- into its readout,
against the state the call is handed. `fsla.fsla_op` validates the context in the caller's grad mode
(`validate_context`), then walks this DAG inside an autograd node whose backward refuses. It runs over the descriptor of the call and the workspace
its caller built (`fsla.build_workspace`) and derives nothing of its own. The layer (`fsla.FSLA`) calls the op once a segment; the segment
length is an output of the prefill launch function (`docs/api.md` section 4.2, `docs/open-work.md` row A-f).

## The nodes, in order

| Node | Gives | What it does |
|---|---|---|
| `nodes.topology` | `widths`, `d_v` | the level widths and the value width |
| `nodes.classify_call` | `call_class` | the engagement bits: state in, state out, paged |
| `rules.variant` | `variant` | the variant: stateless or stateful |
| `nodes.backing` | `page_table`, `state_out` | the planes the launch is handed; a continuation's are one |
| `launch` | `num`, `den` | the carry at its bytes (`num` and the v+1 channel `den`) and the intra into the same planes, each over the reader factor (`fsla.ops.prefill.prefill`, [`#the-seam`](#the-seam)) |
| `post` | `y` | THE POST PASS, `y = num / (den + eps / r)`, `[B, T, H, d_v]` fp32 (`fsla.ops.prefill.readout`; `r` the reader factor the kernels leave out) |

The prefill graph holds the nodes that run over a descriptor and a workspace. The state's entry (`FSLAState._kernel_entry`) needs the arena's
allocation extent, the carry's box leaves, which only the prefill launch function's carve gives, so the entry on the prefill path is that
function's (`docs/open-work.md` rows A-c and A-f); a prefill is refused before any of it could run.

## The seam

The DAG is seeded with the bundle itself. The carry and the intra (not built) read its rows and values where the
producer stores them, grouped or not (`docs/api.md#grouped-queries`). The `launch` node is one composition
(`fsla.ops.prefill.prefill`): the carry reduces its partials and reads each token's earlier-window denominator out as
its v+1 channel, and the intra reduces its in-window pairs into the same two planes, both sums the recurrence's over
the reader factor (`fsla/ops/prefill.py`'s header). No carry or intra kernel is built on this line: the envelope
refuses every segment for want of a carry arm, and the composition raises by name
([`docs/open-work.md`](../../open-work.md)).

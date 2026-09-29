# The prefill DAG

`rola/engine/dags/prefill_dag.py`: one SEGMENT of a prefill -- its routes and values -- into its readout,
against the state the call is handed. `rola.rola_op` validates the context in the caller's grad mode
(`validate_context`), then walks this DAG inside an autograd node whose backward refuses; the layer
(`rola.RoLA`) splits a sequence into segments by its declared memory budget and calls the op once a segment
(`docs/api.md` section 4.2).

## The nodes, in order

| Node | Gives | What it does |
|---|---|---|
| `rules.paging_preference` | `paging` | whether a state bound here gets an arena (the expert may say no) |
| `nodes.topology` | `widths`, `d_v` | the level widths and the value width |
| `nodes.classify_call` | `call_class` | the engagement bits: state in, state out, paged |
| `rules.mask` | `mask` | the band: stateless or stateful |
| `support_modes` | `modes` | per level, which sides can hold exact zeros, off the topology's activations |
| `support_words` | `supports` | the intra's frozen support words of both sides |
| `activity` | `atom_bits` | the per-atom `WRITTEN | READ` byte from both sides |
| `box` | `bc` | the carry's box, the arena's allocation extent |
| `residency` (stateful) | `state_in`, `arena` | the state's entry plane, this call's write atoms committed |
| `nodes.join` (stateful) | | waits for the admission before anything reads the page table |
| `nodes.backing` | `page_table`, `state_out` | the planes the launch is handed; a continuation's are one |
| `launch` | `num`, `den` | the carry, then the intra into the same buffers (`rola.ops.prefill.prefill`) |
| `post` | `y` | THE POST PASS, `y = num / (den + eps)`, `[B, T, H, d_v]` fp32 |

## The seam

The kernels still compute on floating levels and values, so `run_prefill` seeds the DAG with
`RouteFactors.dequantized()` and `Values.dequantized()`, computed once a call; `support_words` and
`activity` derive from those levels. Each goes when its kernel reads the rows: the facts pass emits the
words and the activity from the u8 / f16 rows, and the intra and carry take the rows themselves.

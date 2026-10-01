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
| `support_modes` | `modes` | per level, which sides can hold exact zeros, off the bundle's `sparse` flags (`docs/api.md` §2.1) |
| `facts` | `liveness`, `unions`, `zeta`, `layout`, `statics` | THE FACTS PASS over the bundle, one launch: both sides' class-1 words and their tile unions, the fold's window scales, the layout the words are read in and the two sides' statics (`rola.ops.facts`) |
| `activity` | `atom_bits` | the per-atom `WRITTEN | READ` byte, folded from the words (`rola.engine.facts.liveness.page_bits`) |
| `box` | `bc` | the carry's box, the arena's allocation extent |
| `residency` (stateful) | `state_in`, `arena` | the state's entry plane, this call's write atoms committed |
| `nodes.join` (stateful) | | waits for the admission before anything reads the page table |
| `nodes.backing` | `page_table`, `state_out` | the planes the launch is handed; a continuation's are one |
| `launch` | `num`, `den` | the carry, then the intra into the same buffers (`rola.ops.prefill.prefill`) |
| `post` | `y` | THE POST PASS, `y = num / (den + eps / r)`, `[B, T, H, d_v]` fp32 (`rola.ops.prefill.readout`; `r` the reader factor the kernels leave out) |

## The seam

The DAG is seeded with the bundle itself. The facts pass and the intra read its u8 rows and s8 values
where the producer stores them; the carry, until its int8 port, computes on floating operands the prefill
dequantizes from the bundle once a call (`rola.ops.prefill._carry_levels`: the read rows at unit scale, since
the reader factor cancels, and the write rows and values whole). The words feed the activity fold and the
carry, the tile unions the intra's slab gate. `zeta` has no consumer until the carry's int8 fold divides by it.

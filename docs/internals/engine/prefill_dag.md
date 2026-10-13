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
| `facts` | `liveness`, `unions`, `list_exp`, `layout`, `statics` | THE FACTS PASS over the bundle, one launch: both sides' class-1 words and their tile unions, the write lists' exponents and each side's lists in the form of the carry arm the carve selects, the layout the words are read in and the two sides' statics (`rola.ops.facts`) |
| `activity` | `atom_bits` | the per-atom `WRITTEN | READ` byte, folded from the words (`rola.engine.facts.liveness.page_bits`) |
| `box` | `bc` | the carry's box, the arena's allocation extent |
| `residency` (stateful) | `state_in`, `arena` | the state's entry plane, this call's write atoms committed |
| `nodes.join` (stateful) | | waits for the admission before anything reads the page table |
| `nodes.backing` | `page_table`, `state_out` | the planes the launch is handed; a continuation's are one |
| `launch` | `num`, `den` | the carry, then the intra into the same buffers (`rola.ops.prefill.prefill`) |
| `post` | `y` | THE POST PASS, `y = num / (den + eps / r)`, `[B, T, H, d_v]` fp32 (`rola.ops.prefill.readout`; `r` the reader factor the kernels leave out) |

## The seam

The DAG is seeded with the bundle itself. The facts pass and the intra read its u8 rows and f16 values where the
producer stores them. The words feed the activity fold and the carry, the tile unions the intra's slab gate, and the
write lists' exponents the carry's fold ([`carry/fold.md#units`](../carry/fold.md#units)). The `launch` node's carry
leg, `rola.ops.prefill.prefill`, is not yet re-pointed at the rebuilt carry's launch surface (`rola.ops.carry.carry`,
which reads the bundle's rows and the facts pass's lists), so a prefill does not yet run end to end
([`docs/open-work.md`](../../open-work.md)).

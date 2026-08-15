# Parked cell records — the tiled consumer's fixtures

| file | status | what it is |
|---|---|---|
| `topology_manifest.json` | **PARKED, historical** | the committed statistics manifest of the tiled harness's named cells: per cell the widths, `BC`, seed, support constructors and the realized `mean_K`/`mean_KI` a run was allowed to see. |
| `latency_baseline.json` | **PARKED, historical** | per `(cell, arm)` median and IQR, with `clock_settled` and the timing source, as measured on the tiled consumer. |

Both are keyed to the V3 TILED consumer and its `BC` ladder, which retired at P67
D2 (`docs/internals/DELETIONS.md`). Their arms name `consumer_forward_kernel`
launches that no longer exist, so **no current gate reads either file and no
current number may be compared against one**: a chunk-arm latency is a different
kernel, a different tiling and a different envelope, and quoting these medians
beside it would be comparing two instruments.

They are kept rather than deleted because they are the RECORD of what was measured
— the same reason `perf/ledger.jsonl` keeps rows for retired arms. The successor
is the chunk-arm harness card, which defines its own cells and takes its own
baseline; when it lands, these files stay parked exactly as they are.

The harness that produced them (`benchmarks/bench/{cells,drivers,instruments,pairing}.py`,
`bench_consumer.py`, `benchmarks/probe/`) is recoverable from git at `c7eaeb9`.

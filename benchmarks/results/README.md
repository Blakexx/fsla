# benchmarks/results/ — the K38 measurements record

Committed JSONL, one small file per run: `<hw-profile>/<yyyy-mm>/<run_id>.jsonl`. The
schema, the writer epilogue, and the derived-database CLI (`tools/rola_db.py`) are
documented in `docs/measurement.md` ("The measurements record (JSONL) and its
derived index"). This directory is populated by measurements, not written by hand.

`<hw-profile>` is the GPU model + SM architecture (`bench.hw_profile.hw_profile`,
e.g. `rtx3080ti-sm86`, `a100-sm80`) — never a hostname, username, or machine id
(K38 RULED, 2026-08-28). `.rola/measurements.sqlite`, rebuilt from this directory
by `python tools/rola_db.py rebuild`, is gitignored and disposable.

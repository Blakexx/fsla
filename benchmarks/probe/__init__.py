"""The probe harness: the process-level driver around :mod:`bench`.

It WRAPS the measurement package, it does not replace it. `bench.cells` stays the
fixture registry, `bench.pairing` stays the methodology and the stopwatch,
`bench.instruments` stays the clock protocol, `tools/ratify.py` stays the codegen
gate. What lives here is what used to live in throwaway shell scripts: the process
driver, the discipline that those scripts kept omitting, and the writer that puts
every number into `perf/ledger.jsonl` with its provenance.

* :mod:`probe.registry` -- WHAT can be probed, one entry per question.
* :mod:`probe.discipline` -- the preconditions, enforced.
* :mod:`probe.duration` / :mod:`probe.counters` / :mod:`probe.sass` -- the three
  instruments, one subcommand each.
* :mod:`probe.report` -- the ledger read back as a generated table.
"""

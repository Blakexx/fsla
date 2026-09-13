# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""tools/rola_db.py -- K38: the measurements database, rebuilt from the committed record.

THE RECORD IS `benchmarks/results/<hw-profile>/<yyyy-mm>/<run_id>.jsonl` (committed,
text, mergeable, provenance in every row). THE DATABASE is `.rola/measurements.sqlite`
(gitignored, derived, disposable) -- an index over the record, never the record itself
(`docs/measurement.md`: "a committed SQLite database... a binary blob has no diff").
`rebuild` is the only writer of the database; every other subcommand only reads it,
and `rebuild` refuses nothing that got past the JSONL writer's own provenance gate
(`benchmarks/bench/store.py`) because that gate already ran when the row was written.

Paths: `store.results` and `store.db` in the dev config (`tools/dev_config.py`).

Usage:

    python tools/rola_db.py rebuild
    python tools/rola_db.py last --cell k16_w384 --sha <sha>
    python tools/rola_db.py history --cell k16_w384 --branch k31/measurements-db
    python tools/rola_db.py compare <sha_a> <sha_b> --cells k16_w384,dense
    python tools/rola_db.py baseline --stage <stage-name> --cell k16_w384
    python tools/rola_db.py regressions --threshold 1.10 --cell k16_w384 \\
        --baseline-stage <stage-name>
    python tools/rola_db.py sql "select cell, count(*) from results group by cell"

There is deliberately no console-script entry point: this repository's other
developer tools (`tools/ratify.py`, `tools/probe_cells.py`, ...) are all invoked the
same way, `python tools/<name>.py`, because the installed wheel ships `rola/`, not
`tools/` -- a console script here would be a command that exists in the source tree
and cannot exist in the thing users install (`benchmarks/README.md`'s reasoning about
why `rola-probe` was never a console script, carried forward).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import dev_config  # noqa: E402 -- path insert must precede this import
import host_budget  # noqa: E402

from bench.store import KINDS, results_dir  # noqa: E402

SCHEMA_SQL = """
CREATE TABLE runs (
    run_id          TEXT PRIMARY KEY,
    ts              TEXT,
    hw_profile      TEXT,
    machine_tag     TEXT,
    gpu             TEXT,
    driver          TEXT,
    cuda            TEXT,
    image_digest    TEXT,
    git_sha         TEXT,
    branch          TEXT,
    worktree        TEXT,
    dirty           INTEGER,
    stamp_magic     TEXT,
    arm_census_hash TEXT,
    harness_sha     TEXT,
    stage           TEXT,
    agent           TEXT,
    purpose         TEXT,
    env_json        TEXT,
    source_file     TEXT NOT NULL
);
CREATE TABLE results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    kind        TEXT NOT NULL,
    cell        TEXT,
    arm         TEXT,
    metric      TEXT NOT NULL,
    value       REAL,
    unit        TEXT,
    extra_json  TEXT
);
-- The card's index list (git_sha, cell, metric) / (branch, ts) / (arm, kind) spans
-- both tables once the schema is normalized into runs+results; a real composite
-- index cannot cross a join, so the intent is served here as one index per table
-- side that a `runs JOIN results` query actually uses.
CREATE INDEX idx_runs_git_sha     ON runs(git_sha);
CREATE INDEX idx_runs_branch_ts   ON runs(branch, ts);
CREATE INDEX idx_results_run_id      ON results(run_id);
CREATE INDEX idx_results_cell_metric ON results(cell, metric);
CREATE INDEX idx_results_arm_kind    ON results(arm, kind);
-- The joined shape most queries actually want: a result with its run's provenance
-- already attached, so `last`/`history`/`compare`/`baseline` are one query each.
CREATE VIEW measurements AS
    SELECT results.*, runs.ts AS run_ts, runs.hw_profile, runs.git_sha, runs.branch,
           runs.dirty, runs.stage, runs.agent, runs.purpose, runs.harness_sha,
           runs.stamp_magic
    FROM results JOIN runs ON results.run_id = runs.run_id;
"""


def db_path() -> Path:
    return Path(dev_config.get("store.db")).resolve()


def _iter_jsonl_files(root: Path):
    if not root.exists():
        return
    yield from sorted(root.rglob("*.jsonl"))


def rebuild(args) -> int:
    """Drop and recreate the database FROM the committed JSONL, nothing else. A
    fresh clone plus `rebuild` must reproduce the same database -- that is the
    whole point of the record/database split, and it is what the K38 stage
    validates directly rather than assuming."""
    #: THE K38 RECORDS DATABASE IS ONE SHARED ARTIFACT (LOCKS brief item 3):
    #: two concurrent `rebuild` invocations both truncate-then-rewrite `out`,
    #: so this whole drop/recreate/populate act is one exclusive, short-held
    #: file lock, not a race.
    with host_budget.file_lock("rola_db_epilogue"):
        rdir = results_dir()
        out = db_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        conn = sqlite3.connect(str(out))
        conn.executescript(SCHEMA_SQL)

        n_runs = n_results = n_files = 0
        for path in _iter_jsonl_files(rdir):
            n_files += 1
            rel = str(path.relative_to(ROOT))
            run_id = None
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "run":
                    run_id = row["run_id"]
                    missing = [k for k in ("run_id", "git_sha", "stamp_magic", "harness_sha")
                              if not row.get(k)]
                    if missing:
                        raise SystemExit(
                            f"{rel}:{lineno} is missing required provenance field(s) "
                            f"{missing} -- refusing to index an unusable row "
                            f"(docs/measurement.md).")
                    conn.execute(
                        "INSERT INTO runs (run_id, ts, hw_profile, machine_tag, gpu, driver, "
                        "cuda, image_digest, git_sha, branch, worktree, dirty, stamp_magic, "
                        "arm_census_hash, harness_sha, stage, agent, purpose, env_json, "
                        "source_file) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (row["run_id"], row.get("ts"), row.get("hw_profile"), row.get("machine_tag"),
                         row.get("gpu"), row.get("driver"), row.get("cuda"), row.get("image_digest"),
                         row["git_sha"], row.get("branch"), row.get("worktree"),
                         int(bool(row.get("dirty"))), row["stamp_magic"], row.get("arm_census_hash"),
                         row["harness_sha"], row.get("stage"), row.get("agent"), row.get("purpose"),
                         json.dumps(row.get("env") or {}, sort_keys=True), rel))
                    n_runs += 1
                elif row.get("kind") in KINDS:
                    if row.get("run_id") != run_id:
                        raise SystemExit(
                            f"{rel}:{lineno} is a {row.get('kind')!r} result whose run_id "
                            f"does not match this file's own run record -- a K38 record "
                            f"file is one run, self-contained, and this one is not.")
                    conn.execute(
                        "INSERT INTO results (run_id, kind, cell, arm, metric, value, unit, "
                        "extra_json) VALUES (?,?,?,?,?,?,?,?)",
                        (row["run_id"], row["kind"], row.get("cell"), row.get("arm"),
                         row["metric"], row.get("value"), row.get("unit"), row.get("extra_json")))
                    n_results += 1
                else:
                    raise SystemExit(f"{rel}:{lineno} has unrecognized kind {row.get('kind')!r}")
        conn.commit()
        conn.close()
        print(f"rebuilt {out} from {n_files} file(s): {n_runs} runs, {n_results} results")
        return 0


def _connect_ro() -> sqlite3.Connection:
    path = db_path()
    if not path.exists():
        raise SystemExit(f"{path} does not exist -- run `python tools/rola_db.py rebuild` first")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _print_rows(rows) -> None:
    rows = [dict(r) for r in rows]
    if not rows:
        print("(no rows)")
        return
    print(json.dumps(rows, indent=1, sort_keys=True))


def cmd_last(args) -> int:
    conn = _connect_ro()
    q = "SELECT * FROM measurements WHERE 1=1"
    params: list = []
    if args.cell:
        q += " AND cell = ?"
        params.append(args.cell)
    if args.sha:
        q += " AND git_sha = ?"
        params.append(args.sha)
    if args.metric:
        q += " AND metric = ?"
        params.append(args.metric)
    if args.arm:
        q += " AND arm = ?"
        params.append(args.arm)
    if not args.include_dirty:
        q += " AND dirty = 0"
    q += " ORDER BY run_ts DESC LIMIT ?"
    params.append(args.limit)
    _print_rows(conn.execute(q, params).fetchall())
    return 0


def cmd_history(args) -> int:
    conn = _connect_ro()
    q = "SELECT * FROM measurements WHERE cell = ?"
    params: list = [args.cell]
    if args.branch:
        q += " AND branch = ?"
        params.append(args.branch)
    if not args.include_dirty:
        q += " AND dirty = 0"
    q += " ORDER BY run_ts ASC"
    _print_rows(conn.execute(q, params).fetchall())
    return 0


def cmd_compare(args) -> int:
    conn = _connect_ro()
    cells = args.cells.split(",") if args.cells else [None]
    out = []
    for cell in cells:
        row_entry: dict = {"cell": cell}
        for label, sha in (("a", args.sha_a), ("b", args.sha_b)):
            q = "SELECT * FROM measurements WHERE git_sha = ?"
            params = [sha]
            if cell:
                q += " AND cell = ?"
                params.append(cell)
            q += " ORDER BY run_ts DESC LIMIT 1"
            r = conn.execute(q, params).fetchone()
            row_entry[label] = dict(r) if r else None
        out.append(row_entry)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


def cmd_baseline(args) -> int:
    """The rows THE PREVIOUS STAGE recorded -- read by name (`--stage`), not by
    re-deriving which stage was "previous" from git history. The §16 probe gate
    passes the string it already knows (its own brief names the prior stage)."""
    conn = _connect_ro()
    q = "SELECT * FROM measurements WHERE stage = ?"
    params: list = [args.stage]
    if args.cell:
        q += " AND cell = ?"
        params.append(args.cell)
    if not args.include_dirty:
        q += " AND dirty = 0"
    q += " ORDER BY run_ts DESC"
    if args.metric:
        q += ""  # cell/metric filters compose above; kept for readability
    rows = conn.execute(q, params).fetchall()
    if args.metric:
        rows = [r for r in rows if r["metric"] == args.metric]
    _print_rows(rows[: args.limit])
    return 0


def cmd_regressions(args) -> int:
    """A minimal ratio check for the §16 gate: the latest row for `--cell`/`--metric`
    on the current tree's git sha (or `--sha`) against the previous stage's
    (`--baseline-stage`) most recent matching row. `--threshold` is a ratio, e.g.
    1.10 for the §16 addendum's ">1.10x is RED" bar; this command reports the ratio
    and exits nonzero when it is exceeded, it does not rule."""
    conn = _connect_ro()

    def _latest(where: str, params: list):
        q = f"SELECT * FROM measurements WHERE {where} ORDER BY run_ts DESC LIMIT 1"
        return conn.execute(q, params).fetchone()

    base = _latest("stage = ? AND cell = ? AND metric = ?",
                   [args.baseline_stage, args.cell, args.metric])
    cur_where = "cell = ? AND metric = ?"
    cur_params = [args.cell, args.metric]
    if args.sha:
        cur_where += " AND git_sha = ?"
        cur_params.append(args.sha)
    cur = _latest(cur_where, cur_params)
    if base is None or cur is None:
        print(json.dumps({"verdict": "insufficient_data",
                          "baseline_found": base is not None,
                          "current_found": cur is not None}, indent=1))
        return 2
    ratio = cur["value"] / base["value"] if base["value"] else float("inf")
    verdict = "RED" if ratio > args.threshold else "ok"
    print(json.dumps({
        "verdict": verdict, "ratio": ratio, "threshold": args.threshold,
        "baseline": dict(base), "current": dict(cur),
    }, indent=1, sort_keys=True))
    return 1 if verdict == "RED" else 0


def cmd_sql(args) -> int:
    conn = _connect_ro()
    _print_rows(conn.execute(args.query).fetchall())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("rebuild", help="rebuild .rola/measurements.sqlite from the committed JSONL")

    p = sub.add_parser("last", help="the most recent matching result")
    p.add_argument("--cell")
    p.add_argument("--sha")
    p.add_argument("--metric")
    p.add_argument("--arm")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--include-dirty", action="store_true")

    p = sub.add_parser("history", help="a cell's whole time series")
    p.add_argument("--cell", required=True)
    p.add_argument("--branch")
    p.add_argument("--include-dirty", action="store_true")

    p = sub.add_parser("compare", help="two shas' latest rows, side by side")
    p.add_argument("sha_a")
    p.add_argument("sha_b")
    p.add_argument("--cells")

    p = sub.add_parser("baseline", help="the previous stage's recorded rows")
    p.add_argument("--stage", required=True)
    p.add_argument("--cell")
    p.add_argument("--metric")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--include-dirty", action="store_true")

    p = sub.add_parser("regressions", help="ratio check against a named baseline stage (§16 gate)")
    p.add_argument("--cell", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--baseline-stage", required=True)
    p.add_argument("--sha")
    p.add_argument("--threshold", type=float, default=1.10)

    p = sub.add_parser("sql", help="raw read-only SQL over the joined `measurements` view")
    p.add_argument("query")

    args = ap.parse_args()
    if args.cmd == "rebuild":
        return rebuild(args)
    return {
        "last": cmd_last, "history": cmd_history, "compare": cmd_compare,
        "baseline": cmd_baseline, "regressions": cmd_regressions, "sql": cmd_sql,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

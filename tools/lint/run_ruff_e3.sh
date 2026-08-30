#!/usr/bin/env bash
# RUFF E30x REPORT-ONLY PASS (LINT2 item 4, G_FOUNDATION G5, readability):
# pycodestyle's blank-line-spacing family (E301-E306), preview-gated in ruff
# 0.16.5 (see pyproject.toml's `[tool.rola_lint.report_only]` comment for the
# measurement). Reads the code list from THAT table rather than hardcoding
# "E3" a second time, so the two stay in one place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# A tiny, deliberately non-tomllib extraction (this hook runs under
# `language: system`, whatever `python3` is on PATH -- tomllib needs >=3.11,
# and pre-commit's own invoking shell is not guaranteed to have that; a real
# TOML parser would be overkill for one single-line array this file owns).
CODES=$(python3 -c "
import re
text = open('pyproject.toml').read()
m = re.search(r'ruff_preview_select\s*=\s*\[(.*?)\]', text, re.S)
codes = re.findall(r'\"([^\"]+)\"', m.group(1))
print(','.join(codes))
")

set +e
out=$(ruff check --preview --select "$CODES" --exit-zero --output-format=concise .)
set -e

n=$(printf '%s\n' "$out" | grep -cE ':[0-9]+:[0-9]+: E3[0-9]{2} ' || true)
echo "$out"
echo "run_ruff_e3.sh: $n finding(s) for codes [$CODES] (report-only; not part of the" \
     "gating ruff-check hook -- see pyproject.toml)"
exit 0

#!/usr/bin/env bash
# READABILITY clang-format REPORT-ONLY pass (LINT2 item 5, G_FOUNDATION G5):
# `clang-format --dry-run` against `tools/lint/clang-format-readability.yaml`
# (the root `.clang-format` plus `MaxEmptyLinesToKeep: 1` /
# `SeparateDefinitionBlocks: Always` -- see that file's header for why it is
# separate from the root config rather than an edit to it). Report-only;
# always exits 0.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

files=$(git ls-files 'csrc/rola/*.cu' 'csrc/rola/*.cuh' 'csrc/rola/*.cpp' \
                      'csrc/rola/*.h' 'csrc/rola/*.hpp')

total=0
for f in $files; do
  out=$(clang-format --style="file:tools/lint/clang-format-readability.yaml" \
          --dry-run --Werror "$f" 2>&1)
  n=$(printf '%s\n' "$out" | grep -c 'code should be clang-formatted' || true)
  if [[ "$n" -gt 0 ]]; then
    total=$((total + n))
    echo "== $f: $n diff(s) =="
  fi
done

echo "run_clang_format_readability.sh: $total diff(s) across $(echo "$files" | wc -l) file(s) " \
     "against MaxEmptyLinesToKeep:1 + SeparateDefinitionBlocks:Always " \
     "(report-only; the root .clang-format / clang-format-check hook is untouched)"
exit 0

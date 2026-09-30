#!/usr/bin/env bash
# DEVICE clang-tidy (gating): every device translation unit the build compiles -- `tools/kernels.py --units`, the one
# list the build and ratify read -- through clang's CUDA frontend (`--cuda-host-only`), under the checks in
# `tools/lint/clang-tidy-device.yaml` (clang-analyzer-deadcode.DeadStores, misc-unused-parameters, bugprone-unused-*,
# readability-redundant-*), with the PINNED clang-tidy (`tools/clang_tidy_pin.json`, `toolchain.clang_tidy`).
#
# THE PARSE IS PART OF THE GATE. A unit clang cannot parse is not a unit it analyzed: under the distro's clang 14 a
# torch-bearing unit's failed parse reported USED parameters as unused (measured 2026-09-29), which is why the pin is
# a release that parses every unit. Any compiler error, in our lines or a header's, fails the gate; so does an
# on-target finding. The toolkit's `include/cccl` root (thrust, cub, libcu++ under CUDA 13) is added because nvcc
# adds it implicitly and clang does not.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_PATH="$(python3 "$ROOT/tools/dev.py" get cuda-home)"
CLANG_TIDY="$(python3 "$ROOT/tools/dev.py" get toolchain.clang_tidy)"
CONFIG="$ROOT/tools/lint/clang-tidy-device.yaml"
if [[ -z "$CLANG_TIDY" || "$CLANG_TIDY" == "None" || ! -x "$CLANG_TIDY" ]]; then
  echo "run_clang_tidy_device.sh: FAILED -- no pinned clang-tidy (toolchain.clang_tidy); \`python tools/dev.py init\`" >&2
  exit 1
fi

if [[ "${1:-}" == "--test-fixtures" ]]; then
  # Self-test (excluded from the commit gate, K46 convention): the fixture
  # needs no compdb -- it is self-contained CUDA, no project headers.
  FIX="$ROOT/tools/lint/fixtures/device_lint_unused_param.cu"
  out=$("$CLANG_TIDY" "$FIX" --config-file="$CONFIG" -- \
          -x cuda --cuda-host-only --cuda-path="$CUDA_PATH" \
          --cuda-gpu-arch=sm_80 -ferror-limit=0 2>&1)
  if echo "$out" | grep -q "misc-unused-parameters.*bad\|bad(int used"; then :; fi
  if ! echo "$out" | grep -q "'unused_param' is unused \[misc-unused-parameters\]"; then
    echo "SELF-TEST FAILED: expected finding on fixture's 'bad' did not fire" >&2
    exit 1
  fi
  if echo "$out" | grep -q "'a' is unused\|'b' is unused"; then
    echo "SELF-TEST FAILED: fixture's 'ok' fired unexpectedly" >&2
    exit 1
  fi
  echo "run_clang_tidy_device.sh --test-fixtures: PASS"
  exit 0
fi

#: THE FILES CHANGED, when pre-commit passes them: only the units whose include closure holds one are re-read
#: (`tools/kernels.py --units --touching`); with none, every unit (CI, a manual sweep).
CC_JSON="$ROOT/compile_commands.json"
if [[ $# -gt 0 ]]; then
  wanted_units="$(python3 "$ROOT/tools/kernels.py" --units --touching "$@")"
  if [[ -z "$wanted_units" ]]; then
    echo "run_clang_tidy_device.sh: no device unit compiles the changed files ($*) -- nothing to re-read."
    exit 0
  fi
else
  wanted_units="$(python3 "$ROOT/tools/kernels.py" --units)"
fi

if [[ ! -f "$CC_JSON" || "$ROOT/build/temp/build.ninja" -nt "$CC_JSON" ]]; then
  if ! "$ROOT/tools/lint/compdb.sh" "$CC_JSON" >/dev/null 2>&1; then
    echo "run_clang_tidy_device.sh: FAILED -- no build/temp to read a compile database from; pre-commit is never" >&2
    echo "  green by absence (Blake, 2026-08-28). Build first: pip install -e . --no-build-isolation" >&2
    exit 1
  fi
fi

unit_files=$(python3 - "$CC_JSON" "$wanted_units" << 'PYEOF'
import json, sys
db = json.load(open(sys.argv[1]))
wanted = set(sys.argv[2].split())
for e in db:
    if e["file"].endswith(".cu") and "csrc/" + e["file"].split("/csrc/")[-1] in wanted:
        print(e["file"])
PYEOF
)

if [[ -z "$unit_files" ]]; then
  echo "run_clang_tidy_device.sh: FAILED -- no device unit of the build in the compile database; a gate with no" \
       "target does not pass." >&2
  exit 1
fi

ARCH="$(python3 -c "from rola_cu13._build_config import BUILD_CONFIG as c; print(c['archs'][0].replace('sm_',''))" 2>/dev/null || echo 80)"

total_real=0
total_errors=0
for src_from_db in $unit_files; do
  rel="${src_from_db#*/csrc/}"
  target="$ROOT/csrc/$rel"
  [[ -f "$target" ]] || { echo "run_clang_tidy_device.sh: $target missing, skipping" >&2; continue; }

  flags=$(python3 - "$CC_JSON" "$src_from_db" "$ROOT" "$ARCH" "$CUDA_PATH" << 'PYEOF'
import json, shlex, sys
cc_json, src, root, arch, cuda_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
db = json.load(open(cc_json))
entry = next(e for e in db if e["file"] == src)
old_root = entry["file"].split("/csrc/")[0]
cmd = entry["command"].replace(old_root, root)
toks = shlex.split(cmd)[1:]
skip_with_arg = {"-MF", "-o", "--dependency-output"}
drop_prefixes = ("--threads", "--ptxas-options", "-gencode", "-Xfatbin")
drop_bare = {"-MMD", "-c", "--generate-dependencies-with-compile", "-MP",
             "--expt-relaxed-constexpr", "-lineinfo", "-compress-all"}
out = []
skip_next = False
for t in toks:
    if skip_next:
        skip_next = False
        continue
    if t in skip_with_arg:
        skip_next = True
        continue
    if t in drop_bare:
        continue
    if t.startswith(drop_prefixes):
        continue
    if t == "--compiler-options":
        skip_next = True
        continue
    if t.endswith(".cu"):
        continue
    out.append(t)
out += ["-x", "cuda", "--cuda-host-only", f"--cuda-path={cuda_path}", "-isystem", f"{cuda_path}/include/cccl",
        f"--cuda-gpu-arch=sm_{arch}"]
print(" ".join(shlex.quote(x) for x in out))
PYEOF
)

  echo "== device clang-tidy: ${target#"$ROOT"/} (sm_$ARCH) =="
  # LOCKS brief item 1: one host-compute-budget slot per translation unit --
  # a `clang-tidy --cuda-host-only` pass is a full front-end parse per TU, as
  # CPU-heavy as a `cicc` compile, and previously drew from no shared pool at
  # all.
  # shellcheck disable=SC2086
  out=$(python3 -m rola_devtools.locks.host --slots 1 --label clang-tidy-tu -- \
          "$CLANG_TIDY" "$target" --config-file="$CONFIG" --header-filter='csrc/rola/.*' \
          -- $flags -ferror-limit=0 2>&1)
  real=$(echo "$out" | grep -E '^/.*csrc/rola/.*: warning: ' || true)
  errors=$(echo "$out" | grep -E ': (fatal )?error: ' || true)
  real_n=$(printf '%s\n' "$real" | grep -c . || true)
  errors_n=$(printf '%s\n' "$errors" | grep -c . || true)
  if [[ -n "$real" ]]; then echo "$real"; fi
  if [[ -n "$errors" ]]; then echo "$errors"; fi
  total_real=$((total_real + real_n))
  total_errors=$((total_errors + errors_n))
done

echo "---"
echo "device clang-tidy: $total_real on-target finding(s) and $total_errors compiler error(s) across" \
     "$(echo "$unit_files" | wc -l) unit(s)."
if [[ "$total_real" -gt 0 || "$total_errors" -gt 0 ]]; then exit 1; fi
exit 0

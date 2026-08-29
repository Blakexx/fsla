#!/usr/bin/env bash
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# docs/setup.md "pointer venv, the fast path" (K46 item 4), scripted: a new git
# worktree, its POINTER venv (shared base site-packages + this worktree's own
# in-tree editable build), in one call -- agents kept reverse-engineering this
# by hand from a sibling worktree's venv (K42, K45 SKILLS FEEDBACK).
#
# Usage:
#   ROLA_BASE_VENV=/path/to/shared/venv tools/build/new_worktree.sh <name> <sha>
#
# ROLA_BASE_VENV is REQUIRED (KERNEL_STANDARDS §18: a tool refuses rather than
# guesses) -- the shared venv every pointer venv borrows its site-packages from
# (torch, einops, entmax, ... installed there ONCE; this box's own copy lives at
# a path under $HOME, which is exactly why it is not hard-coded here). Read the
# two .pth files this script writes to see the pointer shape once it has run.
#
# ROLA_WORKTREES_DIR overrides where the new worktree and its venv are created;
# default is the sibling `worktrees/` directory next to this repo's own toplevel
# -- the convention every existing sibling worktree on this box already uses
# (`worktrees-live-in-one-folder`).
set -euo pipefail

usage() {
  echo "usage: ROLA_BASE_VENV=<path> tools/build/new_worktree.sh <name> <sha>" >&2
  exit 2
}

NAME="${1:-}"
SHA="${2:-}"
[[ -n "$NAME" && -n "$SHA" ]] || usage

if [[ -z "${ROLA_BASE_VENV:-}" ]]; then
  echo "refusing: ROLA_BASE_VENV is not set. Name the shared base venv (its" >&2
  echo "lib/python3.11/site-packages is what every pointer venv borrows) --" >&2
  echo "this script does not guess a home path (KERNEL_STANDARDS §18/§17)." >&2
  exit 1
fi
BASE_VENV="$ROLA_BASE_VENV"
if [[ ! -d "$BASE_VENV/lib/python3.11/site-packages" ]]; then
  echo "refusing: ROLA_BASE_VENV=$BASE_VENV has no lib/python3.11/site-packages" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `--show-toplevel` from INSIDE A LINKED WORKTREE returns that worktree's own
# root, not the main repo's -- e.g. /home/blake/worktrees/k31-formb3, which
# already sits directly inside the one shared worktrees folder
# (`worktrees-live-in-one-folder`). The default below is that folder itself,
# not a "/worktrees" appended a second time.
TOPLEVEL="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WORKTREES_DIR="${ROLA_WORKTREES_DIR:-$(dirname "$TOPLEVEL")}"
WORKTREE="$WORKTREES_DIR/$NAME"
VENV="$WORKTREES_DIR/venv-$NAME"

for p in "$WORKTREE" "$VENV"; do
  if [[ -e "$p" ]]; then
    echo "refusing: $p already exists" >&2
    exit 1
  fi
done

echo "[new_worktree] git worktree add $WORKTREE $SHA"
git -C "$TOPLEVEL" worktree add "$WORKTREE" "$SHA"

echo "[new_worktree] uv venv --python 3.11 --seed $VENV"
# FOOTGUN 1: a SEEDLESS venv (no --seed) carries no `pip` binary at all -- a later
# `pip install ...` (the bare command, not `python -m pip`) then silently resolves
# to whatever `pip` is first on PATH OUTSIDE the venv and installs into a
# DIFFERENT python entirely (`editable-finder-beats-pythonpath`'s family: the
# 6th-recurrence extension trap). `--seed` is not optional.
uv venv --python 3.11 --seed "$VENV"

SITE="$VENV/lib/python3.11/site-packages"
echo "[new_worktree] pointer .pth files"
# `aa_` sorts before `zz_` in site.py's alphabetical .pth processing, and the
# editable-install finder pip adds below sorts after both (`__editable__...pth`)
# -- so THIS worktree's own root, and then its own editable install, are always
# resolved before the shared base venv's copy of anything by the same name.
printf '%s\n' "$WORKTREE" > "$SITE/aa_rola_${NAME}.pth"
printf '%s\n' "$BASE_VENV/lib/python3.11/site-packages" > "$SITE/zz_rola_base.pth"

echo "[new_worktree] in-tree editable build (--no-deps, through the build lock)"
# FOOTGUN 2: `pip install -e .` WITHOUT --no-deps resolves pyproject.toml's
# `torch` dependency FRESH from PyPI -- a different (possibly CPU-only, or a
# different CUDA build) torch than the one the shared base venv already carries
# and the extension is about to compile its ABI against, yielding an
# ABI-mismatched `.so` at import time. The shared base venv IS the dependency
# resolution here, via the `zz_rola_base.pth` file above -- `--no-deps` is not
# optional either.
(
  cd "$WORKTREE"
  # The NEW worktree's own build_lock.sh, not this script's -- whatever SHA was
  # checked out is what governs its own build.
  ./tools/build/build_lock.sh -- \
    "$VENV/bin/python" -m pip install -e . --no-deps --no-build-isolation
)

echo "[new_worktree] verifying the import"
# THE FINDER'S OWN MAPPING, read statically first: does the generated editable
# finder actually name THIS worktree.
FINDER="$SITE/__editable___rola_0_1_0_dev0_finder.py"
if [[ ! -f "$FINDER" ]]; then
  echo "refusing: no editable finder at $FINDER -- the editable install did not run as expected" >&2
  exit 1
fi
if ! grep -qF "'rola': '$WORKTREE/rola'" "$FINDER"; then
  echo "refusing: $FINDER does not map 'rola' to $WORKTREE/rola:" >&2
  grep "MAPPING" "$FINDER" >&2
  exit 1
fi

# THE LIVE IMPORT, run FROM the worktree, not from wherever this script's own
# CALLER happened to be sitting. ROOT CAUSE (K46, found across four throwaway
# runs): `python -c` puts `''` -- meaning THE PROCESS's OWN CWD -- first in
# `sys.path`. The editable build's subshell above already `cd`s into
# `$WORKTREE`, so it was never affected; this verification step, running in
# the OUTER shell, was not, and inherited whatever directory the CALLER of
# this script happened to be in. Three real runs launched from inside another
# rola worktree (itself carrying a `rola/` directory) resolved `import rola`
# to THAT worktree instead of this new one, every time -- `''` beat every
# `.pth`-added path because it is `sys.path[0]`. `cd "$WORKTREE"` here makes
# `''` mean the same thing the intended answer does, closing the gap instead
# of working around it.
(
  cd "$WORKTREE"
  "$VENV/bin/python" -c "
from rola._build_config import BUILD_CONFIG
import rola
print('rola:', rola.__file__)
print('archs:', BUILD_CONFIG['archs'])
"
)

echo "[new_worktree] done: worktree=$WORKTREE venv=$VENV"

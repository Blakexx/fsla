#!/usr/bin/env bash
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# K39 (Blake ruling, 2026-08-28): ONE build-lock discipline. This replaces both
# of the previous two mechanisms -- an out-of-repo build-slot script's N
# parallel iteration slots, and the separate EXCLUSIVE `/tmp/rola_ram.lock`
# gate build lock (KERNEL_STANDARDS §15 addendum) -- which used to be able to
# run at the same time as each other because they were two unrelated lock
# files. That is exactly the "three concurrent in-container probes turned a
# minutes-long gate into an hour" shape (K37's own §14 addendum finding, one
# level up: an ITERATION build racing a GATE build for the same cicc/RAM
# budget can OOM the host, not just slow it down.
#
# ONE mechanism now: N slot lock files (`/tmp/rola_build_slot_<i>.lock`,
# `ROLA_BUILD_SLOTS` count, default 2, the same default `build_slot.sh` used).
# An ITERATION build takes exactly ONE free slot (or waits on slot 0, as
# before). A GATE build (`--gate`) takes EVERY slot, in order, before running
# -- so a gate build cannot start while any iteration build holds a slot, and
# no iteration build can start while a gate build is running, with no second
# lock path for the two to silently disagree about.
#
# Usage:
#   tools/build/build_lock.sh [--gate] -- <command...>
#
# `docs/build.md#compile-ledger` and the `rola-build` skill point here now;
# the old out-of-repo build-slot script is superseded (§17: a tool the repo
# depends on lives in the repo, not in a scratch sibling).
set -euo pipefail

N=${ROLA_BUILD_SLOTS:-2}
GATE=0
if [[ "${1:-}" == "--gate" ]]; then
  GATE=1
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--gate] -- <command...>" >&2
  exit 2
fi

if [[ $GATE -eq 1 ]]; then
  echo "[build_lock] gate build: acquiring all $N slot(s) ..." >&2
  fds=()
  for i in $(seq 0 $((N - 1))); do
    exec {fd}>"/tmp/rola_build_slot_${i}.lock"
    flock "$fd"
    fds+=("$fd")
    echo "[build_lock] gate build: holding slot $i" >&2
  done
  echo "[build_lock] gate build: all $N slot(s) held, running" >&2
  # fds stay open (and therefore locked) for the lifetime of this process;
  # they close automatically when it exits, releasing every slot at once.
  exec "$@"
fi

for i in $(seq 0 $((N - 1))); do
  exec {fd}>"/tmp/rola_build_slot_${i}.lock"
  if flock -n "$fd"; then
    echo "[build_lock] iteration build: slot $i" >&2
    exec "$@"
  fi
  exec {fd}>&-
done
exec {fd}>"/tmp/rola_build_slot_0.lock"
flock "$fd"
echo "[build_lock] iteration build: slot 0 (waited)" >&2
exec "$@"

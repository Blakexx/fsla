# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S RUNNER: this checkout's bench subjects as timed arms, built on central cells for the timing system.

`arms(cell)` takes a central carry cell (a routing, a draw and the state it binds) or a central layer cell (an input,
and the routing and gain its layer is built with) and returns a builder for every arm this checkout runs on it, or
refuses the cell by raising, always by a FACT of the cell or the binary and never by the cell's name:
- any cell, when the binary is an iteration build (its build record's `iteration` mark: a narrowed `--//:contract`, or architectures
  other than the built set), or lacks an arm its tree declares (`arms.bzl`, as the graph generated its table): a narrowed binary
  carries a subset of the tree, and its own generated table is that subset, so the record is what refuses it;
- a cell of another kind.
`measure/executors.py`'s timing entry calls it on the cell a registration hands it.
An arm is a subject that applies to the cell (`measure.subjects.applicable`, the cell's own facts) and whose kernel this
binary carries at the cell's shape (the carry arm the prefill launch function selects for the cell's call for `prefill`, the decode arm for
`decode_step`), with its call count: its name is the subject, then `@calls=N` for a call count other than one
(`Subject.calls`). Only the arm asked for is built.
Building one proves two things before anything is timed, and refuses the arm by name when either fails: from a fact the
device produces, that this binary carries the subject's family (its stamp entry; a path or a hash cannot catch a stale
binary), and that `import fsla` resolved inside this checkout.

A built arm is a `fsla_devtools.timing.Timed`: its call times one launch between two CUDA events (the canonical
instrument, `cuda_events`) and returns milliseconds; its reset is the launch's (a carried state restored, a decode state
re-seeded), run untimed before every call. What an arm reports is the cell's facts beside the arm's call count and the
binary's: the family stamp, the device, torch, the assembler, and the SM clock read when the arm
was built (docs/measurement.md).
"""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

#: this tree -- a checkout, or a timing entry's bundle -- as it was imported, its links unresolved (a bundle's runfiles
#: are links into the checkout, and what it reads is the bundle's): `import fsla` must resolve inside it
CHECKOUT = Path(os.path.abspath(__file__)).parents[1]


def arm_name(subject: str, *, calls: int = 1) -> str:
    """An arm's name: the subject, then a call count other than one."""
    return subject + (f"@calls={calls}" if calls != 1 else "")


def arms(data) -> dict:
    """Every arm this checkout runs on a cell's data, each as a builder; raises to refuse the cell."""
    from measure.subjects import SUBJECTS, applicable

    _refuse_an_iteration_binary()
    _refuse_a_partial_binary()
    if data.kind not in ("carry", "layer"):
        raise TypeError(f"fsla's runner takes a central carry cell or layer cell, got a {data.kind} cell")
    out = {}
    for subject in SUBJECTS.values():
        for calls in subject.calls:
            if subject.name in applicable(data, calls) and _kernel_carried(subject.name, data):
                out[arm_name(subject.name, calls=calls)] = partial(_build, subject.name, data, calls)
    return out


def _declared(family: str) -> tuple:
    """The arms this tree declares for `family`: the table the graph generated from the declaration
    (`bazel/arms.bzl`'s `arm_table`), which `bazel run //:library` installs into a checkout's `build/generated` and a
    timing entry's bundle lays out at the same path (`bazel/defs.bzl`'s `timing_entry`)."""
    import importlib.util

    path = CHECKOUT / "build" / "generated" / f"{family}_arms.py"
    if not path.is_file():
        raise RuntimeError(f"{path.relative_to(CHECKOUT)} is missing: the {family} family's declared arms are the graph's "
                           "table; build the library (`bazel run //:library`)")
    spec = importlib.util.spec_from_file_location(f"_{family}_arms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ARMS


def _declared_families() -> list[str]:
    """The families this tree declares: `ARM_FAMILIES` in `bazel/arms.bzl`, the one registry (a literal read by AST), each
    answering what the binary carries through its own op (`fsla_<family>_arms`)."""
    from tools.arm_families import arm_families

    return sorted(arm_families(CHECKOUT))


def _build_config() -> dict:
    """The binary's own build record (`fsla_cu13/_build_config.py`, written by the build); empty when this binary has none."""
    try:
        from fsla_cu13._build_config import BUILD_CONFIG
    except ImportError:
        return {}
    return BUILD_CONFIG


def _refuse_an_iteration_binary() -> None:
    """A binary built under a non-default `--//:contract` (or other than the built architectures) is refused: its generated
    arm table is its own narrowed one, so the declared-against-built comparison below cannot see that it is partial."""
    record = _build_config()
    if not record:
        raise RuntimeError("this binary carries no build record (fsla_cu13/_build_config.py), so it cannot state that it was built "
                           "under the full contract; build the library (`bazel run //:library`)")
    if record.get("iteration", True):
        raise RuntimeError("this binary is an iteration build (its build record's `iteration` is set: a narrowed --//:contract or "
                           "other architectures than the built set), a subset of the tree; measure a build under the full contract "
                           "(`bazel run //:library`, no --//:contract)")


def _carried(family: str) -> tuple:
    """The arms THIS BINARY carries for `family`, as its own op reports them (`fsla_<family>_arms`, the op the build's arm gate asks)."""
    from fsla.ops._ext import extension

    return tuple(tuple(int(x) for x in row) for row in getattr(extension(), f"fsla_{family}_arms")())


def _refuse_a_partial_binary() -> None:
    for family in _declared_families():
        declared = {tuple(int(x) for x in row) for row in _declared(family)}
        built = {tuple(int(x) for x in arm) for arm in _carried(family)}
        if declared - built:
            raise RuntimeError(f"this binary carries the {family} arms {sorted(built)} and lacks the declared arms "
                               f"{sorted(declared - built)}: a binary that lacks arms this tree declares (a stale or "
                               "hand-assembled build) measures a subset of the tree; rebuild it (`bazel run //:library`)")


def _kernel_carried(subject: str, spec) -> bool:
    """Whether this binary carries the kernel `subject` launches at the cell's shape (the cell's own facts are
    `measure.subjects.applicable`'s)."""
    from fsla.ops import decode

    #: the carry is not built on this line (`fsla.ops.carry`): no arm of it is carried, so the op's prefill is never offered
    if subject == "prefill":
        return False
    if subject == "decode_step":
        rows = decode.arms()
        padded = [dv for dv, _d, _decay in rows if dv >= spec.dv]
        return bool(padded) and (min(padded), spec.routing.D, False) in rows
    return True


def _build(name: str, spec, calls: int):
    import torch
    from fsla_devtools.timing import Timed

    import fsla
    from fsla.ops import carry as carry_ops
    from fsla.ops._ext import extension
    from hardware import registry
    from measure.subjects import SUBJECTS

    subject = SUBJECTS[name]
    stamp = getattr(extension(), subject.stamp, None)
    if stamp is None:
        raise RuntimeError(f"this binary carries no {subject.stamp} entry, so it cannot state that it has {name}'s family")
    fsla_file = Path(os.path.abspath(fsla.__file__))
    if not fsla_file.is_relative_to(CHECKOUT):
        raise RuntimeError(f"`import fsla` resolved outside this checkout ({CHECKOUT.name}); the arm would time another "
                           "tree's binary under this tree's name")
    fx = {"cell": spec} if spec.kind == "carry" else _layer_fixture(spec)
    fx.update(calls=calls)
    launch = subject.build(fx)
    BUILD_CONFIG = _build_config()
    props = torch.cuda.get_device_properties(0)

    def call() -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        launch.call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    cell = {"cell": spec.name, "kind": spec.kind, "subject": name, "level": subject.level, "calls": calls,
            "state": spec.state, "backing": spec.backing, "tokens": spec.tokens, "d_v": spec.dv,
            "widths": list(spec.routing.topology),
            "ptxas": BUILD_CONFIG.get("ptxas"),
            "family_stamp": int(stamp()), "device": torch.cuda.get_device_name(0),
            "sm": registry.device_arch(props.major, props.minor),
            "torch": torch.__version__, "fsla_file": fsla_file.relative_to(CHECKOUT).as_posix(),
            "sm_ghz_at_build": carry_ops.sm_clock_ghz()}
    return Timed(call=call, built=cell, instrument="cuda_events", reset=launch.reset,
                 outside_allocator=launch.outside_allocator)


def _layer_fixture(spec):
    from measure.cells.layer import build

    return build(spec)

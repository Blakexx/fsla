# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S RUNNER: this checkout's bench subjects as timed arms, built on central cells for the timing system.

`arms(cell)` takes a central carry cell (a shape, a draw and the state it binds) or a central layer cell (an input)
and returns a builder for every arm this checkout runs on it, or refuses the cell by raising:
- any cell, when the binary is an iteration build (its build record's `iteration` mark: a narrowed `--//:contract`, or architectures
  other than the built set), or lacks an arm its tree declares (`arms.bzl`, as the graph generated its table): a narrowed binary
  carries a subset of the tree, and its own generated table is that subset, so the record is what refuses it;
- data that is neither kind.
`measure/executors.py`'s timing entry calls it on the cell a registration hands it; a profiler launches one arm
through `oneshot_argv`.
An arm is a subject that applies to the cell (`measure.subjects.applicable`, the cell's own facts) and whose kernel this
binary carries at the cell's shape (the carry arm the cell's carve selects for `prefill`, the decode arm for
`decode_step`), with its call count: its name is the subject, then `@calls=N` for a call count other than one
(`Subject.calls`), and on a layer cell `@layer=C` for each FSLA construction declared for that input
(`measure.cells.layer.CONSTRUCTIONS`). Only the arm asked for is built.
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

import sys
from functools import partial
from pathlib import Path

#: this checkout: `import fsla` must resolve inside it
CHECKOUT = Path(__file__).resolve().parents[1]


def arm_name(subject: str, *, calls: int = 1, construction: str | None = None) -> str:
    """An arm's name: the subject, then a call count other than one, then a layer arm's construction."""
    return subject + (f"@calls={calls}" if calls != 1 else "") + (f"@layer={construction}" if construction else "")


def arms(data) -> dict:
    """Every arm this checkout runs on a cell's data, each as a builder; raises to refuse the cell."""
    from itertools import product

    from fsla_devtools.cells.carry import CarryCell
    from fsla_devtools.cells.layer import LayerCell

    from measure.cells.layer import CONSTRUCTIONS
    from measure.subjects import SUBJECTS, applicable

    _refuse_an_iteration_binary()
    _refuse_a_partial_binary()
    if isinstance(data, CarryCell):
        kind, constructions = "carry", [None]
    elif isinstance(data, LayerCell):
        kind, constructions = "layer", [c for c in CONSTRUCTIONS.values() if data.name in c.cells]
        if not constructions:
            raise LookupError(f"{data.name}: no FSLA construction is declared for this layer cell")
    else:
        raise TypeError(f"fsla's runner takes a central carry cell or layer cell, got {type(data).__name__}")
    out = {}
    for subject in SUBJECTS.values():
        for construction, calls in product(constructions, subject.calls):
            if subject.name in applicable(data, kind, calls) and _kernel_carried(subject.name, data, construction):
                name = arm_name(subject.name, calls=calls, construction=construction and construction.name)
                out[name] = partial(_build, subject.name, kind, data, calls, construction)
    return out


def _declared(family: str) -> tuple:
    """The arms this tree declares for `family`: the table the graph generated from the declaration
    (`bazel/arms.bzl`'s `arm_table`), which `bazel run //:library` installs into `build/generated`."""
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
    answering what the binary carries through its op module (`fsla.ops.<family>`)."""
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


def _refuse_a_partial_binary() -> None:
    import importlib

    for family in _declared_families():
        declared = {tuple(int(x) for x in row) for row in _declared(family)}
        built = {tuple(int(x) for x in arm) for arm in importlib.import_module(f"fsla.ops.{family}").arms()}
        if declared - built:
            raise RuntimeError(f"this binary carries the {family} arms {sorted(built)} and lacks the declared arms "
                               f"{sorted(declared - built)}: an iteration build (--//:contract) measures a subset of the "
                               "tree; build the full contract")


def _kernel_carried(subject: str, spec, construction) -> bool:
    """Whether this binary carries the kernel `subject` launches at the cell's shape, or a layer cell's construction
    (the cell's own facts are `measure.subjects.applicable`'s)."""
    from fsla.ops import decode

    #: the carry is not built on this line (`fsla.ops.carry`): no arm of it is carried, so the op's prefill is never offered
    if subject == "prefill":
        return False
    if subject == "decode_step":
        rows = decode.arms()
        padded = [dv for dv, _d, _decay in rows if dv >= spec.dv]
        return bool(padded) and (min(padded), len(construction.widths), False) in rows
    return True


def _build(name: str, kind: str, spec, calls: int, construction):
    import torch
    from fsla_devtools.timing import Timed

    import fsla
    from fsla.ops import carry as carry_ops
    from fsla.ops._ext import extension
    from measure import hardware
    from measure.subjects import SUBJECTS

    subject = SUBJECTS[name]
    stamp = getattr(extension(), subject.stamp, None)
    if stamp is None:
        raise RuntimeError(f"this binary carries no {subject.stamp} entry, so it cannot state that it has {name}'s family")
    fsla_file = Path(fsla.__file__).resolve()
    if not fsla_file.is_relative_to(CHECKOUT):
        raise RuntimeError(f"`import fsla` resolved outside this checkout ({CHECKOUT.name}); the arm would time another "
                           "tree's binary under this tree's name")
    fx = {"cell": spec} if kind == "carry" else _layer_fixture(spec, construction)
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

    widths = spec.widths if kind == "carry" else construction.widths
    cell = {"cell": spec.name, "kind": kind, "subject": name, "level": subject.level, "calls": calls,
            "state": getattr(spec, "state", None), "backing": getattr(spec, "backing", None),
            "construction": construction and construction.name, "tokens": spec.tokens, "d_v": spec.dv,
            "widths": list(widths),
            "ptxas": BUILD_CONFIG.get("ptxas"),
            "family_stamp": int(stamp()), "device": torch.cuda.get_device_name(0),
            "sm": hardware.device_arch(props.major, props.minor),
            "torch": torch.__version__, "fsla_file": fsla_file.relative_to(CHECKOUT).as_posix(),
            "sm_ghz_at_build": carry_ops.sm_clock_ghz()}
    return Timed(call=call, built=cell, instrument="cuda_events", reset=launch.reset,
                 outside_allocator=launch.outside_allocator)


def _layer_fixture(spec, construction):
    from measure.cells.layer import build

    return build(spec, construction)


def oneshot_argv(cell: str, arm: str) -> list[str]:
    """One untimed launch of `arm` on `cell` by this checkout's runner, under this interpreter: what a profiler wraps.
    The arm is built exactly as a comparison builds it, so what a profile counts is the launch a comparison times."""
    #: AS A MODULE, from the checkout: a package file run by path has its own directory for a root and cannot import
    #: its package (the `measure/` rename turned every profiled launch into `No module named 'measure'`)
    return [sys.executable, "-m", "measure.provider", "--oneshot", cell, arm]


def _oneshot(cell: str, arm: str) -> None:
    import torch
    from fsla_devtools.cells import central

    from measure.cells import by_name

    if cell not in central().cells:
        raise SystemExit(f"{cell!r} is not a central cell (fsla_devtools.cells)")
    offered = arms(by_name(cell))
    if arm not in offered:
        raise SystemExit(f"{cell}: this checkout runs no arm {arm}; it runs {sorted(offered)}")
    offered[arm]().call()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--oneshot":
        raise SystemExit("usage: provider.py --oneshot CELL ARM")
    sys.path[:0] = [str(CHECKOUT)]
    _oneshot(sys.argv[2], sys.argv[3])

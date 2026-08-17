# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity capture across a refactor that claims to move no bytes.

    python tools/chunk_identity_capture.py capture ref.pt
    python tools/chunk_identity_capture.py verify  ref.pt

The two commands run the SAME cells through :func:`rola.rola_op` and compare with
`torch.equal`, so a refactor of the host path is proved neutral against the tree it
replaced rather than against a tolerance. `y` is the ONE exception and only where the
topology spans more than one owner block: its fan-in is an fp32 atomic reduction over
the owners, so it is not bit-reproducible run to run and is held to the reassociation
bound instead (the same derivation as
`tests/oracle/test_chunk_final_state.py::_reordering_bound`), which is why the cells
hand the op an fp32 `v`: the divide then returns the fp32 accumulator undowncast and
the bound is the one that was derived, with no bf16 rounding folded into it. The state plane, the
union table and the two bitmaps are exact everywhere -- every atom is stored by
exactly one CTA and the bitmaps are bit ballots. The facts are captured by recording what
the extension RETURNED during the call -- the op's own union table, block bitmap and
atom bitmap, not a second derivation of them -- which is what makes the comparison
independent of whatever host structure produced the launch arguments
(docs/internals/facts/facts.md).
"""
from __future__ import annotations

import argparse
import sys

import torch


class _Recorder:
    """The extension module, recording every tensor the FACT entries returned."""

    _WATCHED = ("chunk_stats", "chunk_union_table", "chunk_block_bits")

    def __init__(self, module):
        self._module = module
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        entry = getattr(self._module, name)
        if name not in self._WATCHED:
            return entry

        def recorded(*args, **kwargs):
            out = entry(*args, **kwargs)
            self.calls.append((name, out if isinstance(out, tuple) else (out,)))
            return out

        return recorded


def _install_recorder():
    from rola.ops import _ext

    recorder = _Recorder(_ext.extension())
    _ext.extension.cache_clear()
    _ext._rola_cuda = recorder
    return recorder


def _cells():
    """`(name, widths, routing, hidden, heads, tokens, paging)` -- the pinned set."""
    from rola.routing.producer import dense_routing, union_routing

    for widths, hidden, heads in (((64,), 128, 2), ((64, 64), 128, 2), ((16, 16), 64, 2)):
        for routing_name, routing in (("dense", dense_routing(widths[0])),
                                      ("union", union_routing(widths[0], 1.5))):
            for mode in ("stateless", "paged", "dense_state"):
                yield (f"{routing_name}-{'x'.join(map(str, widths))}-{mode}",
                       widths, routing, hidden, heads, 128, mode)


def _run_cell(cell) -> dict[str, torch.Tensor]:
    import rola
    from rola.expert import PlanOverrides
    from rola.routing.producer import RouteProducer

    name, widths, routing, hidden, heads, tokens, mode = cell
    torch.manual_seed(20260816)
    producer = RouteProducer([routing.at(w) for w in widths], hidden_size=hidden,
                             num_heads=heads, normalization="global").cuda()
    x = torch.randn(1, tokens, hidden, device="cuda")
    v = torch.randn(1, tokens, heads, 64, device="cuda")
    with torch.no_grad():
        routes = producer(x)

    state = None if mode == "stateless" else rola.state()
    expert = PlanOverrides(paging=False) if mode == "dense_state" else None

    recorder = _install_recorder()
    with torch.no_grad():
        y, state = rola.rola_op(routes, v, state=state, expert=expert)
    torch.cuda.synchronize()

    from math import prod

    from rola.ops.chunk import arm as chunk_arm

    out = {f"{name}/y": y,
           f"{name}/owners": torch.tensor(prod(widths) // chunk_arm(routes)[1])}
    if state is not None:
        out[f"{name}/state"] = state.materialize()
    for index, (entry, tensors) in enumerate(recorder.calls):
        for slot, tensor in enumerate(tensors):
            out[f"{name}/{index}.{entry}.{slot}"] = tensor
    return out


def _capture() -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    for cell in _cells():
        captured.update({k: v.detach().cpu() for k, v in _run_cell(cell).items()})
    return captured


def _agrees(key: str, reference: dict, captured: dict) -> bool:
    got, want = captured[key], reference[key]
    owners = int(reference.get(key.rsplit("/", 1)[0] + "/owners", torch.tensor(1)))
    if not key.endswith("/y") or owners == 1:
        return torch.equal(got, want)
    #: one output element is a sum of at most `owners` fp32 partials, and reassociating
    #: an `n`-term fp32 sum perturbs it by at most `(n-1) * eps * sum|terms|`; bounding
    #: `sum|terms|` by `n * max|y|` gives `n * n * eps * max|y|`.
    bound = owners * owners * torch.finfo(torch.float32).eps * want.abs().max()
    return bool((got.float() - want.float()).abs().max() <= bound)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("path")
    args = parser.parse_args(argv)

    captured = _capture()
    if args.command == "capture":
        torch.save(captured, args.path)
        print(f"captured {len(captured)} tensors to {args.path}")
        return 0

    reference = torch.load(args.path, weights_only=True)
    missing = sorted(set(reference) ^ set(captured))
    differ = sorted(k for k in reference.keys() & captured.keys()
                    if not _agrees(k, reference, captured))
    for key in missing:
        print(f"MISSING {key}")
    for key in differ:
        print(f"DIFFERS {key}")
    if missing or differ:
        print(f"FAIL: {len(missing)} missing, {len(differ)} differing")
        return 1
    print(f"OK: {len(reference)} tensors bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())

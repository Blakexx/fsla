# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GROUP COST MODEL: per warp and window, what a sparse side's step form costs when the warp's 4 digits step in
aligned groups of g (a group's tokens the union of its digits' lists, every digit's MMAs over that union; g = 1 per
list, g = 4 merged), as the MAX of its components -- tensor cycles and issue a scheduler, shared-memory cycles the SM --
for the even carve's readout (fp16 accumulate, den out of the readout) and fold (fp32 accumulate, the mass out of the
fold). The counts a step (issue, shared wavefronts) come from the `read_step` and `list_fold` lab records' ncu counters
(docs/internals/tools/lab.md#read-step, #list-fold), the rates (an MMA, a load wavefront, a shared atomic) from
`common/hardware.json` for `--product` (this device's by default) or `--arch` (its defaults). Then the probe cells' own lists priced under fixed g, the best
of {1, 4} a warp (today's readout choice), the best of {1, 2, 4} a warp, and the best combination over a CTA's warps.

    python -m measure.group_cost [--product rtx3080ti | --arch sm_80]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from measure import hardware

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "measure" / "lab" / "results"
WARPS, SCHEDULERS, DIGITS, STEP, WINDOW, SLICE = 8, 4, 4, 16, 512, 32
#: the labs' launch: a CTA an SM, 400 passes
PASSES = 400
#: MMAs a digit a step on the even carve (32 x 32 of a digit: two 16-row or two k16 halves x four 8-channel n-tiles)
MMAS = 8
GROUPS = (1, 2, 4)
#: a panel add's atomic instructions: 16 tokens x 32 channels as s16 pairs over 32 lanes
ATOMICS_AN_ADD = 8


@dataclass(frozen=True)
class Phase:
    """A phase's step costs: issue a group step `step + digit x g`, shared-memory cycles a group step `smem`, tensor
    cycles an MMA `mma`."""
    name: str
    step: float
    digit: float
    smem: float
    mma: int


def _records(lab: str) -> tuple[dict, dict]:
    counters = json.loads((RESULTS / lab / "sm_86.counters.json").read_text())["kernels"]
    timing = {r["name"]: r["median"] for r in json.loads((RESULTS / lab / "sm_86.json").read_text())["results"]}
    return counters, timing


def _per_warp_pass(counters: dict, kernel: str, metric: str, sms: int) -> float:
    (key,) = [k for k in counters if k.startswith(f"void {kernel}(")]
    return counters[key][metric] / (sms * WARPS * PASSES)


INST, WAVES = "smsp__inst_executed.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"


def calibrate(sms: int, rates: dict[str, hardware.Rate]) -> tuple[Phase, Phase, list[tuple[str, float, float]]]:
    """The two phases' constants at `rates` (`hardware.resolve`) and the labs' rows as (row, measured,
    max-of-components model) cycles a pass."""
    f16_acc, f32_acc = rates["mma.hmma_f16_f16"].cycles, rates["mma.hmma_f16_f32"].cycles

    def shared(loads: float, atomics: float, atomic_waves: float) -> float:
        return (loads * rates["wavefront"].cycles
                + max(atomics * rates["atomic"].cycles, atomic_waves * rates["atomic_wavefront"].cycles))

    rc, rt = _records("read_step")
    fc, ft = _records("list_fold")

    def read(kind, mode, form, steps, live):
        return _per_warp_pass(rc, f"read<{mode}, {form}, {steps}, {live}>", kind, sms)

    def fold(kind, form, steps, live, gather):
        return _per_warp_pass(fc, f"fold<{form}, {steps}, {live}, {gather}>", kind, sms)

    #: the readout: per digit a gathered digit step (8 steps a digit less 2, 4 digits); merged a 4-digit step
    per_digit = (read(INST, 0, 0, 8, 16) - read(INST, 0, 0, 2, 16)) / 24
    merged = read(INST, 0, 1, 8, 16) / 8
    digit = (merged - per_digit) / 3
    adds = (read(INST, 1, 0, 8, 16) - read(INST, 0, 0, 8, 16)) / 32
    ld = (read(WAVES, 0, 0, 8, 16) - read(WAVES, 0, 0, 2, 16)) / 24
    atom = (read(WAVES, 1, 0, 8, 16) - read(WAVES, 0, 0, 8, 16)) / 32
    #: an add's atomic instructions with their wavefronts; den's two MMAs a digit leave the readout
    readout = Phase("readout", per_digit - digit + adds, digit - 2, shared(ld, ATOMICS_AN_ADD, atom), f16_acc)

    #: the fold: per list a gathered list step (8 live, 2 steps less 1, 4 lists); merged a 4-list step; the mass's two
    #: MMAs a step leave the fold
    per_list = (fold(INST, 1, 2, 8, 1) - fold(INST, 1, 1, 8, 1)) / 4
    merged_f = fold(INST, 2, 8, 16, 0) / 8
    digit_f = (merged_f - per_list) / 3
    fold_phase = Phase("fold", per_list - digit_f - 2, digit_f,
                       fold(WAVES, 1, 2, 16, 1) / 8 * rates["wavefront"].cycles, f32_acc)

    rows = []
    for form, steps, live, label in ((1, 8, 16, "merged / 8 steps / 16 live"),
                                     (0, 8, 16, "per digit / 8 steps a digit / 16 live"),
                                     (0, 2, 16, "per digit / 2 steps a digit / 16 live"),
                                     (0, 1, 8, "per digit / 1 steps a digit / 8 live")):
        mmas = (8 if form else 4 * steps) * (40 if form else 10)
        adds = 8 if form else 4 * steps
        for mode, tail in ((0, "steps alone"), (1, "+ panel adds")):
            atoms = read(WAVES, mode, form, steps, live) - read(WAVES, 0, form, steps, live)
            smem = WARPS * shared(read(WAVES, 0, form, steps, live), mode * adds * ATOMICS_AN_ADD, atoms)
            model = max(2 * mmas * f16_acc, 2 * read(INST, mode, form, steps, live), smem)
            rows.append((f"read_step {label} / {tail}", rt[f"{label} / {tail}"], model))
    names = ("merged, 8 lists x 16 x 32", "per list, 4 lists x 32 x 32", "merged, 4 lists x 32 x 32",
             "per list, 8 lists x 16 x 32")
    for form, steps, live, gather in ((2, 8, 16, 0), (1, 8, 16, 0), (0, 8, 16, 0), (3, 8, 16, 0), (1, 2, 16, 1),
                                      (1, 2, 8, 1), (1, 1, 8, 1), (3, 2, 16, 1), (3, 1, 8, 1)):
        lists, mtiles = (4, 2) if form in (1, 2) else (8, 1)
        mmas = steps * (lists * mtiles * 4 + mtiles) if form in (0, 2) else lists * steps * (mtiles * 5)
        label = (f"{names[form]} / {steps} steps{'' if form in (0, 2) else ' a list'} / {live} live / "
                 f"{'gathered' if gather else 'contiguous'}")
        model = max(2 * mmas * f32_acc, 2 * fold(INST, form, steps, live, gather),
                    WARPS * fold(WAVES, form, steps, live, gather) * rates["wavefront"].cycles)
        rows.append((f"list_fold {label}", ft[label], model))
    return readout, fold_phase, rows


def lists(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The cell's sparse sides' live masks `[tokens, digits]`: level S's reads (the readout's lists) and level R's
    writes (the fold's)."""
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(name)
    routes, _, _, statics = bundle_of(spec, realize(spec, H=1))
    level_r = facts_ops.list_level(statics[0])
    read = (routes.amp_read[1 - level_r][0, :, 0, :] != 0).cpu().numpy()
    write = (routes.amp_write[level_r][0, :, 0, :] != 0).cpu().numpy()
    del routes
    torch.cuda.empty_cache()
    return read, write


def steps_by_group(live: np.ndarray) -> dict[int, np.ndarray]:
    """For each g, `[windows, slices, warps, groups]` steps: a group's union of live tokens a window over 16."""
    tokens, digits = live.shape
    w = live[: tokens // WINDOW * WINDOW].reshape(tokens // WINDOW, WINDOW, digits // SLICE, WARPS, DIGITS)
    out = {}
    for g in GROUPS:
        union = w.reshape(*w.shape[:4], DIGITS // g, g).any(axis=-1).sum(axis=1)
        out[g] = np.ceil(union / STEP).astype(np.int64)
    return out


def components(phase: Phase, steps: np.ndarray, g: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`[windows, slices, warps]` tensor and issue cycles (the warp's, toward its scheduler) and shared-memory cycles
    (toward the SM) of the warp stepping its groups of g."""
    s = steps.sum(axis=-1)
    return s * g * MMAS * phase.mma, s * (phase.step + phase.digit * g), s * phase.smem


def cta_cost(tensor: np.ndarray, issue: np.ndarray, smem: np.ndarray) -> np.ndarray:
    """`[windows, slices]` the max of the busiest scheduler's tensor and issue and the SM's shared memory (warp k on
    scheduler k mod 4)."""
    def sched(x):
        return x.reshape(*x.shape[:2], WARPS // SCHEDULERS, SCHEDULERS).sum(axis=2).max(axis=-1)

    return np.maximum(np.maximum(sched(tensor), sched(issue)), smem.sum(axis=-1))


def price(phase: Phase, live: np.ndarray) -> dict[str, tuple[float, dict[int, float]]]:
    """Each strategy's total CTA-window cycles and its share of warp-windows at each g."""
    by_g = steps_by_group(live)
    comp = {g: components(phase, by_g[g], g) for g in GROUPS}
    #: a warp's own objective: its share of each resource as if every warp were like it
    own = {g: np.maximum(np.maximum(comp[g][0], comp[g][1]), SCHEDULERS * comp[g][2]) for g in GROUPS}
    out = {}

    def choose(allowed):
        pick = np.stack([own[g] for g in allowed]).argmin(axis=0)
        chosen = [np.choose(pick, [comp[g][i] for g in allowed]) for i in range(3)]
        share = {g: float((pick == k).mean()) for k, g in enumerate(allowed)}
        return float(cta_cost(*chosen).sum()), share

    for g in GROUPS:
        out[f"g{g}"] = (float(cta_cost(*comp[g]).sum()), {g: 1.0})
    out["best {1,4}"] = choose((1, 4))
    out["best {1,2,4}"] = choose((1, 2, 4))
    #: the best combination over a CTA's 8 warps (3^8 a CTA-window)
    best = None
    for combo in product(range(len(GROUPS)), repeat=WARPS):
        parts = [np.stack([comp[GROUPS[c]][i][..., k] for k, c in enumerate(combo)], axis=-1) for i in range(3)]
        cost = cta_cost(*parts)
        best = cost if best is None else np.minimum(best, cost)
    out["best a CTA"] = (float(best.sum()), {})
    return out


def main() -> None:
    import sys

    import torch

    device = torch.cuda.get_device_properties(0)
    arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else None
    product = (sys.argv[sys.argv.index("--product") + 1] if "--product" in sys.argv
               else None if arch else hardware.product_of(device.name))
    if arch is None and product is None:
        raise SystemExit(f"{device.name} has no product row in common/hardware.json; name --product or --arch")
    readout, fold_phase, rows = calibrate(device.multi_processor_count, hardware.resolve(arch, product))
    print(f"constants ({product or arch}'s rates, the RTX 3080 Ti's counts): issue a group step = step + digit x g; "
          "shared-memory cycles a group step")
    for p in (readout, fold_phase):
        print(f"  {p.name:8s} step {p.step:6.1f}  digit {p.digit:5.1f}  smem {p.smem:6.1f}  MMA {p.mma} cycles")
    print("\nthe labs: measured against the max of components (cycles a pass)")
    for label, measured, model in rows:
        print(f"  {label:78s} {measured:8.0f} {model:8.0f}  {measured / model:5.2f}x")
    cells = ("flagship-alt-k4-L16384", "flagship-alt-k16-L16384", "flagship-cohort-k16-L16384", "flagship-cohort-k4",
             "nl16k-alt-k4")
    for name in cells:
        read, write = lists(name)
        print(f"\n{name}: lists a digit a window, reads {read.sum() / read.shape[1] / (read.shape[0] / WINDOW):.1f}, "
              f"writes {write.sum() / write.shape[1] / (write.shape[0] / WINDOW):.1f}")
        for phase, live in ((readout, read), (fold_phase, write)):
            result = price(phase, live)
            base = result["best {1,4}" if phase is readout else "g4"][0]
            for strategy, (total, share) in result.items():
                picks = "  ".join(f"g{g} {s:4.0%}" for g, s in share.items()) if len(share) > 1 else ""
                print(f"  {phase.name:8s} {strategy:13s} {total / base:6.3f}  {picks}")


if __name__ == "__main__":
    main()

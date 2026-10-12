# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAB: small CUDA experiments whose measurements become kernel rules (docs/internals/tools/lab.md).

An experiment is one `measure/lab/experiments/<name>.cu` built on `measure/lab/lab.cuh`. This driver compiles it with
the pinned toolkit for the running card, reads its SASS and refuses to measure unless every expectation the experiment
declares holds (an instruction the experiment means to time is really there -- ptxas deletes work whose result is dead)
and no kernel touches local memory; then it checks the GPU is quiet, runs the repetitions, flags a result whose samples
spread, and writes a record keyed by the architecture.

    python tools/lab.py list
    python tools/lab.py sass tensor_tax [KERNEL_REGEX] [--full]
    python tools/lab.py run tensor_tax [--reps 7] [--record] [--force]
    python tools/lab.py ncu pool_layout --metric lts__t_sectors_srcunit_tex_op_read.sum [--record]

`--record` writes `measure/lab/results/<name>/<sm_XX>.json` (or `<sm_XX>.counters.json` for `ncu`), the file a
KERNEL_STANDARDS rule cites; without it the record goes to the scratch space. `ncu` reads hardware counters per kernel
(the median over its launches) where a question is about traffic rather than time. An experiment that reads data (a
cell's routing, say) has a `<name>.inputs.py` beside it whose `write(directory)` writes the files; the driver runs it
first, passes `--inputs <directory>`, and records each file's digest.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import dev_config
import sass
import sass_gate

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "measure" / "lab"
EXPERIMENTS = LAB / "experiments"
#: the kernels' include root: a part harness runs a component's own code (docs/internals/tools/lab.md#parts)
KERNELS = ROOT / "csrc" / "rola" / "src"
RESULTS = LAB / "results"

#: a result whose largest sample exceeds its smallest by more than this is flagged unstable
SPREAD = 1.05
#: GPU utilization above this percent before a run means another consumer shares the card
QUIET_PERCENT = 5


def experiments() -> dict[str, Path]:
    return {p.stem: p for p in sorted(EXPERIMENTS.glob("*.cu"))}


def question(path: Path) -> str:
    """The experiment's QUESTION, from its header comment."""
    text = " ".join(line.lstrip("/ ").strip() for line in path.read_text().splitlines()[:6])
    found = re.search(r"QUESTION:\s*(.*?)\s*METHOD:", text)
    return found.group(1) if found else "(no QUESTION in the header)"


def smi(*query: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [field.strip() for field in out.splitlines()[0].split(",")]


def device_arch() -> str:
    return "sm_" + smi("compute_cap")[0].replace(".", "")


def build(path: Path, arch: str) -> Path:
    """The experiment's binary for `arch`, compiled with the pinned toolkit."""
    out = dev_config.scratch("lab") / f"{path.stem}.{arch}"
    subprocess.run([dev_config.cuda_bin("nvcc"), "-O3", "-std=c++17", f"-arch={arch}", "-lineinfo", f"-I{LAB}",
                    f"-I{KERNELS}", "--expt-relaxed-constexpr", "-o", str(out), str(path)], check=True)
    return out


def demangle(names: list[str]) -> dict[str, str]:
    out = subprocess.run([dev_config.cuda_bin("cu++filt")], input="\n".join(names), capture_output=True, text=True,
                         check=True).stdout.splitlines()
    return dict(zip(names, out, strict=True))


def kernels(binary: Path) -> dict[str, dict]:
    """Per kernel (demangled): its opcode counts, registers and local bytes."""
    ops: dict[str, Counter] = {}
    current = None
    with sass.sass_lines(binary) as lines:
        for line in lines:
            if m := sass.FUNCTION.match(line):
                current = m.group(1)
                ops[current] = Counter()
            elif current and (m := sass.INSTRUCTION.match(line)):
                ops[current][m.group(3)] += 1
    usage = subprocess.run([dev_config.cuda_bin("cuobjdump"), "-res-usage", str(binary)], capture_output=True,
                           text=True, check=True).stdout
    regs: dict[str, dict] = {}
    name = None
    for line in usage.splitlines():
        if m := re.search(r"Function (\S+):", line):
            name = m.group(1)
        elif name and (m := re.search(r"REG:(\d+).*LOCAL:(\d+)", line)):
            regs[name] = {"regs": int(m.group(1)), "local": int(m.group(2))}
    names = demangle(list(ops))
    return {names[k]: {"ops": dict(v), **regs.get(k, {})} for k, v in ops.items()}


def count(ops: dict[str, int], prefixes: str) -> int:
    """The instructions whose opcode starts with one of `prefixes` (`|`-separated alternatives: ptxas may spell one
    operation two ways, an integer add as `IADD3` or as `IMAD.IADD` on the FMA pipe)."""
    return sum(n for op, n in ops.items() if any(op.startswith(p) for p in prefixes.split("|")))


def signatures(binary: Path) -> dict[str, dict]:
    """Per kernel (demangled): the SASS gate's signature (`tools/sass_gate.py`) -- local memory, and the calls into a
    convergence subroutine a converged warp executes (not ptxas's loop-exit `CALL.REL.NOINC`, nor a guarded fallback)."""
    found: dict[str, dict] = {}
    for cubin in sass.cubins(binary):
        sass_gate.gate(cubin, None, None)
        found |= sass_gate.STATS
    names = demangle(list(found))
    return {names[k]: v for k, v in found.items()}


#: KERNEL_STANDARDS section 29's debts: the experiments recorded before a lab had to budget its MMA kernels and check
#: its outputs; each is held to both when it is next changed, and the list only shrinks
_PRE_29 = {"collective_branch", "den_sweep", "facts_block", "facts_pipeline", "fold_mchan", "fold_part", "fold_step",
           "gang_ring", "list_fold", "mma_k8", "mma_overlap", "mma_pairing", "mma_queue", "pass_fixed", "pass_pipeline",
           "pool_layout", "publish_rounds", "read_step", "row_span", "smem_rates", "sparse_pass", "state_publish",
           "tensor_tax", "vote_stash"}


def check(binary: Path, table: dict[str, dict]) -> list[str]:
    """Every expectation the experiment declares, no local memory and no undeclared convergence call anywhere: the
    failures, empty when it may run."""
    lines = subprocess.run([str(binary), "--expect"], capture_output=True, text=True, check=True).stdout
    expected = [e for e in map(json.loads, lines.splitlines()) if e["lab"] == "expect"]
    failures = []
    for e in expected:
        hits = [k for k in table if re.search(e["kernel"], k)]
        if not hits:
            failures.append(f"no kernel matches {e['kernel']!r}")
        for k in hits:
            n = count(table[k]["ops"], e["op"])
            ok = {">=": n >= e["count"], "==": n == e["count"], "<=": n <= e["count"]}[e["cmp"]]
            if not ok:
                failures.append(f"{k}: {e['op']} {n}, expected {e['cmp']} {e['count']}")
    declared_calls = {k for e in expected for k in table if re.search(e["kernel"], k) and "CALL" in e["op"]}
    declared_branches = {k for e in expected for k in table if re.search(e["kernel"], k) and e["op"] == "BSSY"}

    if binary.name.split(".")[0] not in _PRE_29:
        stated = budgets(binary) + budgets(binary, "exempt")
        for k, info in table.items():
            if count(info["ops"], "HMMA|IMMA") and not any(re.search(b["kernel"], k) for b in stated):
                failures.append(f"{k}: an MMA kernel with no instruction budget (KERNEL_STANDARDS 29: `Bench::budget`, "
                                "or `Bench::exempt` with its reason)")
    declared = {op: {k for e in expected for k in table if re.search(e["kernel"], k) and e["op"] == op}
                for op in ("STEP_XU", "STEP_GLOBAL")}

    for k, sig in signatures(binary).items():
        if sig["ldl"] or sig["stack"]:
            failures.append(f"{k}: local memory ({sig['ldl']} LDL/STL, a {sig['stack']}-byte stack frame)")
        if sig["mma_loop_branches"] and k not in declared_branches:
            failures.append(f"{k}: {sig['mma_loop_branches']} branches inside an MMA step nobody declared "
                            "(KERNEL_STANDARDS 27: a step body is one basic block; a form that keeps them declares BSSY)")
        if sig["step_xu"] and k not in declared["STEP_XU"]:
            failures.append(f"{k}: {sig['step_xu']} conversion-unit instructions inside an MMA step nobody declared "
                            "(KERNEL_STANDARDS 29: convert on the FMA pipe; a form that keeps them declares STEP_XU)")
        if sig["step_global"] and k not in declared["STEP_GLOBAL"]:
            failures.append(f"{k}: {sig['step_global']} global accesses inside an MMA step nobody declared "
                            "(KERNEL_STANDARDS 29: stage by cp.async; a form that keeps them declares STEP_GLOBAL)")
        if sig["calls"] and k not in declared_calls:
            failures.append(f"{k}: {sig['calls']} convergence calls nobody declared (a collective under "
                            "role-dependent control flow becomes a call into a convergence subroutine)")
    return failures


def budgets(binary: Path, kind: str = "budget") -> list[dict]:
    """The instruction budgets the experiment states (`Bench::budget`: kernel regex, opcode class, max per MMA), or
    with `kind="mma"` the MMA each budgeted kernel issues (`Bench::mma`: kernel regex, hardware kind, K)."""
    lines = subprocess.run([str(binary), "--expect"], capture_output=True, text=True, check=True).stdout
    return [e for e in map(json.loads, lines.splitlines()) if e["lab"] == kind]


#: THE CARDS a lab's counts are priced on (`common/hardware.json`): the local card's product rates, sm_80's defaults
CARDS = (("sm_86 (RTX 3080 Ti)", "sm_86", "rtx3080ti"), ("sm_80 (A100)", "sm_80", None))


def predict(per: dict[str, float], kind: str, k: int) -> list[str]:
    """KERNEL_STANDARDS section 29's carry-over, per card: an MMA's tensor cycles (the kind's cycles a scheduler, K/16
    of them, over the schedulers), its shared-memory cycles (wavefronts at the wavefront rate, each shared atomic's cycles
    past its wavefront) and its issue cycles (instructions over the schedulers), all a tensor-pipe instruction an SM;
    the largest binds."""
    sys.path.insert(0, str(ROOT))
    from measure import hardware

    lines = []
    for label, arch, product in CARDS:
        rates = hardware.resolve(arch=arch, product=product)
        schedulers = rates["schedulers"].cycles
        pipes = {
            "tensor": rates[f"mma.{kind}"].cycles * k / 16 / schedulers,
            "shared": per.get("_wavefronts", 0.0) * rates["wavefront"].cycles
                      + per.get("shared atomic", 0.0) * (rates["atomic"].cycles - rates["atomic_wavefront"].cycles),
            "issue": sum(v for c, v in per.items() if not c.startswith("_")) * rates["issue"].cycles / schedulers,
        }
        bound = max(pipes, key=pipes.get)
        lines.append(f"    {label:<20} tensor {pipes['tensor']:5.2f}  shared {pipes['shared']:5.2f}  "
                     f"issue {pipes['issue']:5.2f}  -> {pipes[bound]:5.2f} cycles an MMA an SM ({bound} binds)")
    return lines


def _plain(name: str) -> str:
    """A kernel's name as both demanglers can agree on it: no return type, namespaces, casts or parameters."""
    name = re.sub(r"^void\s+", "", name)
    name = name[:name.index(">(") + 1] if ">(" in name else name.split("(")[0]
    return re.sub(r"\s+", "", re.sub(r"\((int|bool|unsigned int|long)\)", "", re.sub(r"\b\w+::", "", name)))


def executed_classes(binary: Path, extra: list[str]) -> dict[str, dict[str, float]]:
    """Per kernel (`_plain` name), each opcode class's executed instructions per tensor-pipe instruction (ncu's
    per-opcode counts over one repetition, the first launch of each kernel), and its shared-memory wavefronts and bank
    conflicts per tensor-pipe instruction (`_wavefronts`, `_conflicts`)."""
    out = subprocess.run([dev_config.get("toolchain.ncu"), "--metrics",
                          "sass__inst_executed_per_opcode,sm__inst_executed_pipe_tensor_op_hmma.sum,"
                          "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,"
                          "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
                          "--print-metric-instances", "details", "--csv", str(binary), "--reps", "1", *extra],
                         capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    seen: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(lines[start:]):
        seen.setdefault(_plain(row["Kernel Name"]), {}).setdefault(row["Metric Name"], row["Metric Value"])
    found = {}
    for k, m in seen.items():
        mma = float(m["sm__inst_executed_pipe_tensor_op_hmma.sum"].replace(",", ""))
        per: dict[str, float] = {}
        for op, n in re.findall(r"([A-Z][A-Z0-9_]*)\s*[:=]\s*([0-9.]+)", m["sass__inst_executed_per_opcode"]):
            per[sass.opcode_class(op)] = per.get(sass.opcode_class(op), 0.0) + float(n) / max(mma, 1.0)
        per["_wavefronts"] = float(m["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"].replace(",", "")) / max(mma, 1.0)
        per["_conflicts"] = float(m["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"].replace(",", "")) / max(mma, 1.0)
        found[k] = per
    return found


def over_budget(binary: Path, table: dict[str, dict], extra: list[str]) -> list[str]:
    """KERNEL_STANDARDS sections 29-30: each budgeted kernel's executed instructions per MMA by class, and its
    shared-memory wavefronts per MMA (the class `wavefronts`), against its stated budget, printed; those over it."""
    stated = budgets(binary)
    if not stated:
        return []
    mmas = budgets(binary, "mma")
    measured = executed_classes(binary, extra)
    over = []
    for k in table:
        mine = [b for b in stated if re.search(b["kernel"], k)]
        if not mine:
            continue
        per = measured.get(_plain(k), {})
        print(f"  budget, instructions an MMA: {k[:110]}")
        for b in mine:
            got = per.get("_wavefronts" if b["class"] == "wavefronts" else b["class"], 0.0)
            #: a slack for once-a-kernel setup and rounding: 3% and 0.02 an MMA
            flag = "OVER" if got > b["max"] * 1.03 + 0.02 else "ok"
            print(f"    {b['class']:<16} {got:6.2f}  budget {b['max']:6.2f}  {flag}")
            if flag == "OVER":
                over.append(f"{k[:80]}: {b['class']} {got:.2f} > {b['max']:.2f}")
        print(f"    {'total':<16} {sum(v for c, v in per.items() if not c.startswith('_')):6.2f}"
              f"   shared-memory wavefronts {per.get('_wavefronts', 0.0):.2f}"
              f" ({per.get('_conflicts', 0.0):.2f} of them bank conflicts)")
        for m in (m for m in mmas if re.search(m["kernel"], k)):
            print(*predict(per, m["kind"], m["k"]), sep="\n")
    return over


def quiet() -> int:
    """The lowest GPU utilization over three looks: another consumer time-slices every kernel and inflates cycles."""
    looks = []
    for _ in range(3):
        looks.append(int(smi("utilization.gpu")[0]))
        time.sleep(0.2)
    return min(looks)


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    #: the sources a record depends on; the records themselves are outputs
    sources = ["measure/lab/lab.cuh", "measure/lab/experiments", "tools/lab.py"]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *sources], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return {"sha": sha, "dirty": bool(dirty)}


def summarize(samples: list[float]) -> dict:
    lo, hi = min(samples), max(samples)
    return {"median": statistics.median(samples), "min": lo, "max": hi,
            "unstable": len(samples) > 1 and lo > 0 and hi / lo > SPREAD}


def inputs(name: str) -> tuple[list[str], dict[str, str]]:
    """The experiment's inputs: `<name>.inputs.py`, if it has one, writes them into the scratch space; the binary's
    extra arguments and each file's digest."""
    script = EXPERIMENTS / f"{name}.inputs.py"
    if not script.exists():
        return [], {}
    directory = dev_config.scratch("lab") / "inputs" / name
    directory.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(f"{name}_inputs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.write(directory)
    digests = {f: hashlib.sha256((directory / f).read_bytes()).hexdigest()[:16] for f in files}
    return ["--inputs", str(directory)], digests


def execute(binary: Path, reps: int, extra: list[str]) -> tuple[dict, list[dict]]:
    """The experiment's device line and its results, each with its samples summarized."""
    out = subprocess.run([str(binary), "--reps", str(reps), *extra], capture_output=True, text=True,
                         check=True).stdout
    device, results = {}, []
    for line in out.splitlines():
        if not line.startswith('{"lab"'):
            continue
        item = json.loads(line)
        if item["lab"] == "device":
            device = item
        elif item["lab"] == "result":
            results.append({**item, **summarize(item["samples"])})
        elif item["lab"] in ("check", "unchecked"):
            device.setdefault("_checks", []).append(item)
    return device, results


def unverified(name: str, device: dict, results: list[dict]) -> list[str]:
    """KERNEL_STANDARDS section 29: the variants whose outputs failed their check, and (past the debts) the recorded
    variants with neither a check nor a stated reason for none -- a result belongs to the variant whose label it starts
    with."""
    checks = device.get("_checks", [])
    bad = [f"{c['variant']}: {c['detail']}" for c in checks if c["lab"] == "check" and not c["ok"]]
    if name in _PRE_29:
        return bad
    covered = [c["variant"] for c in checks]
    bad += [f"{r['name']}: no output check" for r in results if not any(r["name"].startswith(v) for v in covered)]
    return bad


def report(name: str, arch: str, device: dict, results: list[dict], reps: int) -> None:
    width = max(len(r["name"]) for r in results)
    print(f"{name} on {device.get('name')} ({arch}), {reps} repetitions -- median [min, max]")
    for r in results:
        flag = "  UNSTABLE" if r["unstable"] else ""
        print(f"  {r['name']:<{width}}  {r['median']:10.3f} [{r['min']:.3f}, {r['max']:.3f}] {r['unit']}{flag}")


def keep(name: str, arch: str, device: dict, results: list[dict], table: dict, reps: int, utilization: int,
         record: bool, digests: dict[str, str]) -> Path:
    """The run's record: into `measure/lab/results` under `--record`, else the scratch space."""
    driver, = smi("driver_version")
    toolkit = subprocess.run([dev_config.cuda_bin("nvcc"), "--version"], capture_output=True,
                             text=True).stdout.strip().splitlines()[-1]
    body = {"experiment": name, "question": question(experiments()[name]), "arch": arch, "device": device.get("name"),
            "checks": device.get("_checks", []),
            "driver": driver, "toolkit": toolkit, "git": git_state(), "reps": reps,
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "utilization_before": utilization, "inputs": digests, "results": results,
            "kernels": {k: {"regs": v.get("regs"), "ops": dict(Counter(v["ops"]).most_common(12))}
                        for k, v in table.items()}}
    target = (RESULTS / name if record else dev_config.scratch("lab") / "results" / name) / f"{arch}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=1) + "\n")
    return target


def run(name: str, reps: int, record: bool, force: bool) -> int:
    arch = device_arch()
    binary = build(experiments()[name], arch)
    table = kernels(binary)
    failures = check(binary, table)
    if failures:
        print(f"lab: {name} REFUSED -- its SASS is not what it measures:", *failures, sep="\n  ")
        return 1

    extra, digests = inputs(name)
    utilization = quiet()
    if utilization > QUIET_PERCENT and not force:
        print(f"lab: the GPU is {utilization}% busy before the run; another consumer would inflate every cycle "
              "(--force to measure anyway)")
        return 1

    device, results = execute(binary, reps, extra)
    report(name, arch, device, results, reps)
    for c in device.get("_checks", []):
        print(f"  {'checked' if c['lab'] == 'check' else 'unchecked'} {c['variant']}: "
              f"{'ok' if c.get('ok') else 'FAILED' if c['lab'] == 'check' else c['reason']}"
              + (f" ({c['detail']})" if c["lab"] == "check" else ""))
    wrong = unverified(name, device, results)
    if wrong:
        print(f"lab: {name}'s outputs are not verified, not recorded:", *wrong, sep="\n  ")
        return 1
    over = over_budget(binary, table, extra)
    if over and not force:
        print(f"lab: {name} OVER ITS INSTRUCTION BUDGET, not recorded (--force to record anyway):", *over,
              sep="\n  ")
        return 1
    target = keep(name, arch, device, results, table, reps, utilization, record, digests)
    print(f"lab: recorded {target.relative_to(ROOT) if record else target}")
    return 0


def ncu_counters(binary: Path, metrics: list[str], extra: list[str]) -> dict[str, dict[str, float]]:
    """Per kernel (ncu's name), each metric's median over the kernel's launches in one repetition of the experiment."""
    out = subprocess.run([dev_config.get("toolchain.ncu"), "--metrics", ",".join(metrics), "--csv", str(binary),
                          "--reps", "1", *extra], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    seen: dict[str, dict[str, list[float]]] = {}
    for row in csv.DictReader(lines[start:]):
        value = float(row["Metric Value"].replace(",", ""))
        seen.setdefault(row["Kernel Name"], {}).setdefault(row["Metric Name"], []).append(value)
    return {k: {m: statistics.median(v) for m, v in ms.items()} for k, ms in seen.items()}


def counters(name: str, metrics: list[str], record: bool) -> int:
    arch = device_arch()
    binary = build(experiments()[name], arch)
    failures = check(binary, kernels(binary))
    if failures:
        print(f"lab: {name} REFUSED -- its SASS is not what it measures:", *failures, sep="\n  ")
        return 1
    extra, digests = inputs(name)
    found = ncu_counters(binary, metrics, extra)
    width = max(len(k) for k in found)
    for k, ms in found.items():
        print(f"  {k:<{width}}  " + "  ".join(f"{m} {v:,.0f}" for m, v in ms.items()))
    body = {"experiment": name, "question": question(experiments()[name]), "arch": arch, "git": git_state(),
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "metrics": metrics,
            "inputs": digests, "kernels": found}
    target = (RESULTS / name if record else dev_config.scratch("lab") / "results" / name) / f"{arch}.counters.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=1) + "\n")
    print(f"lab: recorded {target.relative_to(ROOT) if record else target}")
    return 0


def show_sass(name: str, pattern: str | None, full: bool, arch: str | None) -> int:
    path = experiments()[name]
    binary = build(path, arch or device_arch())
    table = kernels(binary)
    sigs = signatures(binary)
    for k, info in table.items():
        if pattern and not re.search(pattern, k):
            continue
        top = ", ".join(f"{op} {n}" for op, n in Counter(info["ops"]).most_common(10))
        sig = sigs.get(k, {})
        gap = f", MMA gap {sig['mma_gap']} (around the loop {sig['mma_gap_around']})" if sig.get("mma_gap_around") else ""
        print(f"{k}\n  {info.get('regs')} registers{gap}; {top}")
    if full:
        print(subprocess.run([dev_config.cuda_bin("cuobjdump"), "-sass", str(binary)], capture_output=True,
                             text=True, check=True).stdout)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("sass")
    s.add_argument("experiment")
    s.add_argument("kernel", nargs="?")
    s.add_argument("--full", action="store_true")
    s.add_argument("--arch", help="compile for this architecture (sm_80, ...) in place of the device's: SASS only")
    r = sub.add_parser("run")
    r.add_argument("experiment")
    r.add_argument("--reps", type=int, default=7)
    r.add_argument("--record", action="store_true", help="write the record into measure/lab/results")
    r.add_argument("--force", action="store_true", help="measure on a busy GPU")
    n = sub.add_parser("ncu")
    n.add_argument("experiment")
    n.add_argument("--metric", action="append", required=True, help="an ncu metric; repeat for several")
    n.add_argument("--record", action="store_true", help="write the counters into measure/lab/results")
    args = ap.parse_args()
    if args.cmd == "list":
        for name, path in experiments().items():
            print(f"{name}: {question(path)}")
        return 0
    if args.experiment not in experiments():
        print(f"lab: no experiment {args.experiment!r}; `list` names them")
        return 1
    if args.cmd == "sass":
        return show_sass(args.experiment, args.kernel, args.full, args.arch)
    if args.cmd == "ncu":
        return counters(args.experiment, args.metric, args.record)
    return run(args.experiment, args.reps, args.record, args.force)


if __name__ == "__main__":
    sys.exit(main())

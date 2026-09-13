#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEV ENVIRONMENT (KERNEL_STANDARDS §23): the one command that sets up, checks and describes a host or container.

    python tools/dev.py init [--force]          # write this machine's dev config, install the pinned tools, wire hooks
    python tools/dev.py check                   # every value, tool, pin and lock this tree depends on; exit 1 on a red
    python tools/dev.py show                    # every resolved setting and the file it came from
    python tools/dev.py get <section.key>       # one value, for a shell caller (cuda-home resolves the toolkit)
    python tools/dev.py clock --mhz 1665        # lock this host's SM clock once, prove the round trip, record it
    python tools/dev.py worktree <name> <sha> [--build]   # a worktree, its pointer venv, its gate, its extension

The settings live in the dev config directory (`tools/dev_config.py`). Docs: docs/internals/tools/dev.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede this import
import mold_toolchain  # noqa: E402
import sccache_toolchain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PINS = {"sccache": ROOT / "tools" / "sccache_pin.json", "mold": ROOT / "tools" / "mold_pin.json"}
#: each pinned tool's own version reader, the one its build gate compares against the pin
VERSION_OF = {"sccache": sccache_toolchain.sccache_version, "mold": mold_toolchain.mold_version}


def _write_section(section: str, values: dict, force: bool) -> None:
    """Merge `values` into the section's file: keys already there are kept unless `force`."""
    path = dev_config.directory() / f"{section}.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update({k: v for k, v in values.items() if force or k not in merged})
    if not merged:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    dev_config.load.cache_clear()
    print(f"wrote {path}")


def install_pinned(name: str, dest: Path) -> str:
    """Download the pinned release, verify its tarball sha256 (and the binary's, where the pin has one), and unpack the
    whole release under `dest` -- its layout is part of the tool (mold selects `ld.mold` beside itself)."""
    pin = json.loads(PINS[name].read_text())
    with tempfile.TemporaryDirectory(prefix=f"rola_{name}_") as tmp:
        tar_path = Path(tmp) / pin["release_asset"]
        urllib.request.urlretrieve(pin["release_url"], tar_path)
        got = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if got != pin["tarball_sha256"]:
            raise SystemExit(f"{name}: tarball sha256 {got} is not the pin's {pin['tarball_sha256']}")
        with tarfile.open(tar_path) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith(f"/{name}") and m.isfile())
            dest.mkdir(parents=True, exist_ok=True)
            tf.extractall(dest, filter="tar")
    binary = dest / member.name
    if "binary_sha256" in pin and hashlib.sha256(binary.read_bytes()).hexdigest() != pin["binary_sha256"]:
        raise SystemExit(f"{name}: {binary} sha256 is not the pin's {pin['binary_sha256']}")
    print(f"installed {pin['version_string']} -> {binary}")
    return str(binary)


def base_venv_of(prefix: Path) -> str | None:
    """The shared venv a pointer venv borrows from, or the venv itself when it carries torch directly."""
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    pointer = site / "zz_rola_base.pth"
    if pointer.exists():
        return str(Path(pointer.read_text().strip()).parents[2])
    return str(prefix) if (site / "torch").is_dir() else None


def wire_hooks(checkout: Path) -> None:
    """The commit gate as a property of the checkout: `core.hooksPath` set per worktree to the tracked hooks."""
    subprocess.run(["git", "-C", str(checkout), "config", "extensions.worktreeConfig", "true"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "--worktree", "core.hooksPath", "tools/git-hooks"], check=True)


def cmd_init(a) -> int:
    for section, values in dev_config.detect().items():
        _write_section(section, values, a.force)
    if dev_config.get("toolchain.cuda_home") is None:
        _write_section("toolchain", {"cuda_home": dev_config.cuda_home()}, a.force)
    base = base_venv_of(Path(sys.prefix))
    if base:
        _write_section("workspace", {"base_venv": base}, a.force)
    tools_dir = Path(dev_config.get("host.tools_dir"))
    for name in PINS:
        current = dev_config.get(f"toolchain.{name}")
        pin = json.loads(PINS[name].read_text())
        if current is None or _pinned_version(name, current) != pin["version_string"]:
            _write_section("toolchain", {name: install_pinned(name, tools_dir)}, force=True)
    wire_hooks(ROOT)
    if dev_config.get("clock.ghz") is None:
        print("clock: unset -- this host measures UNLOCKED until `python tools/dev.py clock --mhz <MHz>` runs")
    return cmd_check(a)


def _pinned_version(name: str, path: str) -> str:
    try:
        return VERSION_OF[name](path)
    except (OSError, RuntimeError):
        return ""


def _version(tool: str) -> str:
    try:
        return subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except OSError:
        return ""


def _ncu_version(ncu: str) -> tuple[int, ...]:
    m = re.search(r"Version (\d+)\.(\d+)", _version(ncu))
    return tuple(int(x) for x in m.groups()) if m else ()


def cmd_check(_a) -> int:
    rows: list[tuple[bool, str, str]] = []

    def row(ok: bool, what: str, fix: str = "") -> None:
        rows.append((ok, what, fix))

    try:
        dev_config.load()
        row(True, f"config {dev_config.directory()}")
    except SystemExit as ex:
        row(False, f"config: {ex}", "fix the named file")
        return _report(rows)
    try:
        home = dev_config.cuda_home()
        for tool in ("nvcc", "ptxas", "nvdisasm", "cuobjdump"):
            path = Path(home) / "bin" / tool
            row(path.exists(), f"toolchain {tool}: {path}", "set toolchain.cuda_home to a CUDA toolkit with it")
    except SystemExit as ex:
        row(False, f"toolchain: {ex}", "python tools/dev.py init")
    ncu = dev_config.get("toolchain.ncu")
    row(_ncu_version(ncu) >= (2025, 3), f"toolchain ncu {'.'.join(map(str, _ncu_version(ncu))) or 'missing'}: {ncu}",
        "install Nsight Compute 2025.3+ and set toolchain.ncu")
    for name, gate in (("sccache", sccache_toolchain.gate), ("mold", mold_toolchain.link_flags)):
        try:
            gate()
            row(True, f"toolchain {name}: {dev_config.get(f'toolchain.{name}')} matches its pin")
        except RuntimeError as ex:
            row(False, f"toolchain {name}: {str(ex).splitlines()[0][:140]}", "python tools/dev.py init")
    hooks = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "core.hooksPath"], capture_output=True, text=True)
    row(hooks.stdout.strip() == "tools/git-hooks", f"commit gate wired in {ROOT.name}", "python tools/dev.py init")
    for key in ("host.lock_dir", "host.scratch"):
        path = Path(dev_config.get(key))
        path.mkdir(parents=True, exist_ok=True)
        row(os.access(path, os.W_OK), f"{key} writable: {path}")
    row(os.access(Path(dev_config.get("host.gpu_lock")).parent, os.W_OK), f"host.gpu_lock: {dev_config.get('host.gpu_lock')}")
    ghz = dev_config.get("clock.ghz")
    row(True, f"clock: {'locks at ' + str(ghz) + ' GHz' if ghz else 'unlocked (rows carry their measured clock)'}")
    base = dev_config.get("workspace.base_venv")
    row(bool(base) and (Path(base) / "bin" / "python").exists(), f"workspace.base_venv: {base}", "python tools/dev.py init")
    row(Path(dev_config.get("workspace.worktrees")).is_dir(), f"workspace.worktrees: {dev_config.get('workspace.worktrees')}")
    row(Path(dev_config.get("store.results")).is_dir(), f"store.results: {dev_config.get('store.results')}")
    return _report(rows)


def _report(rows: list[tuple[bool, str, str]]) -> int:
    for ok, what, fix in rows:
        print(f"{'ok ' if ok else 'RED'}  {what}" + ("" if ok or not fix else f"  -- {fix}"))
    red = sum(1 for ok, _, _ in rows if not ok)
    print(f"dev check: {'green' if not red else f'{red} red'}")
    return 1 if red else 0


def cmd_show(_a) -> int:
    for section, keys in dev_config.load().items():
        for name, (value, source) in keys.items():
            print(f"{section}.{name} = {json.dumps(value)}  [{source}]")
    return 0


def cmd_get(a) -> int:
    print(dev_config.cuda_home() if a.key == "cuda-home" else json.dumps(dev_config.get(a.key)).strip('"'))
    return 0


def _windows(rel: str) -> Path:
    tool = dev_config.windows_tool(rel)
    if tool is None or not tool.exists():
        raise SystemExit(f"host.windows_system32 does not hold {rel}; python tools/dev.py init detects it under WSL")
    return tool


def _clock_wsl(mhz: int) -> dict:
    """An elevated scheduled task a non-administrator may start: registered once, one administrator prompt."""
    smi, tasks, ps = _windows("nvidia-smi.exe"), _windows("schtasks.exe"), _windows("WindowsPowerShell/v1.0/powershell.exe")
    appdata = subprocess.run([str(ps), "-NoProfile", "-Command", "$env:LOCALAPPDATA"],
                             capture_output=True, text=True, check=True).stdout.strip()
    home = Path(f"/mnt/{appdata[0].lower()}{appdata[2:].replace(chr(92), '/')}") / "rola"
    home.mkdir(parents=True, exist_ok=True)

    def to_windows(path: Path) -> str:
        s = str(path)
        return f"{s[5].upper()}:{s[6:]}".replace("/", "\\")

    lines = ["$ErrorActionPreference = 'Stop'"]
    for name, arg in (("gpu-lock", f"-lgc {mhz},{mhz}"), ("gpu-unlock", "-rgc")):
        vbs = home / f"{name}.vbs"
        vbs.write_text(f'CreateObject("WScript.Shell").Run "C:\\Windows\\System32\\nvidia-smi.exe {arg}", 0, True\r\n')
        lines.append(f"schtasks /create /tn '{name}' /sc once /st 00:00 /rl highest /f /tr 'wscript.exe \"{to_windows(vbs)}\"'")
    script = home / "register-gpu-clock-tasks.ps1"
    script.write_text("\r\n".join(lines) + "\r\n")
    print(f"registering the tasks (one administrator prompt): {to_windows(script)}")
    subprocess.run([str(ps), "-NoProfile", "-Command", f"Start-Process powershell -Verb RunAs -Wait -ArgumentList "
                    f"'-NoProfile -ExecutionPolicy Bypass -File \"{to_windows(script)}\"'"], check=True)
    run = [str(tasks), "/run", "/tn"]
    return {"ghz": mhz / 1000.0, "lock": [*run, "gpu-lock"], "unlock": [*run, "gpu-unlock"],
            "read": [str(smi), "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


def _clock_linux(mhz: int) -> dict:
    """`nvidia-smi -lgc` needs root: the commands run through `sudo -n`, and the sudoers rule is printed."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise SystemExit("no nvidia-smi on PATH")
    print(textwrap.dedent(f"""
        the lock runs through sudo without a password; give it the rule (visudo):
            {os.environ.get("USER", "USER")} ALL=(root) NOPASSWD: {smi} -lgc *, {smi} -rgc
        """))
    return {"ghz": mhz / 1000.0, "lock": ["sudo", "-n", smi, "-lgc", f"{mhz},{mhz}"],
            "unlock": ["sudo", "-n", smi, "-rgc"], "read": [smi, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


def cmd_clock(a) -> int:
    """Lock the SM clock, read the driver's clock under the lock, unlock: the round trip the harness makes, proven."""
    cfg = _clock_wsl(a.mhz) if dev_config.get("host.windows_system32") else _clock_linux(a.mhz)
    subprocess.run(cfg["lock"], check=True, capture_output=True, timeout=60)
    time.sleep(2)
    read = subprocess.run(cfg["read"], capture_output=True, text=True, timeout=60).stdout.strip()
    subprocess.run(cfg["unlock"], check=True, capture_output=True, timeout=60)
    if not read or abs(int(read.split()[0]) - a.mhz) > 0.01 * a.mhz:
        raise SystemExit(f"the lock did not hold: the driver reports {read!r} MHz, wanted {a.mhz}")
    print(f"lock proven: {read} MHz under lock, released")
    _write_section("clock", {k: cfg[k] for k in ("ghz", "lock", "unlock")}, force=True)
    return 0


def cmd_worktree(a) -> int:
    """A worktree in `workspace.worktrees`, its commit gate wired before anything builds, a pointer venv over
    `workspace.base_venv`, and its extension: copied from the main checkout when that checkout is the same commit and
    built, else built in place under the host budget (`--build` always builds)."""
    base = dev_config.get("workspace.base_venv")
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if not base or not (Path(base) / "lib" / py / "site-packages").is_dir():
        raise SystemExit(f"workspace.base_venv {base!r} has no lib/{py}/site-packages; python tools/dev.py init")
    main = Path(subprocess.run(["git", "-C", str(ROOT), "worktree", "list", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.split("\n", 1)[0].split(" ", 1)[1])
    worktree = Path(dev_config.get("workspace.worktrees")) / a.name
    venv = worktree.parent / f"venv-{a.name}"
    for p in (worktree, venv):
        if p.exists():
            raise SystemExit(f"refusing: {p} already exists")
    subprocess.run(["git", "-C", str(main), "worktree", "add", str(worktree), a.sha], check=True)
    if not os.access(worktree / "tools" / "git-hooks" / "pre-commit", os.X_OK):
        raise SystemExit(f"refusing: {worktree}/tools/git-hooks/pre-commit is missing or not executable")
    wire_hooks(worktree)
    subprocess.run(["uv", "venv", "--python", py.removeprefix("python"), "--seed", str(venv)], check=True)
    site = venv / "lib" / py / "site-packages"
    (site / f"aa_rola_{a.name}.pth").write_text(f"{worktree}\n")
    (site / "zz_rola_base.pth").write_text(f"{Path(base) / 'lib' / py / 'site-packages'}\n")
    main_so = sorted((main / "rola").glob("_C.cpython-*.so"))
    same = subprocess.run(["git", "-C", str(main), "rev-parse", "HEAD"], capture_output=True, text=True).stdout == \
        subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    install = [str(venv / "bin" / "python"), "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"]
    if main_so and same and not a.build:
        subprocess.run(install, cwd=worktree, env={**os.environ, "ROLA_NO_EXTENSION": "1"}, check=True)
        shutil.copy(main_so[0], worktree / "rola")
    else:
        subprocess.run(install, cwd=worktree, check=True)
    finder = site / "__editable___rola_0_1_0_dev0_finder.py"
    if not finder.exists() or f"'rola': '{worktree}/rola'" not in finder.read_text():
        raise SystemExit(f"refusing: the editable finder {finder} does not map rola to {worktree}/rola")
    subprocess.run([str(venv / "bin" / "python"), "-c", "import rola; from rola._build_config import BUILD_CONFIG; "
                    "print('rola:', rola.__file__, 'archs:', BUILD_CONFIG['archs'])"], cwd=worktree, check=True)
    print(f"worktree {worktree}  venv {venv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write this machine's dev config, install pinned tools, wire hooks, then check")
    p.add_argument("--force", action="store_true", help="overwrite detected values already in the config")
    p.set_defaults(fn=cmd_init)
    sub.add_parser("check", help="verify every setting, tool, pin and lock").set_defaults(fn=cmd_check)
    sub.add_parser("show", help="every resolved setting and its source").set_defaults(fn=cmd_show)
    p = sub.add_parser("get", help="print one value: <section.key>, or cuda-home")
    p.add_argument("key")
    p.set_defaults(fn=cmd_get)
    p = sub.add_parser("clock", help="lock this host's SM clock once and record it")
    p.add_argument("--mhz", type=int, required=True, help="a clock the card holds under the kernel's power")
    p.set_defaults(fn=cmd_clock)
    p = sub.add_parser("worktree", help="a worktree with its pointer venv, commit gate and extension")
    p.add_argument("name")
    p.add_argument("sha")
    p.add_argument("--build", action="store_true", help="build the extension even when the main checkout's can be copied")
    p.set_defaults(fn=cmd_worktree)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

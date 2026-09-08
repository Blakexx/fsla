#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ONE-TIME HOST SETUP for the clock lock the harness relies on (`tools/clock_lock.py`).

    python tools/setup_clock_lock.py --mhz 1665

A kernel number compares across runs only at one SM clock (`docs/internals/common/sm_clock.md`),
and locking a clock is a privileged, host-specific act. This tool knows the hosts; the
harness does not. It writes `~/.config/rola/clock.json` with the lock and unlock commands the
host needs, then proves the round trip: lock, read the clock, unlock.

WSL under Windows: clocks belong to the Windows driver, and a non-administrator process may
not set them but may START an elevated scheduled task it owns. The tool registers two
tasks, `gpu-lock` and `gpu-unlock`, each running a hidden launcher (`wscript`, no console
window) that calls `nvidia-smi.exe`; registering them asks for administrator consent ONCE.

Linux with the NVIDIA driver: `nvidia-smi -lgc` needs root. The tool writes the commands
through `sudo -n` and prints the sudoers line that lets them run without a password.

Anything else: the host cannot lock, and the harness records its runs as UNLOCKED.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

CONFIG = Path(os.environ.get("ROLA_CLOCK_CONFIG", Path.home() / ".config" / "rola" / "clock.json"))
WIN_SMI = Path("/mnt/c/Windows/System32/nvidia-smi.exe")
WIN_TASKS = Path("/mnt/c/Windows/System32/schtasks.exe")
WIN_PS = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def windows_profile_dir() -> Path:
    out = subprocess.run([str(WIN_PS), "-NoProfile", "-Command", "$env:LOCALAPPDATA"],
                         capture_output=True, text=True, check=True).stdout.strip()
    drive, rest = out[0].lower(), out[2:].replace("\\", "/")
    return Path(f"/mnt/{drive}{rest}") / "rola"


def to_windows(path: Path) -> str:
    s = str(path)
    assert s.startswith("/mnt/") and len(s) > 6, s
    return f"{s[5].upper()}:{s[6:]}".replace("/", "\\")


def setup_wsl(mhz: int) -> dict:
    if not (WIN_SMI.exists() and WIN_TASKS.exists() and WIN_PS.exists()):
        raise SystemExit("WSL, but nvidia-smi.exe, schtasks.exe or powershell.exe is missing "
                         "on the Windows side")
    home = windows_profile_dir()
    home.mkdir(parents=True, exist_ok=True)
    launchers = {}
    for name, arg in (("gpu-lock", f"-lgc {mhz},{mhz}"), ("gpu-unlock", "-rgc")):
        vbs = home / f"{name}.vbs"
        vbs.write_text(f'CreateObject("WScript.Shell").Run '
                       f'"C:\\Windows\\System32\\nvidia-smi.exe {arg}", 0, True\r\n')
        launchers[name] = vbs
    script = home / "register-gpu-clock-tasks.ps1"
    lines = ["$ErrorActionPreference = 'Stop'"]
    for name, vbs in launchers.items():
        #: PowerShell passes a single-quoted argument verbatim, inner double quotes included.
        lines.append(f"schtasks /create /tn '{name}' /sc once /st 00:00 /rl highest /f "
                     f"/tr 'wscript.exe \"{to_windows(vbs)}\"'")
    script.write_text("\r\n".join(lines) + "\r\n")
    print(f"registering the tasks (one administrator prompt): {to_windows(script)}")
    subprocess.run([str(WIN_PS), "-NoProfile", "-Command",
                    f"Start-Process powershell -Verb RunAs -Wait -ArgumentList "
                    f"'-NoProfile -ExecutionPolicy Bypass -File \"{to_windows(script)}\"'"],
                   check=True)
    run = [str(WIN_TASKS), "/run", "/tn"]
    return {"ghz": mhz / 1000.0, "lock": run + ["gpu-lock"], "unlock": run + ["gpu-unlock"],
            "read": [str(WIN_SMI), "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


def setup_linux(mhz: int) -> dict:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise SystemExit("no nvidia-smi on PATH")
    me = os.environ.get("USER", "USER")
    print(textwrap.dedent(f"""
        the lock runs through sudo without a password; give it the rule (visudo):
            {me} ALL=(root) NOPASSWD: {smi} -lgc *, {smi} -rgc
        """))
    return {"ghz": mhz / 1000.0, "lock": ["sudo", "-n", smi, "-lgc", f"{mhz},{mhz}"],
            "unlock": ["sudo", "-n", smi, "-rgc"],
            "read": [smi, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]}


def prove(cfg: dict) -> None:
    """lock, read the driver's clock, unlock: the round trip the harness will make."""
    subprocess.run(cfg["lock"], check=True, capture_output=True, timeout=60)
    time.sleep(2)
    read = subprocess.run(cfg["read"], capture_output=True, text=True, timeout=60).stdout.strip()
    subprocess.run(cfg["unlock"], check=True, capture_output=True, timeout=60)
    want = int(round(cfg["ghz"] * 1000))
    if not read or abs(int(read.split()[0]) - want) > 0.01 * want:
        raise SystemExit(f"the lock did not hold: the driver reports {read!r} MHz, wanted {want}")
    print(f"lock proven: {read} MHz under lock, released")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mhz", type=int, required=True,
                    help="the SM clock to lock, in MHz: one the card holds under this kernel's "
                         "power (nvidia-smi -q -d SUPPORTED_CLOCKS lists them)")
    args = ap.parse_args()
    if is_wsl():
        cfg = setup_wsl(args.mhz)
    elif sys.platform.startswith("linux"):
        cfg = setup_linux(args.mhz)
    else:
        raise SystemExit("this host cannot lock its clock from here; the harness will record "
                         "its runs as UNLOCKED")
    prove(cfg)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({k: cfg[k] for k in ("ghz", "lock", "unlock")}, indent=1) + "\n")
    print(f"wrote {CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

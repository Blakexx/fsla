# `tools/dev.py` — the dev environment, set up and checked by one command

KERNEL_STANDARDS §23. One command sets up a host or a container for this tree, checks it and describes it. It
replaced `tools/build/new_worktree.sh`, `tools/setup_clock_lock.py` and `tools/build/build_lock.sh`. The
settings it writes and reads are the dev config ([dev_config.md](dev_config.md)).

    python tools/dev.py init [--force]
    python tools/dev.py check
    python tools/dev.py show
    python tools/dev.py get <section.key> | cuda-home
    python tools/dev.py clock --mhz 1665
    python tools/dev.py worktree <name> <sha> [--build]

**`init`** writes this machine's config:
- The CUDA toolkit on `PATH`, or torch's.
- The newest Nsight Compute CLI under `/usr/local/cuda-*`.
- `sccache` and `mold` where they are installed.
- The Windows System32 mount under WSL.
- The base venv of the running interpreter: a pointer venv's `zz_rola_base.pth` target, or the venv itself.

A key already in a file is kept unless `--force`. A pinned tool that is missing, or does not report its pin's
version, is downloaded from the pin's release URL (`tools/sccache_pin.json`, `tools/mold_pin.json`). Its
tarball sha256 is verified, and so is its binary sha256 where the pin has one. The whole release is unpacked
under `host.tools_dir`, because the layout is part of the tool: `-fuse-ld=mold` selects the `ld.mold` next to
`mold`. `init` then wires this checkout's commit gate (`core.hooksPath` per worktree) and runs `check`. The
clock is not set by `init`, because locking it asks for administrator consent once.

**`check`** prints one row per dependency and exits 1 on any red:
- The config parses.
- nvcc, ptxas, nvdisasm and cuobjdump exist in the toolkit.
- `ncu` is 2025.3 or later.
- sccache and mold pass the build's own gates (`sccache_toolchain.gate`, `mold_toolchain.link_flags`), so
  `check` refuses exactly what a build would.
- The commit gate is wired.
- The lock directory, scratch directory and GPU lock directory are writable.
- The clock is set, or the host is marked unlocked.
- The base venv, the worktrees folder and the store exist.

**`clock --mhz N`** locks the SM clock, reads the driver's clock under the lock, and releases it: the round trip
the harness makes (`tools/clock_lock.py`, [sm_clock.md](../common/sm_clock.md)). Only after the lock is proven
does it write `clock.json`.
- **Under WSL:** it registers two elevated scheduled tasks, `gpu-lock` and `gpu-unlock`. Each runs a hidden
  `wscript` launcher that calls `nvidia-smi.exe`, and registering them asks for administrator consent once.
- **On Linux:** it writes `sudo -n nvidia-smi -lgc` commands and prints the sudoers rule they need.

**`worktree <name> <sha>`** creates the worktree in `workspace.worktrees` and wires its commit gate before
anything is built. It makes a seeded pointer venv beside it with two `.pth` files: this worktree's root
(`aa_`), then the base venv's site-packages (`zz_`). Then it installs the package editable with `--no-deps`.
The extension is copied from the main checkout when that checkout is at the same commit and built; `--build`
builds it anyway. It refuses unless the editable finder maps `rola` to the new worktree, and it imports `rola`
from inside the worktree. The three footguns this avoids are in `docs/setup.md` §3c.

**`get`** prints one value for a shell caller. `cuda-home` resolves the toolkit, which the clang-tidy device
scripts pass as `--cuda-path`.

**Running a CPU-heavy command under the host budget** is `python tools/host_budget.py [--slots N]
[--exclusive] -- <cmd>`, niced per `host.nice`. `setup.py` takes the budget itself.

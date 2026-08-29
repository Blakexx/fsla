# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# THE DEV CONTAINER (K37). USERS get a wheel (prebuilt per-arch binaries, a driver
# dependency, closed-world -- see README.md). DEVELOPERS get this: a box that can
# compile the kernels under the SAME toolchain the ratified manifests were measured
# under, plus the reading/searching/profiling tools the queue's briefs assume.
#
# BASE: nvidia/cuda:12.4.1-devel-ubuntu22.04, ubuntu22.04 for the apt package
# surface the rest of this file needs (deadsnakes, clangd-14, ...).
#
# THE BASE IMAGE'S OWN nvcc/ptxas DO NOT MATCH THE RATIFIED MANIFEST, AND THIS
# WAS MEASURED, NOT ASSUMED. `docs/build.md` rule 1 / `setup.py:_gate_toolchain`
# check `ptxas --version`'s WHOLE string, byte for byte, against
# `tools/manifests/sm_XX.json`; the manifests were ratified under
# `V12.4.99, Build cuda_12.4.r12.4/compiler.33961263_0` (Built
# Tue_Feb_27_16:15:50_PST_2024). BOTH `nvidia/cuda:12.4.0-devel-ubuntu22.04` AND
# `12.4.1-devel-ubuntu22.04` were pulled and checked (2026-08-29) and both report
# `V12.4.131, Build cuda_12.4.r12.4/compiler.34097967_0` (Built
# Thu_Mar_28_02:14:54_PDT_2024) instead -- a LATER CUDA 12.4 toolkit package
# revision than what the host (and the manifests) were measured under, even
# though every current Docker Hub tag under "12.4" ships it. So the RUN block
# below does not trust the base image's toolchain: it explicitly downgrades
# `cuda-nvcc-12-4`/`cuda-nvvm-12-4`/`cuda-crt-12-4` to the exact `12.4.99-1`
# package version via NVIDIA's own apt repo (already configured by this base
# image), which the same 2026-08-29 check confirmed reproduces the manifest's
# pinned string BYTE FOR BYTE. `12.4.99-1` was still present in NVIDIA's repo
# at check time; if a future NVIDIA repo prune ever removes it, this build
# fails at that RUN step with an explicit apt "unable to locate package"
# error, not a silent toolchain swap -- `setup.py`'s rule-1 gate is the
# second, independent line of defense that would catch it even if this one
# did not (a build under the wrong `ptxas` refuses at the FIRST `pip install`,
# never producing an unmeasured binary).
#
# Everything else pinned in this file (sccache, mold, hyperfine, difftastic) is
# pinned the SAME way the repo pins its own vendored inputs (`tools/sccache_pin.json`,
# `csrc/third_party/cutlass/PIN.json`): an exact version, an exact upstream release
# asset, and a sha256 verified before the archive is trusted -- so a base-image
# rebuild can never silently swap in a different build of any of them.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIPX_HOME=/opt/pipx \
    PIPX_BIN_DIR=/usr/local/bin

# --------------------------------------------------------------------------
# ROLA_IMAGE_DIGEST -- stamped by the build (below), read by
# `tools/ratify.py` as `source.image_digest` (K37). Set to the DIGEST of the
# Dockerfile itself (not a moving `latest` tag, and not the eventual pushed
# image ID, which does not exist until after the push this repo never does):
# "was this ratification's compile run under the toolchain this exact
# Dockerfile describes" is the checkable question, and a Dockerfile digest
# answers it without needing a registry. Computed as a build-arg so a
# `docker build` from the checked-in file can set it without a second pass.
ARG ROLA_IMAGE_DIGEST=""
ENV ROLA_IMAGE_DIGEST=${ROLA_IMAGE_DIGEST}

# --------------------------------------------------------------------------
# APT PACKAGES -- everything with a real Ubuntu-22.04 (jammy) package. Left
# out on purpose: `docker.io`/nvidia-container-toolkit (host-side,
# docs/setup.md), `hyperfine`/`mold`/`difftastic` (not in 22.04's repos --
# pinned release binaries below). CUDA/ptxas is a SEPARATE, later RUN block
# (below): the base image's own nvcc/ptxas do not match the pin, see the
# FROM-line comment at the top of this file.
#
# PYTHON 3.11 IS NOT IN JAMMY'S DEFAULT REPOS (jammy ships 3.10). The
# deadsnakes PPA is the standard, widely-used source for it; added here
# rather than building CPython from source, which would be a second,
# unpinned toolchain to maintain for no benefit over a maintained PPA build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg2 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        ninja-build \
        graphviz \
        sqlite3 \
        libsqlite3-dev \
        clangd-14 \
        clang-tools-14 \
        clang-tidy-14 \
        universal-ctags \
        jq \
        fd-find \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        pipx \
        vim \
        less \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/clangd-14 /usr/local/bin/clangd \
    && ln -sf /usr/bin/clang-query-14 /usr/local/bin/clang-query \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# THE EXACT ASSEMBLER -- downgrade nvcc/ptxas AWAY from whatever this base
# image tag currently ships (measured 12.4.131, see the FROM-line comment
# above) to the exact `12.4.99-1` package build the ratified manifests pin.
# `cuda-nvvm-12-4`/`cuda-crt-12-4` are `cuda-nvcc-12-4`'s own versioned
# dependencies and must be pinned in the SAME command or apt refuses the
# downgrade with an unmet-dependency error (measured). `--allow-downgrades`
# is required because the base image ships a NEWER package than this pin.
RUN apt-get update && apt-get install -y --allow-downgrades \
        cuda-nvcc-12-4=12.4.99-1 \
        cuda-nvvm-12-4=12.4.99-1 \
        cuda-crt-12-4=12.4.99-1 \
    && rm -rf /var/lib/apt/lists/*
#: FAILS THE BUILD, LOUDLY, IF THE DOWNGRADE DID NOT TAKE: `docs/setup.md`'s
#: own "verify the toolchain" step is this same string comparison run by
#: hand; asserting it here means a future apt-repo change that silently
#: reintroduces a newer `ptxas` (e.g. a version bump inside a mirrored cache)
#: is caught at BUILD time, not discovered later at `pip install -e .`'s own
#: rule-1 gate -- which would still catch it, but a build that appears to
#: succeed while carrying the wrong assembler is exactly the silent-drift
#: shape the whole closed-world design refuses.
RUN ptxas --version | grep -q "V12.4.99" || \
    (echo "FATAL: ptxas is not V12.4.99 after the pinned downgrade:" && ptxas --version && exit 1)

# --------------------------------------------------------------------------
# uv -- the Python resolver/installer this repo's ONE lock file
# (`requirements.lock`, generated by `uv pip compile`, see `requirements.in`'s
# header and `docs/setup.md`) is built with. Pinned release, sha256-verified
# against upstream's own published `.sha256` (same recipe as
# `tools/sccache_pin.json`).
ARG UV_VERSION=0.12.7
ARG UV_SHA256=788f18abea7c5f55d6216e4f5613fd89d4d59b631efeec117b2b07fe72f1da21
RUN curl -sSL -o /tmp/uv.tar.gz \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/uv.tar.gz -C /tmp \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv \
    && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx \
    && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

# --------------------------------------------------------------------------
# sccache -- REUSES THE REPO'S OWN PIN, not a second one. The exact asset,
# tarball sha256, and binary sha256 are read out of `tools/sccache_pin.json`
# at build time so this file cannot drift from the pin the build/ratify gates
# already check (`tools/sccache_toolchain.py`) -- a Dockerfile hand-copy of
# those bytes would be a second place the version could go stale.
COPY tools/sccache_pin.json /tmp/sccache_pin.json
COPY docker/verify_sccache.py /tmp/verify_sccache.py
RUN python3 /tmp/verify_sccache.py \
    && rm -rf /tmp/sccache.tar.gz /tmp/sccache_extract /tmp/sccache_pin.json /tmp/verify_sccache.py

# --------------------------------------------------------------------------
# mold -- pinned release, sha256-verified (not in 22.04's apt repos).
ARG MOLD_VERSION=2.42.0
ARG MOLD_SHA256=f5ed2f6e31d1ada4f07fe766fe0de7a73104d1c5cdc59086fcecc16a43720b6d
RUN curl -sSL -o /tmp/mold.tar.gz \
        "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz" \
    && echo "${MOLD_SHA256}  /tmp/mold.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/mold.tar.gz -C /opt \
    && ln -sf "/opt/mold-${MOLD_VERSION}-x86_64-linux/bin/mold" /usr/local/bin/mold \
    && rm -f /tmp/mold.tar.gz

# --------------------------------------------------------------------------
# hyperfine -- pinned release, sha256-verified (not in 22.04's apt repos).
ARG HYPERFINE_VERSION=1.20.0
ARG HYPERFINE_SHA256=63ad53934062118f5b0be11785e0bb1603d4b91667d1921f2fd8df9a8712040a
RUN curl -sSL -o /tmp/hyperfine.tar.gz \
        "https://github.com/sharkdp/hyperfine/releases/download/v${HYPERFINE_VERSION}/hyperfine-v${HYPERFINE_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${HYPERFINE_SHA256}  /tmp/hyperfine.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/hyperfine.tar.gz -C /tmp \
    && install -m 0755 "/tmp/hyperfine-v${HYPERFINE_VERSION}-x86_64-unknown-linux-gnu/hyperfine" /usr/local/bin/hyperfine \
    && rm -rf /tmp/hyperfine.tar.gz "/tmp/hyperfine-v${HYPERFINE_VERSION}-x86_64-unknown-linux-gnu"

# --------------------------------------------------------------------------
# difftastic -- pinned release, sha256-verified. NOT installed via pipx: it
# has no PyPI package (verified against the index at authoring time -- a
# Rust binary with no Python wheel, unlike `ast-grep-cli`/`py-spy` below), so
# it is pinned the same way as mold/hyperfine rather than forced through a
# tool that cannot actually install it.
ARG DIFFT_VERSION=0.70.0
ARG DIFFT_SHA256=2997d2bbe620534edbd79b0049f00ce84eef3fedb15c7822456d58e38d8b05c9
RUN curl -sSL -o /tmp/difft.tar.gz \
        "https://github.com/Wilfred/difftastic/releases/download/${DIFFT_VERSION}/difft-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${DIFFT_SHA256}  /tmp/difft.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/difft.tar.gz -C /usr/local/bin \
    && chmod 0755 /usr/local/bin/difft \
    && rm -f /tmp/difft.tar.gz

# --------------------------------------------------------------------------
# pipx tools -- ast-grep-cli, py-spy: real PyPI packages, so pipx is the
# actual right tool for them (difftastic is not one, see above).
# pre-commit is also installed here (a dev tool, not a repo runtime
# dependency) so it is present even before the Python lock's venv exists.
#
# clang-format==19.1.7 IS PINNED, NOT A FREE CHOICE: `.clang-format`'s own
# header states "Pinned tool: clang-format 19.1.7 (pip package
# `clang-format==19.1.7`)" and `tools/gen_shards.py`'s `--check` shells out to
# whatever `clang-format` resolves on PATH to compare against the committed
# generated shards -- found the hard way (2026-08-29): this image originally
# carried ONLY apt's `clang-tools-14` family (clangd/clang-tidy/clang-query,
# all fine for editor tooling) and NO `clang-format` at all, so a fresh
# in-container build's `pip install -e .` failed at `setup.py`'s
# `_gate_shards` with a generic "shards do not match the arm list" message
# that was actually `FileNotFoundError: clang-format` swallowed by the
# subprocess-returncode check -- a version 14 apt package would have been
# the WRONG fix anyway, since a formatter-version mismatch against files
# committed under 19.1.7 reads as a false "shards drifted" failure, the same
# failure shape as a real drift. pipx installs the exact pinned PyPI package.
RUN pipx install ast-grep-cli \
    && pipx install py-spy \
    && pipx install pre-commit \
    && pipx install clang-format==19.1.7

# --------------------------------------------------------------------------
# THE PYTHON ENVIRONMENT -- ONE LOCK FILE, `requirements.lock` (generated by
# `uv pip compile requirements.in`, `uv` being present on this box -- the
# card's "uv lock, or pip-compile if not" -- see `requirements.in`'s own
# header and `docs/setup.md`), replacing the hand-grown dev venv this box
# iterated in. `rola` itself is installed separately, `--no-build-isolation`,
# for the same reason `docs/build.md` states it: pip's isolation would fetch
# a CPU-only torch and build the extension against the wrong ABI.
WORKDIR /workspace
COPY requirements.lock /tmp/requirements.lock
RUN uv venv /opt/venv --python 3.11 --seed \
    && . /opt/venv/bin/activate \
    && uv pip install -r /tmp/requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# --------------------------------------------------------------------------
# WORKTREES MOUNT POINT. `.devcontainer/devcontainer.json` binds the host
# worktrees directory here; a plain `docker run` does the same with `-v`
# (`docs/setup.md`). No home path baked in (§17): the mount source is a
# host-side argument in both cases, never a value compiled into the image.
RUN mkdir -p /workspace/worktrees

CMD ["/bin/bash"]

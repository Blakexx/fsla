# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
#
# THE DEV CONTAINER (K37). USERS get a wheel (prebuilt per-arch binaries, a driver
# dependency, closed-world -- see README.md). DEVELOPERS get this: a box that can
# compile the kernels under the SAME toolchain the ratified manifests were measured
# under, plus the reading/searching/profiling tools the queue's briefs assume.
#
# THE TOOLCHAIN COMES FROM ITS RECORD (tools/toolchains/<name>.json, docs/internals/tools/toolchains.md).
# `tools/dev.py container build` passes the record's `container.base_image` (CUDA_BASE_IMAGE), its exact apt
# `container.packages` (CUDA_PACKAGES), its `torch_index` (TORCH_INDEX) and its name (ROLA_TOOLCHAIN). A base tag's own
# nvcc is not trusted: the packages are installed at their pinned versions whatever the tag ships (a tag can carry a
# newer patch -- measured for 12.4, whose every tag shipped V12.4.131 against a V12.4.99 pin), and the build fails
# unless ptxas, cuobjdump and nvdisasm report the record's strings byte for byte, the comparison rule 1 makes.
# ubuntu22.04 for the apt package surface the rest of this file needs (deadsnakes, clang-tools-14, ...).
#
# sccache, mold, bazelisk and the devbox's tools (hyperfine, difftastic, Nsight Compute and Systems) are installed by
# `tools/dev.py init --image` from the repo's own pins (`tools/*_pin.json`; init links `bazel` onto PATH and the
# devbox's commands into the image's venv's bin, first on PATH) -- the same command and the same pins a bare host uses, so the image and the host
# cannot disagree on any of them; the developer tools the environment carries (py-spy, clangd, clang-format,
# ast-grep) come with requirements.lock. The container-only uv below is pinned the same way: an exact version, an
# exact upstream release asset, and a sha256 verified before the archive is trusted.
#
# THE IMAGE IS KEPT CURRENT MECHANICALLY (KERNEL_STANDARDS §23 (3)): the environment key hashes this file and every
# other input `tools/dev.py` lists in ENV_INPUTS; a commit that changes one carries a passing
# `python tools/dev.py container check` for its key, or the commit gate refuses it.
ARG CUDA_BASE_IMAGE
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------------------------
# APT PACKAGES -- everything with a real Ubuntu-22.04 (jammy) package. Left
# out on purpose: `docker.io`/nvidia-container-toolkit (host-side,
# docs/setup.md) and the devbox's pinned tools (mold, hyperfine, difftastic, Nsight Compute and Systems: init's,
# from their pins). CUDA/ptxas is a SEPARATE, later RUN block
# (below): the base image's own nvcc/ptxas are not trusted, see the
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
        clang-tools-14 \
        universal-ctags \
        jq \
        fd-find \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        vim \
        less \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/clang-query-14 /usr/local/bin/clang-query \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# THE EXACT ASSEMBLER -- the toolchain record's packages at their pinned versions (`--allow-downgrades`: a base tag can
# ship a newer patch than the record). cuobjdump is one of them (the SASS gate, the region ledger and the composer
# extract cubins with it).
ARG CUDA_PACKAGES
RUN apt-get update && apt-get install -y --allow-downgrades ${CUDA_PACKAGES} \
    && rm -rf /var/lib/apt/lists/*
#: FAILS THE BUILD, LOUDLY, IF THE ASSEMBLER OR A READER IS NOT THE RECORD'S (ptxas, cuobjdump, nvdisasm, each its
#: whole `--version` text, rule 1's comparison): a build that appears to succeed while carrying the wrong toolkit is
#: exactly the silent-drift shape the closed-world design refuses.
ARG ROLA_TOOLCHAIN
COPY tools/toolchains/ /opt/rola/bootstrap/tools/toolchains/
RUN python3.11 -c "import json, subprocess, sys; \
record = json.load(open('/opt/rola/bootstrap/tools/toolchains/${ROLA_TOOLCHAIN}.json')); \
norm = lambda text: ' '.join(text.split()); \
want = {'ptxas': record['ptxas'], **record['readers']}; \
bad = [t for t, w in want.items() if norm(subprocess.run([t, '--version'], capture_output=True, text=True).stdout) != norm(w)]; \
sys.exit('FATAL: not toolchain ${ROLA_TOOLCHAIN}: ' + ', '.join(bad) if bad else 0)"

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
# THE PYTHON ENVIRONMENT -- ONE LOCK FILE, `requirements.lock` (generated by
# `uv pip compile requirements.in`, `uv` being present on this box -- the
# card's "uv lock, or pip-compile if not" -- see `requirements.in`'s own
# header and `docs/setup.md`), replacing the hand-grown dev venv this box
# iterated in. `rola` itself is installed separately, `--no-build-isolation`,
# for the same reason `docs/build.md` states it: pip's isolation would fetch
# a CPU-only torch and build the extension against the wrong ABI.
WORKDIR /workspace
ARG TORCH_INDEX
COPY requirements.lock /tmp/requirements.lock
RUN uv venv /opt/venv --python 3.11 --seed \
    && . /opt/venv/bin/activate \
    && uv pip install -r /tmp/requirements.lock \
        --extra-index-url "${TORCH_INDEX}" \
        --index-strategy unsafe-best-match

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# --------------------------------------------------------------------------
# THE IMAGE'S DEV CONFIG -- `tools/dev.py init --image`, the one init every machine runs, from a bootstrap copy of the
# files it needs: the image's fixed layout (the mount points below, which `python tools/dev.py container compose` binds
# the host's lock directories, store and worktrees onto), the pinned sccache and mold installed under /opt/rola/tools,
# then `check --image`. A red check FAILS THE IMAGE BUILD: a tool the tree needs cannot go missing from the image
# unnoticed again. No home path is baked in (§17): mount sources are host-side, never compiled into the image.
# The dev config's reader is rola-devtools' (`tools/devtools.txt`): the bootstrap installs that commit beside itself
# and runs init with it on the path; at run time the image's venv imports the mounted checkout init links.
#
# ROLA_ENV_KEY is the environment key the image is built from (`container build` passes it): sha256 over the
# environment's inputs, recorded as `environment.image_digest`, which `tools/ratify.py` stamps as
# `source.image_digest`. Declared HERE, after every heavy layer, so a new key rebuilds only this step.
ARG ROLA_ENV_KEY=""
ENV ROLA_DEV_CONFIG=/opt/rola/dev-config
#: Under WSL the Windows driver's Linux libraries (libdxcore, libcuda) are mounted read-only at /usr/lib/wsl/lib
#: (`host.wsl_lib`); without them first on the loader path CUDA fails to initialize with error 500 (measured
#: 2026-09-12 on Docker Desktop 20.10.17, driver 595.95). On native Linux the directory does not exist and changes nothing.
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
COPY requirements.lock /opt/rola/bootstrap/requirements.lock
COPY tools/dev.py tools/dev_config.py tools/bootstrap.py tools/sccache_toolchain.py tools/mold_toolchain.py \
     tools/sccache_pin.json tools/mold_pin.json tools/bazelisk_pin.json tools/devtools.txt \
     tools/hyperfine_pin.json tools/difftastic_pin.json tools/nsight_compute_pin.json tools/nsight_systems_pin.json tools/cuda_toolkit_pin.json \
     /opt/rola/bootstrap/tools/
RUN mkdir -p /workspace/rola /workspace/store /workspace/suite /workspace/worktrees /run/rola/locks /run/rola/gpu \
    && uv pip install --python /opt/venv/bin/python --target /opt/rola/bootstrap/devtools \
        -r /opt/rola/bootstrap/tools/devtools.txt \
    && PYTHONPATH=/opt/rola/bootstrap/devtools \
        /opt/venv/bin/python /opt/rola/bootstrap/tools/dev.py init --image --env-key "${ROLA_ENV_KEY}"

CMD ["/bin/bash"]

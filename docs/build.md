# Build — ahead of time, closed world

## The governing constraint

The consumer kernel is compiled **ahead of time** into a fatbin containing
`code=sm_XX` images only. There is no JIT, no PTX for the driver to compile, and no
runtime autotuning. Everything the binary can execute has been assembled by a
`ptxas` whose output was measured and committed
([`ratification.md`](ratification.md)).

That is a deliberate trade. A JIT-and-autotune build (Triton's model) adapts to
hardware nobody tested, at the cost of shipping codegen nobody looked at; the
supported matrix then means "we expect it to work". Here the supported matrix
**is** the ratified set, by construction, and the build refuses to widen it
silently.

## Building

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required: pip's isolation would fetch a CPU-only torch
from PyPI and build the extension against the wrong ABI. Flash-attention and
mamba-ssm require the same thing for the same reason.

| Environment variable | Effect |
|---|---|
| `ROLA_CUDA_ARCHS` | `;`-separated compute capabilities, default `80;86`. Naming one without a ratified manifest **fails before anything compiles** (rule 3). |
| `MAX_JOBS` | Parallel compile jobs. The default is derived from **available RAM**, not core count: `MAX_JOBS=8` was OOM-killed on a 23 GB / 16-core host after 8m32. The divisor is a measured GB-per-`cicc` constant in `tools/build_flags.py`, so the same source gives 7 jobs on a 23 GB host and 4 on a 16 GB-class one. |
| `ROLA_NVCC_THREADS` | `nvcc --threads`, default 2 = the number of `-gencode` targets. Beyond the target count it buys nothing and multiplies the resident set, since the two parallelism axes multiply. |
| `ROLA_SKIP_POST_BUILD_RATIFY` | Skips the post-build gate. For iterating on non-kernel code only; a shippable binary is one that passed it. |

## What the build does, in order

1. **Rule 2 — gencode.** Emits `-gencode arch=compute_XX,code=sm_XX` only. A
   `code=compute_XX` entry would embed PTX and let the driver just-in-time compile
   for an architecture whose register and spill behaviour was never measured.
2. **Rule 3 — ratified archs.** `tools/manifests/sm_XX.json` must exist and be
   non-empty for every arch being built. A file-existence question, asked first,
   because the cheapest check is the one that must never be tempting to skip.
3. **Rule 1 — toolchain.** The host `ptxas --version` must equal the string the
   manifests pin, whitespace-normalized. The whole output, not a parsed release
   number: the build string distinguishes two shipments of one nominal release, and
   it is exactly those that can move an allocation while the version says nothing
   changed.
4. **Shards.** The checked-in instantiation shards must be the ones
   `tools/gen_shards.py` produces. Generated sources are committed so that what is
   reviewed is what compiles; a stale one fails here rather than shipping a matrix
   nobody wrote.
5. **Compile.** `ninja`. The consumer's 512 kernel instantiations are spread over
   eight translation units (`csrc/rola/src/instantiations/`), so the matrix
   parallelizes across `MAX_JOBS` processes instead of serializing inside one
   `cicc`.
6. **The fatbin gate.** `tools/fatbin_entries.py --check` reads the built `.so` and
   requires exactly one device entry per instantiation per architecture, each in
   the shard the matrix assigns it, and every ratified name present. This is the
   only gate that can see a *failed* split: duplicated device code leaves the build
   green, the manifest byte-identical and `nm` at 512, because a `__global__`
   template's host stub is weak and the linker keeps one.
7. **Post-build ratification.** `tools/ratify.py --arch …` for exactly the archs
   just built — the pre-build gates prove the *inputs* were ratified; this proves
   the codegen still matches what was measured.
8. **`rola/_build_config.py`.** Generated, carrying the archs, the assembler, the
   torch version and ABI flag, the manifest digest and per-arch entry counts, and
   `ptx_jit_fallback: False`.

## What it costs

MEASURED on a 16-core / 23 GB host, clean build directory, both architectures,
including every gate above:

| | wall |
|---|---|
| clean rebuild, `MAX_JOBS` at the derived default | see `workflows/refactor/p02_landing.md` |
| edit inside one shard's arms (one shard + link + gates) | idem |
| the post-build ratification alone | idem |

The dominant term used to be one `nvcc` process holding the whole 512-instantiation
matrix: 411 s of wall, 4.4 GB of resident set, and no way to spend a second core on
it. A shard is 43 s and 0.61 GB.

## Verifying a built binary

```bash
python -c "from rola import _build_config; print(_build_config.show())"
cuobjdump --list-ptx rola_cuda.cpython-*.so     # must list NOTHING
python tools/ratify.py --arch 80 --arch 86       # must PASS
python tools/fatbin_entries.py --check rola/_C*.so   # must PASS
```

The shipped `.so` carries 8 ELF images and **0 PTX images**. That is not a
side effect; it is rule 2, checked.

## Dependencies

`torch` and `einops`. Not `transformers` (the HF model classes stay in the fork),
not `fla`, not `triton` — the consumer is CUDA. Optional interop lives behind
extras (`[transformers]`, `[fla]`) and `rola` itself imports neither.

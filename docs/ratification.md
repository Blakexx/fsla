# Ratification — the closed-world codegen policy

A **ratification manifest** is a measured, committed record of what the assembler
did to every instantiation of the consumer kernel, on one architecture, under one
exact `ptxas`:

```
tools/manifests/sm_86.json
  { "toolchain": { "ptxas": "<the whole --version string>", "defines": {},
                   "nvcc_flags": [...], "sass_hash_version": 2,
                   "sccache": {"version": "sccache 0.17.0", "sha256": "..."} },
    "source":   { "csrc_sha256": "...", "date": "..." },
    "shards":   { "consumer_gt16_decay_bc16": { "members_sha256": "...",
                                                "count": 8 }, ... },
    "entries":  { "sm_86:<mangled name>": { "regs": 64, "spill_stores": 8,
                                            "spill_loads": 24,
                                            "hot_locals": 0, "local_ops": 12,
                                            "sass_sha256": "..." }, ... } }
```

144 entries per architecture, measured on the current tree: 64 for the consumer —
one per point of the `consumer_forward_kernel<BC, DV, D, DECAY, GT>` product (the
P26 endgame executor collapsed the matrix from 256 to 32 by deleting the `BT`,
`PREC` and `GLOBAL` axes, and the window batch then widened it back to 64 by adding
`GT`; [`docs/internals/DELETIONS.md`](internals/DELETIONS.md)) — plus 16
`decode_gemv_kernel` (`DV x D x DECAY`; its own `GLOBAL` axis died with raw
normalization the P26 sweep) and 32 each of `union_forward_kernel` and
`union_backward_kernel`, whose matrix the window batch did not touch. `defines` is
the empty tuple — `RATIFIED_DEFINES = ()` in `tools/ratify.py`, since the
closed-world codegen surface carries no `#define` this gate has to pin. Two
ratified architectures (`sm_80`, `sm_86`) make 288 entries total. The manifest is
the authority on the count and no comment restates it independently. The mangled
name is the key because it is exact, stable, and encodes every template argument
— an entry cannot be silently re-pointed at a different instantiation by a rename
or a reordering.

## Why it exists

The kernel's `__launch_bounds__` is computed from a three-term residency formula,
and a spill accepted as a measured optimum against worse alternatives is a
judgement that has to live SOMEWHERE nothing else checks — a document that says so
is not enforced, and a later codegen change can grow the spill silently. The
manifest is what makes that judgement an asserted invariant instead: a property
the project believes is a property the project checks. **The ratchet is a
ratchet, not a spill ban** (P25's recorded adjudication): a spill is admissible
when it is priced, recorded here, and worse than every alternative in hand — not
when it is absent.

On the current tree the consumer's 32-arm matrix ratifies at **zero spill on every
arm but one**: `<BC=64, DV=64, D=4, DECAY=true>` — the widest carve in the matrix —
holds 4 bytes stores / 4 bytes loads. That is the register budget's doing rather than
an accident of it. The budget is `128`, which puts every consumer arm at two CTAs/SM,
and two CTAs is the first rung at which `ptxas` is given the kernel's measured
121-register working set; at three (the previous budget's rung, a cap of 85) the
flagship spilled 52/140 (`internals/smem_layout.md#reg-budget` carries the whole
spill-vs-bound curve, which has no interior point). The occupancy that raise costs is
priced against what it bought — zero spill, the visit's second CTA-wide barrier
deleted, and staging depth up on every arm — in `workflows/perf/p50_remedy.md`; it is
a trade, recorded, and a successor's to overturn with new evidence.

The other two live spilling entries in the manifest are both `union_backward_kernel`
(entmax), 16 bytes stores / 16 bytes loads each on `sm_86`, unrelated to the consumer.

## What it compiles, and why that is the shipped source

`python tools/ratify.py` compiles **the translation units the build compiles** —
the eight generated instantiation shards under `csrc/rola/src/instantiations/`,
plus `decode.cu` and `entmax.cu` — under the flag list `setup.py` uses, imported
from `tools/build_flags.py` rather than restated. Two properties follow, and
neither held before:

- **the numbers describe the shipped binary.** `nvcc` derives an
  anonymous-namespace kernel's mangled discriminator from the file that
  instantiates it, so while the gate compiled a probe under `tools/`, its 80
  decode/entmax keys named symbols that existed in no shipped `.so`. They now name
  the entry points the binary carries, and `tools/fatbin_entries.py` checks that
  rather than asserting it.
- **there is one enumeration of the matrix.** The shards, the dispatch table and
  this gate's source list all come out of `tools/gen_shards.py`; there is no second
  file that could enumerate a different 64.

The consumer's 64 keys are unaffected by the split by construction:
`consumer_forward_kernel` lives in a NAMED namespace, so its mangled name does not
depend on which translation unit expands it.

`--shard NAME` scopes the compile to one shard's 8-entry slice for triage. It is a
measurement scope only: `--write` refuses it, because a manifest written from a
subset would silently drop every instantiation it did not compile.

## What the gate refuses

The gate reads `ptxas -v` and the disassembly of the same objects. Its outcomes,
and the difference between them, are deliberate:

| Outcome | Meaning |
|---|---|
| **FAIL — toolchain drift** | The invoked `ptxas` is not the one the manifest was ratified under. A spill/register manifest is a property of the *assembler* as much as of the source, so these numbers say nothing about this assembler — in **both** directions. Re-ratification is required. |
| **FAIL — spill growth** | An instantiation spills more than its manifest entry. Spills are the thing being bounded. |
| **FAIL — an unhonored bound** | `ptxas` reported *"Value of threads per SM … is out of range"*. That entry compiled with **no launch bound at all**, so the assembler was free to spend registers as it liked. A hard failure even at zero spills. |
| **FAIL — source drift** | The tree's `csrc/rola/src` digest is not the one the manifest was measured against. |
| **FAIL — hot-loop growth** | An instantiation executes more `LDL`/`STL` *inside a loop* than it ratified. In-loop local traffic is a memory round-trip per iteration, disqualifying whatever the byte count. |
| **FAIL — local-memory growth** | An instantiation executes more `LDL`/`STL` *anywhere* than it ratified. A cheap complement to the two above: `ptxas`'s spill bytes miss local traffic that is not a spill (a dynamically indexed local array), and the in-loop count only sees a backward-branch interval. |
| **FAIL — SASS body drift** | An instantiation assembles to different code than was ratified. |
| **REPORT — register drift** | A changed register count with no spill growth is printed, not failed. Register allocation is an outcome that legitimately moves; what must not move is the spilling. |
| **REPORT — sccache drift** | The recorded `toolchain.sccache` (`{version, sha256}` or `null`) differs from this run's. Printed, not failed: `sccache` is a measured-transparent pass-through (`docs/build.md#sccache`), so its version cannot move a ratified register/spill number the way a `ptxas` swap can. What IS gated hard, before it is used at all, is that a resolved `sccache` matches `tools/sccache_pin.json` (`tools/sccache_toolchain.py`). |

### The body hash is the incremental half of the freshness trigger

The source digest is **wholesale**: any edit under `csrc/rola/src` moves it and
invalidates all ratified entries at once. That is right as a trigger and useless as
a diagnosis. `sass_sha256` — the sha256 of an instantiation's assembled instruction
text, whitespace-normalised, encoding words included, listing padding excluded — is
the per-entry answer to what the digest can only ask globally, and it reads two ways:

* **the digest matches** → every ratified body must still hash the same. The digest
  says the *inputs* did not move; this says the *output* did not either, which is an
  independent statement and the one the spill numbers depend on.
* **the digest moved** → the numbers must be re-measured, but the entries whose
  bodies are unchanged are provably still described by their ratified numbers, and
  the gate names only the moved ones. That is what makes a scoped `--shard`
  re-measurement a checkable claim rather than an assertion.

**Maps may now come from binaries built at different paths.** `decode.cu` and
`entmax.cu` used to place their kernels in an anonymous namespace, so the mangled
name embedded a discriminator derived from the source path; a manifest written in a
worktree named 192 kernels that no build at any other path would ever define. Fixed
by moving those kernels into a named internal namespace (P19,
`workflows/refactor/p19_symbols.md`) — the ratified names are a function of source
content and toolchain only, and a worktree build now ratifies clean. History:
`docs/build.md`'s path-trap section.

**And from the same SCANNER.** What reaches the digest is decided by
`ratify.sass_scan`, so widening the capture moves every hash in an unchanged binary
— a false verdict with exactly the (now-fixed) path trap's shape. `ratify.SASS_HASH_VERSION` is
the one definition of that decision; it is stamped into the manifest's `toolchain`
block and into every map `tools/sass_bodies.py` writes, and both refuse a comparison
that does not agree on it. A record carrying no stamp is refused rather than assumed
to be version 1, because guessing produces a confident wrong answer.

## The four closed-world rules

| Rule | Refused | Enforced by |
|---|---|---|
| **R1** | building under an assembler the manifest was not ratified under | `setup.py::_gate_toolchain`, `ratify.py::check_toolchain` |
| **R1b** | building under CODEGEN FLAGS the manifest was not ratified under | `ratify.py::check_flags` |
| **R2** | emitting `code=compute_XX` — any PTX a driver could JIT | `setup.py::gencodes`, asserted on the shipped `.so` (`cuobjdump --list-ptx` reports **0 PTX images**) |
| **R3** | an architecture in the fatbin with no ratified manifest | `setup.py::_gate_ratified` — a **file-existence** question, asked before anything compiles |
| **R4** | launching on a device the kernel's arch table does not describe | `consumer_check_arch_table()` at run time, `static_assert` at compile time |

There is no environment variable that turns any of these off.

## Re-ratification is a measurement, not an edit

The manifest is regenerated only by `--write`, which re-runs the same compile and
records what it reports. Hand-editing an entry to make the gate pass would be
asserting a spill nobody measured — exactly the failure the manifest replaces. When
a kernel change legitimately moves the numbers: re-run `--write`, and record why
they moved.

## One file per architecture

`--arch` selects what is measured, checked, and written, so bringing up `sm_89`
writes one reviewable file and cannot restate a measurement it did not take. The
split is **filing only**, and that is checked rather than claimed: the digest
`_build_config.py` records is taken over the merged canonical
`{toolchain, entries, shards}` record with sorted keys, so N files hash to exactly
what one file hashed to. Two manifests pinning different assemblers is a hard
failure in every reader, and so is two manifests disagreeing about shard
membership — a split filing may not become a split toolchain or a split matrix.

## The `shards` block, and what it makes provable

`shards` records WHICH instantiations this ratification covers, as one membership
hash per shard: the sha256 of the shard's sorted list of template tuples, not of
its file's bytes. A comment edit does not move it; adding, dropping or moving one
instantiation does.

It enters the digest, and that is the point. "Only `consumer_gt16_decay_bc16` was
re-measured" is otherwise an unprovable sentence; with membership in the ratified
record it is a diff. Because the block carries membership and nothing else — no
paths, no ordering — re-filing the same measurements cannot move the digest, which
`--self-test` checks in both directions (a re-filing must not move it, a membership
change must).

## The gate self-tests

```bash
python tools/ratify.py --self-test
```

requires the gate to **fail** on the things it must reject:

1. a manifest tightened at one entry,
2. a perturbed `ptxas` version string,
3. an architecture with no committed manifest (so rule 3 has a *no* answer),
4. a per-arch split that does not rejoin to one digest,
5. per-arch files that disagree about shard membership,
6. a shard block that is decorative (a membership change that moves nothing) or one
   that is a property of the filing (a re-filing that moves the digest),
7. a perturbed source digest,
8. in-loop local-access growth past a ratified count — plus the hot-loop parser
   proving, on synthetic SASS, that it counts an in-loop `LDL` and clears a cold
   one,
9. local-access growth on the `LDL` the hot-loop gate is right to ignore, which is
   what shows the cheap ratchet is not a restatement of the expensive one,
10. a ratified `sass_sha256` that no longer matches, and an entry carrying none at
    all — plus the hash itself proving, on synthetic SASS, that two different
    instruction texts hash differently and that re-padding one listing does not
    move its hash.

A gate whose green run is unfalsifiable says nothing. This project has already
shipped two gates that turned out to be vacuous before they were fixed, which is
why the self-test is part of the gate rather than a note about it.

## Where the record lives after the build

`rola/_build_config.py` is generated by `setup.py` and answers *"which measured
evidence backs this exact `.so`?"* from an installed wheel with no repository
checkout:

```python
>>> from rola import _build_config; _build_config.show()
{'version': ..., 'archs': ['sm_80', 'sm_86'], 'ptxas': ...,
 'manifest_sha256': '...', 'manifest_entries': {'sm_80': 112, 'sm_86': 112},
 'ptx_jit_fallback': False, 'vendored': {'cutlass': {...}}}
```

The tier-2 census independently recomputes `manifest_sha256` from the committed
manifests and requires it to match, so the tests and the binary cannot be looking
at different evidence.

# Ratification — the closed-world codegen policy

A **ratification manifest** is a measured, committed record of what the assembler
did to every instantiation of the consumer kernel, on one architecture, under one
exact `ptxas`:

```
tools/manifests/sm_86.json
  { "toolchain": { "ptxas": "<the whole --version string>", "defines": {...} },
    "source":   { "csrc_sha256": "...", "date": "..." },
    "shards":   { "consumer_bf16_raw_bt32": { "members_sha256": "...",
                                              "count": 64 }, ... },
    "entries":  { "sm_86:<mangled name>": { "regs": 64, "spill_stores": 8,
                                            "spill_loads": 24,
                                            "hot_locals": 0 }, ... } }
```

608 entries per architecture, measured on the current tree: 512 for the consumer,
one per point of the
`(BT, BC, DV, D, FOLD_PREC, READ_PREC, DECAY, GLOBAL, COMPACT)` product, plus 32
`decode_gemv_kernel` and 32 each of `union_forward_kernel` and
`union_backward_kernel`. The manifest is the authority on that count and no
comment restates it. The mangled name is the key because it is exact, stable, and
encodes every template argument — an entry cannot be silently re-pointed at a different instantiation by a
rename or a reordering.

## Why it exists

The kernel's `__launch_bounds__` is computed from a three-term residency formula,
and one instantiation was **accepted with a 4-byte spill as the measured optimum**
— the alternatives spill worse (unbounded: 20 st / 56 ld; a thread-slot cap:
70 st / 284 ld). That acceptance was a judgement recorded in a document, and it
lived nowhere else: **nothing failed if a later codegen change grew it.** The
manifest makes it an asserted invariant. A property the project believes is a
property the project checks.

## What it compiles, and why that is the shipped source

`python tools/ratify.py` compiles **the translation units the build compiles** —
the eight generated instantiation shards under `csrc/rola/src/instantiations/`,
plus `decode.cu` and `entmax.cu` — under the flag list `setup.py` uses, imported
from `tools/build_flags.py` rather than restated. Two properties follow, and
neither held before:

- **the numbers describe the shipped binary.** `nvcc` derives an
  anonymous-namespace kernel's mangled discriminator from the file that
  instantiates it, so while the gate compiled a probe under `tools/`, its 96
  decode/entmax keys named symbols that existed in no shipped `.so`. They now name
  the entry points the binary carries, and `tools/fatbin_entries.py` checks that
  rather than asserting it.
- **there is one enumeration of the matrix.** The shards, the dispatch ladder and
  this gate's source list all come out of `tools/gen_shards.py`; there is no second
  file that could enumerate a different 512.

The consumer's 512 keys are unaffected by the split by construction:
`consumer_forward_kernel` lives in a NAMED namespace, so its mangled name does not
depend on which translation unit expands it.

`--shard NAME` scopes the compile to one shard's 64-entry slice for triage. It is a
measurement scope only: `--write` refuses it, because a manifest written from a
subset would silently drop every instantiation it did not compile.

## What the gate refuses

The gate reads `ptxas -v`. Four outcomes, and the difference between them is deliberate:

| Outcome | Meaning |
|---|---|
| **FAIL — toolchain drift** | The invoked `ptxas` is not the one the manifest was ratified under. A spill/register manifest is a property of the *assembler* as much as of the source, so these numbers say nothing about this assembler — in **both** directions. Re-ratification is required. |
| **FAIL — spill growth** | An instantiation spills more than its manifest entry. Spills are the thing being bounded. |
| **FAIL — an unhonored bound** | `ptxas` reported *"Value of threads per SM … is out of range"*. That entry compiled with **no launch bound at all**, so the assembler was free to spend registers as it liked. A hard failure even at zero spills. |
| **REPORT — register drift** | A changed register count with no spill growth is printed, not failed. Register allocation is an outcome that legitimately moves; what must not move is the spilling. |

## The four closed-world rules

| Rule | Refused | Enforced by |
|---|---|---|
| **R1** | building under an assembler the manifest was not ratified under | `setup.py::_gate_toolchain`, `ratify.py::check_toolchain` |
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

It enters the digest, and that is the point. "Only `consumer_bf16_raw_bt32` was
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
   one.

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
 'manifest_sha256': '...', 'manifest_entries': {'sm_80': 608, 'sm_86': 608},
 'ptx_jit_fallback': False, 'vendored': {'cutlass': {...}}}
```

The tier-2 census independently recomputes `manifest_sha256` from the committed
manifests and requires it to match, so the tests and the binary cannot be looking
at different evidence.

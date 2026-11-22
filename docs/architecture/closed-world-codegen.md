# Closed-world codegen

> **Only measured binaries run.**

The engine ships a fixed, enumerated set of kernel instantiations, compiled ahead
of time by a pinned assembler: the arms each kernel declares, the architectures
the hardware table builds. A device whose architecture the binary does not carry
is refused at launch. There is no path by which unmeasured code executes.

The mechanics — the declared targets, the gates every build runs, the census, the
self-test — are documented once, in
[`docs/ratification.md`](../ratification.md). This document is the *why*, which
that file deliberately does not repeat.

## 1. The problem: codegen cannot be commanded, only pinned

Register allocation and instruction selection are not controllable from CUDA
source. PTX registers are virtual; the assembler decides. There are no supported
SASS assemblers. So the usual engineering answer — *specify the output* — is not
available.

What *is* available: **compilation is deterministic per (source, toolchain,
flags, architecture)**. The same four inputs give the same SASS, every time. So
the way to get a deterministic binary is not to constrain the allocator but to
**close the input space**.

*Reading the numbers back* — compile, measure registers and spills, assert they
are within bounds — is **not** the sufficiency argument. The defect, in the
project owner's ruling: "read them back is not structurally good enough when we
are only testing in a limited architectural regime … the only way to be sure is
deterministic code and rules" (user ruling, 2026-08-01). A read-back gate proves
a property of the binary you happened to build, on the architecture you happened
to have. It says nothing about the binary a different assembler would produce for
an architecture nobody measured. Read-back has a narrower job in this design:
**drift detection on a build you chose to measure**.

## 2. The rules

The rules and their enforcement points are one table, in
[`docs/ratification.md`](../ratification.md#the-closed-world-rules); this section
keeps only their argument.

R2 is the no-fallbacks rule applied to the compiler: an unknown architecture must
**fail to load**, not silently JIT unmeasured code. R3 and R4 are what make
"supported regime" a fact by construction rather than a claim in a README — the
supported set *is* the built set. Bringing up new hardware or a new compiler
therefore requires a bring-up (a row's window, a toolchain record, the census and
the gates) before anything executes.

There is no environment variable that disables any of them.

## 3. The JIT fork is rejected

FlashAttention's active line compiles by **runtime JIT rather than AOT**.
That is a considered position by strong engineers, and the pressures behind it
are real: an AOT matrix over `(tile shape × dtype × head dim × architecture …)`
grows multiplicatively, compile times grow with it, and a JIT path compiles
exactly the one kernel a user's call needs.

**FSLA does not follow it.** The AOT-only build is a decision taken against that
alternative — not an oversight and not an un-revisited default — and this page is
its record.

**The reason is reproducibility for future readers — not process overhead.**
A JIT path means the binary that produced a published number does not exist
anywhere afterwards. It was assembled on the reader's machine, by the reader's
driver, from source the reader has plus a compiler the reader did not pin. Every
claim this project makes about the kernel — a register count, a spill count, a
latency, a bit-identity — is a claim about a *specific* stream of instructions.
Under JIT those claims degrade into claims about a *family* of instruction
streams that were never all measured, and the degradation is silent: the numbers
still print.

The failure mode is concrete rather than imagined, and this engine has held the
instance: an instantiation once carried a **4-byte spill accepted as the measured
optimum**, because every alternative spilled worse. A judgement of that kind
recorded only in a document is enforced by nothing: a codegen change can grow the
spill with no gate failing. Under AOT the binary is the same bytes for the same
inputs, so the spill is a number that can be read from it, and it is: the SASS gate
([`KERNEL_STANDARDS.md`](../KERNEL_STANDARDS.md#s20)) runs on every build and refuses a nonzero
stack frame (its `stack_memory` rule, the `LDL`/`STL` instructions named beside it) on every function of the
extension, whatever its size. Its one declared exception is an instrument's probe naming a
step signature it keeps on purpose with its cost (a declaration of the probe's unit), and that is a step
signature (branches, the conversion unit, global memory inside an MMA step), never a stack frame: a spill
the design accepts has no declaration to be accepted by, so it is red until the gate is amended to take
one. The census (`bazel run //:ratify -- --census`, [`ratification.md`](../ratification.md)) reports each
instantiation's spill beside it, never gating it. Under JIT none of this could be read at all, because
the entries would not exist until run time, on a machine the project does not control.

The cost of the decision is honest and it is paid: the instantiation matrix is a
budget item that every design must price (a design that doubles compile time
twice is not shippable), and translation-unit sharding is a standing
precondition for growing it. That is the trade — **compile-time cost bounded and
visible, in exchange for run-time behaviour that is enumerable.** A project whose
central claims are numeric buys the second with the first.

## 4. Consequences a contributor should expect

- **A new architecture is a bring-up**, not a `-gencode` line. Naming `sm_XX` that
  the hardware table does not build fails the build (rule 3).
- **A toolchain bump is a re-measurement of every architecture**, in both
  directions. The numbers are a property of the assembler as much as of the
  source, so they say nothing about a different assembler even if they got
  *better*; the toolchain record names exactly one.
- **The census is a measurement, not an edit.** Its numbers are read from the
  build's own compiles, never typed; a change that moves one is read in the census
  before and after.
- **A refusal is cleared by changing the change.** When a gate refuses, the change
  is what must move: the in-tree precedent is a measured optimization held back
  because 12 instantiations gained spill, re-scoped to the structural axes where it
  is clean.
- **Anything that quietly reopens the input space is refused even when it is
  convenient.** The known instance: `TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES`
  escapes a compiler-cache/depfile collision, but it deletes header dependency edges,
  after which a stale object file passes every gate the project owns. Hard
  reject; a build-cache wrapper is admissible only with its own toolchain pin,
  which is R1 extended to the cache.

## 5. The input-closure rule

Every stamping, keying, or fingerprinting mechanism this design adds must enumerate,
explicitly, everything its key and its value depend on — not just what it *means* to
depend on. A mechanism that silently depends on more than it enumerates produces a
false verdict that reads as a true one, because the gate still runs and still prints
PASS or FAIL; it is answering a question nobody asked.

Two measured instances, both in this tree:

- **A symbol name is a key, and an anonymous namespace makes it depend on the checkout
  PATH.** A kernel in an anonymous namespace carries a mangled discriminator derived
  from the source directory — an input the key never named, because
  "the source" was assumed to mean the file's *content* — so a build at another path
  is certifiable-looking and wrong. The fix is at the cause, not a path convention:
  every kernel lives in a named namespace (the ast-grep rule
  `tools/lint/rules/global_fn_in_anon_namespace.yml`), so the key is a function of
  content and toolchain only, the two things a build's identity claims.
- **A record of numbers depended on the codegen flag list.** A register and spill
  record pinned the assembler version and the calibration macros but not the `nvcc`
  flag list itself, so a codegen-affecting flag change left no fingerprint in
  anything a gate read. The flag list is now one value in one file
  (`tools/build_flags.py`) that the graph writes into its compile, so no second
  compile can disagree with it.

Both instances have the same shape: a fingerprint whose *stated* input space (source
content, or source content plus assembler) was narrower than its *actual* input space
(source content plus checkout path; source content plus assembler plus flags). The
rule this teaches is checked at design time, not caught by a gate — the gate is what
reveals the gap after the fact, which is exactly the failure mode it is meant to
avoid repeating. Before landing a new stamp, key, or hash: write down everything it
is sensitive to, then verify nothing sensitive is missing from that list.

## 6. What this discipline is *not* claiming

It does not claim the kernel is optimal, or that the measured numbers are good.
It claims they are **the numbers**, that they were taken, and that they cannot
change without something failing. Every other document in this tree cites
measurements; this is the machinery that makes a citation mean something a decade
from now.

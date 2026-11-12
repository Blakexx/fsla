# `LEARNINGS.md` — what an agent got wrong, and what now stops it

A lesson an agent learned by writing something the standards already forbade is not kept by remembering it. Each row
is one such lesson: the incident, the rule it broke, and THE MECHANISM THAT ENFORCES IT -- a hook, a lint, a tool's
check -- named as `path::text`, a file of this tree and a string it holds. `tools/lint/learnings.py` gates the commit
on every row naming one that exists ([`lint.md`](tools/lint.md)): a row with no mechanism is a wish, and is refused.
A lesson whose mechanism is removed is removed with it or given another.

Adding a row: write the mechanism first, in the same commit. A mechanism that refuses what the tree already holds is
a ratchet -- its debts a list that only shrinks, as `tools/lab.py`'s `_PRE_TIMELINE` is.

| the lesson | the incident | the rule | enforced by |
|---|---|---|---|
| <a id="timeline"></a>An experiment's kernel is kernel code: its header states its TIMELINE -- each phase's stream, which load is a stage ahead of which use, the live registers -- before the kernel is written | `fold_stream`'s first producers (2026-11-11) were written with no timeline and read 551 and 318 cycles a token; with one, 26 to 84 | [`KERNEL_STANDARDS.md#s32`](../KERNEL_STANDARDS.md#s32) | `tools/lab.py::TIMELINE:` (a pack or a run of an experiment whose header has none is refused) |
| <a id="load-in-trip"></a>A global load is issued a stage ahead of its use; a ring's stages sit at fixed registers, never rotated by copies | the same producers loaded a token's value and stored it in one trip, each token waiting out its own load; the first fix rotated two batches by copies and still read the load in its trip | [`KERNEL_STANDARDS.md#s32`](../KERNEL_STANDARDS.md#s32), [`#s27`](../KERNEL_STANDARDS.md#s27) | `tools/lab.py::LOAD_IN_TRIP` (the SASS of every kernel's innermost loops; a count is declared or the experiment is refused) |
| <a id="store-reload"></a>A value no other warp reads stays in a register: a count is not kept in shared memory | the first producer read and wrote a fill count through shared memory at every leaf | [`KERNEL_STANDARDS.md#s33`](../KERNEL_STANDARDS.md#s33) | `tools/lab.py::STORE_RELOAD` (a shared load of an address its trip stored to) |
| <a id="loop-convergence"></a>Divergent regions inside a loop are removed or declared in EVERY kernel of an experiment, not only a budgeted one | the first producer held 36 divergent one-lane stores a batch; the check for them read budgeted kernels only, and a kernel marked exempt of its budget passed | [`KERNEL_STANDARDS.md#s29`](../KERNEL_STANDARDS.md#s29) | `tools/lab.py::held = experiment not in _PRE_TIMELINE` (the loop-convergence check on every kernel of an experiment past the debts) |
| <a id="enforced"></a>A lesson is recorded with its enforcement or not at all | the standards stated all four rules above, a hook printed one at the file's creation, and the kernel was written against them | [`KERNEL_STANDARDS.md#s26`](../KERNEL_STANDARDS.md#s26) | `tools/lint/learnings.py::names no mechanism` (this page's own gate) |

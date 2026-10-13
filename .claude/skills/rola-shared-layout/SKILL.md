---
name: rola-shared-layout
description: Load when designing or changing a shared-memory buffer, or data that picks shared addresses (slots, pools, panels): the access table first, conflict-free by construction, checked by the bank model before the GPU.
---
# Shared-memory layouts

The law: `docs/KERNEL_STANDARDS.md#s30`. The model: `docs/internals/tools/bank_model.md`.

1. Write the buffer's access table: every instruction that touches it, by lanes, addresses and banks; choose the
   pitch and swizzle that satisfy all of them.
2. Where data picks the addresses, the producer makes them conflict-free: the coloring and the round placement
   (`docs/internals/tools/lab.md#color-union`).
3. A pad sits where a real element would.
4. Run `tools/bank_model.py` on what the producer emits before any GPU run (a lab's `.inputs.py` calls it).
5. Budget the wavefronts beside the instructions (`docs/KERNEL_STANDARDS.md#s29`), priced across densities.

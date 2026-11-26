# `common/atoms/address.cuh` — the addresses

Mirrors `csrc/fsla/src/common/atoms/address.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): shared addresses and gathered and pinned bases. An atom's page in the state plane is `plane + atom * Page::kBytes`: the plane is dense, `[BH, N/16, page bytes]`, and no translation sits between an atom and its page.

- `smem_addr(p)`: the CTA-local address of a shared object, the address type every shared operation consumes.

- <a id="gather-ptr"></a>`gather_ptr(base, off)`: a gathered row's address, a 64-bit base formed once plus a 32-bit
  BYTE displacement ZERO-EXTENDED rather than multiplied out. The arithmetic is the contract: at element type the
  compiler re-derives `base + index` as a widening multiply per access (measured 30x `IMAD.WIDE` in a loop this form
  emits none for). The caller owns the 32-bit bound. The base is asserted GLOBAL, because a generic pointer turns a
  reduction into an atomic read-modify-write and every gathered load into a generic one. (no shipped caller)

- <a id="t"></a>`pin_address(p)`: a base the compiler must keep rather than re-derive. Under register pressure a
  compiler drops a live base and recomputes it in the loop -- the widening multiply moved rather than removed; the
  value becomes the result of an operation ptxas cannot see through, at the cost of the two registers it occupies.

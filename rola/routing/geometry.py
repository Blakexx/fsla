"""THE OWNER RECTANGLE (spec §3) -- the leaf space cut into `BC`-aligned blocks.

`N = prod(widths)` leaves in MSB-first mixed radix; a block of `BC` consecutive
leaves is a rectangle only when `BC` aligns to the radix, and when it does the
decomposition is UNIQUE. That is a fact about the ADDRESSING, not about any one
kernel: `rola/routing/summaries.py`'s pass takes a geometry because the summary
is stated per owner, and the addressing is not any one consumer's to own.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

__all__ = ["OwnerGeometry", "owner_geometry"]


@dataclass(frozen=True, slots=True)
class OwnerGeometry:
    """The unique ``BC``-aligned rectangle decomposition of the leaf space (§3).

    An aligned block is always exactly one rectangle::

        fixed digits (levels 0..j-1) x aligned sub-range of c digits (level j)
                                     x full span (levels j+1..D-1)

    ``first_spanned_level = j`` and ``slice_digits = c`` are *derived* from
    ``(widths, BC)``; they are not free parameters. Unaligned ranges would
    decompose into up to ``2D-1`` rectangles each with its own AND chain,
    row-scale and MMA, which is why §3 requires alignment.
    """

    widths: tuple[int, ...]
    BC: int
    first_spanned_level: int
    slice_digits: int

    @property
    def inner_extent(self) -> int:
        """``prod(widths[j+1:])`` -- the fully spanned levels below the boundary."""
        return prod(self.widths[self.first_spanned_level + 1:])

    @property
    def n_slices(self) -> int:
        """Number of aligned sub-ranges of the boundary level: ``b_j / c`` (§4)."""
        return self.widths[self.first_spanned_level] // self.slice_digits

    @property
    def fixed_levels(self) -> tuple[int, ...]:
        return tuple(range(self.first_spanned_level))

    @property
    def owners_total(self) -> int:
        return prod(self.widths) // self.BC

    def owner_digits(self, owner: int) -> tuple[tuple[int, ...], int]:
        """Decompose an owner id into ``(fixed digits, slice index)``.

        Inverse of the leaf-base identity ``owner * BC = sum_{l<j} d_l * suffix[l+1]
        + slice * c * suffix[j+1]``, which reduces exactly to
        ``owner = n_slices * mixed_radix(d_0..d_{j-1}) + slice``.
        """
        j = self.first_spanned_level
        slice_index = owner % self.n_slices
        rest = owner // self.n_slices
        digits = []
        for level in range(j):
            stride = prod(self.widths[level + 1:j])
            digits.append((rest // stride) % self.widths[level])
        return tuple(digits), slice_index

    def owner_of(self, fixed_digits: tuple[int, ...], slice_index: int) -> int:
        j = self.first_spanned_level
        rest = 0
        for level in range(j):
            rest += fixed_digits[level] * prod(self.widths[level + 1:j])
        return rest * self.n_slices + slice_index

    def leaf_base(self, owner: int) -> int:
        return owner * self.BC


def owner_geometry(widths: tuple[int, ...], BC: int) -> OwnerGeometry:
    """Derive the §3 owner rectangle for ``BC`` over an MSB-first mixed radix.

    Raises loudly (no fallback, no silent re-alignment) when ``BC`` does not
    produce an aligned rectangle. ``BC`` is a launch-geometry choice, so an
    inadmissible ``BC`` is a caller defect, not a runtime condition.
    """
    if BC < 1:
        raise ValueError(f"BC must be >= 1, got {BC}")
    N = prod(widths)
    if N % BC != 0:
        raise ValueError(f"BC={BC} must divide N={N} (owners partition the leaf space)")
    if BC > N:
        raise ValueError(f"BC={BC} exceeds the leaf capacity N={N}")
    D = len(widths)
    for j in range(D - 1, -1, -1):
        # The boundary level is the outermost-innermost level whose span already
        # reaches BC: prod(widths[j+1:]) < BC <= prod(widths[j:]). BC <= N makes
        # j = 0 the worst case, so the loop always returns.
        if prod(widths[j:]) < BC:
            continue
        inner = prod(widths[j + 1:])
        if BC % inner != 0:
            raise ValueError(
                f"BC={BC} is not a multiple of the spanned inner extent {inner} "
                f"(widths={widths}); it does not name a BC-aligned rectangle (spec §3)")
        c = BC // inner
        if c > widths[j] or widths[j] % c != 0:
            raise ValueError(
                f"BC={BC} implies {c} digits of level {j} (width {widths[j]}), which is not "
                f"an aligned sub-range (spec §3 requires c | width_j)")
        return OwnerGeometry(widths=tuple(widths), BC=BC, first_spanned_level=j, slice_digits=c)
    raise AssertionError(
        f"owner_geometry loop invariant violated (BC={BC}, widths={widths}); BC <= N was checked")

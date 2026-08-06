"""``FeederFlags`` -- the plan -> kernel per-level density word (``THUNDER_RUN_SPEC.md`` §4.4).

One word, 2 bits per level, ``D <= 16``. It is a CONTACT STRUCT: §4.4 freezes the
definition and neither track may edit it, so everything below implements the spec
sentence for sentence and adds nothing.

**TWO TRIGGERS, ONE MECHANISM.** A ``(side, level)`` cell is flagged dense either
because the CONFIGURATION cannot produce an exact structural zero there
(:class:`~rola.planner.schedule.StaticDensity`, read off the activation registry's
``exact_zeros`` column) or because the REALIZED routing saturated it
(``occupancy[side, l] >= C_SATURATION``, measured by the producer epilogue's summary
pass). §4.4 is explicit that the two set the SAME bits and that the kernel cannot
distinguish them and must not: the feeder's dense variant is correct whenever the
support really is full, and WHY it is full is not a fact the CTA has any use for.

**WHAT THE FLAG MEANS TO A CONSUMER, exactly.** ``dense_read[l]`` asserts that every
digit of level ``l`` is live on the read side at every token of the launch. It is a
claim about support, never about magnitude: a dense level may still carry a zero
AMPLITUDE (§12 rule 4's zero-route annihilation is a value, not a support fact), and
nothing here licenses skipping the amplitude read.

**WHAT A WRONG FLAG COSTS.** A flag set when the support is NOT full is a
correctness defect, not a mis-pricing: the dense feeder variant collapses the
intersection and would visit leafs whose support word says they are dead (§5.4). The
conservative default is therefore ALL-ZERO -- no level flagged -- which is today's
behaviour exactly (the sparse feeder is the general path), and it is what
:data:`FeederFlags.NONE` is for. The saturation trigger's placeholder
``C_SATURATION = 1.0`` (§8) keeps the detected arm at exact-dense for the same
reason: an occupancy fraction below 1.0 means some ``(token, digit)`` cell is a
structural zero, and one zero breaks the claim.
"""
from __future__ import annotations

from dataclasses import dataclass

#: §4.4: ``l < D <= 16``, two bits per level, one 32-bit word.
MAX_FEEDER_LEVELS = 16

#: Bits per level in the packed word (§4.4: ``dense_read[l]`` then ``dense_write[l]``).
BITS_PER_LEVEL = 2


@dataclass(frozen=True, slots=True)
class FeederFlags:
    """Per-``(side, level)`` density declarations, as §4.4's single word.

    ``read``/``write`` are one boolean per level, level-major. :meth:`packed` is the
    ``uint32`` the plan hands ``ConsumerParams``; bit ``BITS_PER_LEVEL * l + 0`` is
    ``dense_read[l]`` and bit ``BITS_PER_LEVEL * l + 1`` is ``dense_write[l]``, which
    is the layout the kernel decodes with shifts.
    """

    read: tuple[bool, ...]
    write: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.read) != len(self.write):
            raise ValueError(
                f"FeederFlags needs one flag per level per side; got {len(self.read)} read "
                f"and {len(self.write)} write")
        if len(self.read) > MAX_FEEDER_LEVELS:
            raise ValueError(
                f"FeederFlags carries {BITS_PER_LEVEL} bits per level in ONE 32-bit word "
                f"(spec §4.4), so D <= {MAX_FEEDER_LEVELS}; got D={len(self.read)}. A deeper "
                "topology is a spec change that must widen the word, not a runtime condition.")

    @property
    def D(self) -> int:
        return len(self.read)

    @classmethod
    def none(cls, D: int) -> FeederFlags:
        """The CONSERVATIVE default: nothing flagged, i.e. today's sparse feeder."""
        return cls(read=(False,) * D, write=(False,) * D)

    @classmethod
    def derive(cls, density, occupancy=None, *, saturation: float) -> FeederFlags:
        """§4.4's two triggers, ORed into one set of bits.

        ``density`` is a :class:`~rola.planner.schedule.StaticDensity` (the
        configuration trigger). ``occupancy`` is the ``[2, D]`` per-``(side, level)``
        mean support fraction from :class:`~rola.routing.summaries.RoutingSummaries`
        (the detection trigger), or ``None`` when no summary pass has run -- in which
        case only the configuration trigger fires, which is strictly the smaller flag
        set and therefore safe.

        ``saturation`` is ``C_SATURATION`` (§8), passed rather than imported so the
        placeholder has exactly one owner (:mod:`rola.planner.select`) and this module
        cannot silently disagree with it.
        """
        D = len(density.read)
        if occupancy is None:
            detected = ((False,) * D, (False,) * D)
        else:
            # ONE TRANSFER, not one per (side, level). `occupancy` is a device tensor, so
            # reading it element by element costs a host sync per element -- four at
            # D=2 -- and what a sync costs is the pipeline drain, not the bytes. Compare
            # on the device and bring back the whole boolean array at once. (Paired with
            # `mean_quantum`'s histogram read: each alone measures NULL because the other
            # still drains the pipe; together they are 2.88% of the step at L=4096.
            # Evidence: `workflows/refactor/p12_lane3.md`.)
            if hasattr(occupancy, "ge"):
                flags = (occupancy[:, :D] >= saturation).tolist()
            else:
                # A recorded row (a plain nested sequence) is already on the host, so
                # there is nothing to transfer and the comparison is the same one.
                flags = [[value >= saturation for value in occupancy[side][:D]]
                         for side in (0, 1)]
            detected = (tuple(bool(flag) for flag in flags[0]),
                        tuple(bool(flag) for flag in flags[1]))
        return cls(
            read=tuple(bool(a) or bool(b) for a, b in zip(density.read, detected[0])),
            write=tuple(bool(a) or bool(b) for a, b in zip(density.write, detected[1])))

    def side_dense(self, duty: str) -> bool:
        """Is the WHOLE side full-support?

        A leaf is live at a token iff EVERY level's digit is live there, so one dense
        level among sparse ones does NOT make the side's support the full rectangle.
        This is the same conjunction :meth:`StaticDensity.side_dense` takes, restated
        here over the OR of both triggers so a caller can ask the question of the
        DETECTED arm and not only the declared one.
        """
        if duty == "read":
            return all(self.read) and bool(self.read)
        if duty == "write":
            return all(self.write) and bool(self.write)
        raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")

    def packed(self) -> int:
        """§4.4's ``uint32 feeder_flags``: bit ``2l`` = read, bit ``2l+1`` = write."""
        word = 0
        for level in range(self.D):
            if self.read[level]:
                word |= 1 << (BITS_PER_LEVEL * level)
            if self.write[level]:
                word |= 1 << (BITS_PER_LEVEL * level + 1)
        return word

    @classmethod
    def unpack(cls, word: int, D: int) -> FeederFlags:
        """Inverse of :meth:`packed`, for tests and for reading a recorded plan back."""
        return cls(
            read=tuple(bool((word >> (BITS_PER_LEVEL * level)) & 1) for level in range(D)),
            write=tuple(bool((word >> (BITS_PER_LEVEL * level + 1)) & 1) for level in range(D)))


__all__ = [
    "BITS_PER_LEVEL",
    "MAX_FEEDER_LEVELS",
    "FeederFlags",
]

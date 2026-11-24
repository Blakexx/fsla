"""THE NUMERIC BAND, stated ONCE for every tier that grades against the oracle.

The comparison basis is fixed by the arms, not by a file: the fp64 oracle
(`fsla.ops.naive.naive_fsla`) is the reference, the kernel arm stores bf16, and
the derived budget is therefore the **bf16 family bound**

    BF16_RTOL = 1e-2      (u_bf16 = 2^-8 = 3.9e-3 per requantization)

Every suite that compares a kernel launch to the oracle imports this number.
That is the point of the module: a tolerance restated per file is a tolerance
that can be widened in one file by somebody who did not have to say so in a
diff a reviewer of the numeric contract would read.

THE RULE IS PER SLOT (`tests.oracle.fixtures.assert_slots_close`), and every
kernel-vs-oracle gate names the OUTPUT KIND it grades (`Output` below), which
carries the whole rule for that output:

- ITS CLAUSES, the shared form (Blake, 2026-09-15). A clause is an `(rtol, atol)`
  pair and FAILS a slot whose relative error exceeds `rtol` AND whose absolute
  error exceeds `atol`, so a slot of size `s` passes within
  `min_i max(rtol_i * s, atol_i)`: one clause is `allclose`'s own rule, a clause
  at `rtol = 0` is a pure absolute cap, and each further clause is another hinge
  in that curve. Every gate in the tree is held this way -- the decode
  kernels, the entmax solve and its gradients, the producer's
  stored levels and routes -- so one rule reads every oracle comparison and no
  gate states tolerances of its own.
- ITS ENVELOPE, where the output is MULTILINEAR: at most `rtol` times the sum of
  the sizes of the terms the slot adds up, which the same fp64 reference computes
  when run on `|v|` (`fixtures.oracle_run`; every routing weight, gain and decay
  factor is non-negative). A readout `num / (den + eps)` adds `|y|` for its
  denominator's rounding; a denominator or a mass column is its own envelope.
  No curve over slot size can separate a slot that is small because its terms
  cancelled from one that is small because its terms were small; the envelope
  gives each slot its own allowance from its own terms, which is why it sits
  beside the clauses rather than replacing them.

The tightest of the two binds, and a failure names which one it was.

A GRADIENT is no multilinear output: its coefficients are a nonlinear function of the stored values (roots, a division by their
sum), so it takes no envelope. Its reference is the operator's mathematical gradient evaluated in fp64 at the SAME stored values the
backward kernel reads (`tests/oracle/stored_vjp.py`), and its kind (`ENTMAX_GRADIENTS_FROM_STORED`) states the family budget on the
result with the absolute floor the output already carries. Its margins are to be recorded by the device run.

MEASURED 2026-09-14 (`pytest --oracle-margins=<file>` records every comparison):
the worst error over its allowance under its budget, and the smallest uniform
relative error the rule sees on some slot of the output.

    decode y (a step)    BF16_RTOL            0.11                    41% at most
    decode state         BF16_RTOL            < 1e-3                  1%
    projection logits    2^-8 (derived)       0.42                    1.2%

One decode step's `y` reads a random-signed entry state over up to
65536 leaves, so its terms cancel to about a fortieth of their sizes and that
step alone sees only a uniform error above 41%; the same gate's state sees 1%.
The TEETH are measured on the kernels' own outputs: the decode and projection
gates each wipe the slot of median size and move the smallest slots past their
allowance (`fixtures.assert_planted_errors_fail`).
"""
from __future__ import annotations

from dataclasses import dataclass

#: The derived bf16 family budget. See the module docstring for the derivation.
BF16_RTOL = 1e-2

#: One bf16 rounding's largest relative error: a round to nearest moves an 8-significant-bit value by at most 2^-8
#: of itself, and keeping a float's HIGH HALF (a truncation) by at most two of them.
U_BF16 = 2.0 ** -8

#: THE PRODUCTION ENTMAX SOLVE'S RATIFIED TOLERANCES against the fp64 reference solve (values and their gradients),
#: and the stored bf16 levels' (`tests/integration/test_production_levels.py` states their derivation beside the seam).
#: Never loosened.
ENTMAX_RTOL, ENTMAX_ATOL = 2e-5, 2e-6
ENTMAX_GRAD_RTOL, ENTMAX_GRAD_ATOL = 5e-5, 5e-6
STORED_ROUTE_RTOL, STORED_ROUTE_ATOL = 4e-3, 2e-3


@dataclass(frozen=True)
class Output:
    """ONE KIND OF ORACLE-GRADED OUTPUT: the clauses every slot of it is held to, and on a multilinear output the
    envelope budget. A clause ``(rtol, atol)`` FAILS a slot whose relative error exceeds ``rtol`` AND whose absolute
    error exceeds ``atol`` (Blake, 2026-09-15: ``OR(AND(r1, a1), AND(r2, a2), ...)``), so a slot of size ``s`` passes
    within ``min_i max(rtol_i·s, atol_i)``: one clause is `allclose`'s rule, each further clause another hinge in that
    curve, and a clause at ``rtol = 0`` is a pure absolute cap. ``rtol`` is the multilinear envelope budget
    (`fixtures.allowances`), None for an output that has no envelope. Every gate names its output kind here; no gate
    states its own numbers.
    """

    name: str
    clauses: tuple[tuple[float, float], ...]
    rtol: float | None = None


@dataclass(frozen=True)
class Owed:
    """AN OUTPUT KIND WHOSE KERNEL IS NOT BUILT: its clauses are owed by the kernel's own measurement (`pytest
    --oracle-margins=FILE`, read with `fixtures.frontier_rtols`), taken when a hopper schedule of it lands, and none is
    stated before. Grading against it refuses by name (`fsla.ops._ext.KernelNotBuilt`); a launch of the kernel refuses
    first, so a test of it collects and fails by that one named refusal until the kernel exists.
    """

    name: str
    kernel: str

    def _refuse(self):
        from fsla.ops._ext import KernelNotBuilt

        raise KernelNotBuilt(f"the {self.name} tolerance is owed by the {self.kernel} kernel's own measurement: that "
                             "kernel is not built on this line, so no clause is stated")

    @property
    def clauses(self):
        self._refuse()

    @property
    def rtol(self):
        self._refuse()


#: THE OUTPUT KINDS OF THE KERNELS THIS LINE HAS NOT BUILT, each OWED by its kernel's measurement on the hardware it
#: lands on (the ampere line's clauses were measured on another machine against another kernel and are the branch's).
CARRY_NUM = Owed("carry num", kernel="carry")
CARRY_DEN = Owed("carry den", kernel="carry")
PREFILL_READOUT = Owed("prefill readout", kernel="prefill")
PREFILL_STATE = Owed("prefill state", kernel="prefill")

#: THE CLAUSES ARE MEASURED, and these are the measurements (`pytest --oracle-margins=FILE`, read with
#: `fixtures.frontier_rtols`): for each relative error, the largest ABSOLUTE error any slot past it took on the whole
#: battery, from the comparisons that passed. A clause declares that relative error with twice that absolute error, so
#: no measured slot is inside a clause it fails; clauses below the output's own declared budget are not declared at all,
#: because a draw that reached the budget would fail one. A kind's numbers are re-measured when the cells or the
#: hardware change, and tightening is always the safe direction. MEASURED 2026-09-15 on RTX 3080 Ti (sm_86), the oracle,
#: integration and entmax tiers: 2,700 comparisons.
DECODE_Y = Output("decode y", rtol=BF16_RTOL, clauses=((0.0, 3.52e-3), (1.0, 1.25e-3), (1e2, 3.35e-4)))
#: The int8 state's error grows as one rounding per write (sqrt of the writes per leaf; decay adds roundings): re-measured
#: 2026-10-10 over the decode battery and g6 at widths (64, 64), seeds 64-79, with and without decay: 2 x 1.013e-6.
DECODE_STATE = Output("decode state", rtol=BF16_RTOL, clauses=((0.0, 2.027e-6), (1e-2, 7.80e-9), (1e-1, 0.0)))
PROJECTION_LOGITS = Output("projection logits", rtol=U_BF16, clauses=((0.0, 1.32e-1), (1.0, 6.42e-2), (1e1, 0.0)))
ENTMAX_VALUES = Output("entmax values", clauses=((0.0, 8.34e-2), (ENTMAX_RTOL, ENTMAX_ATOL)))
ENTMAX_GRADIENTS = Output("entmax gradients", clauses=((0.0, 5.96e-5), (ENTMAX_GRAD_RTOL, ENTMAX_GRAD_ATOL)))
STORED_LEVELS = Output("stored bf16 levels", clauses=((0.0, 1.52e-1), (BF16_RTOL, BF16_RTOL)))
#: THE SOLVE'S OWN bf16 STORED FORM against the reference's value rounded to bf16 (the CUDA solve writes bf16 and nothing else): one bf16 unit in
#: the last place, `2^-7` of the value (a round to nearest moves a value by at most `2^-8` of itself, and the reference's own rounding may land on
#: the neighbouring side of the cut), over the absolute floor the fp32 form was held to (`ENTMAX_ATOL`). Derived, not measured.
ENTMAX_STORED_VALUES = Output("entmax stored values", clauses=((2.0 ** -7, ENTMAX_ATOL),))
#: THE ENTMAX, SOFTMAX AND UNION BACKWARDS' GRADIENTS INTO THE LOGITS, against the reference's gradient evaluated in fp64 at the SAME stored
#: bf16 values the backward reads (`tests/oracle/stored_vjp.py`; the union's is derived from the reference forward, not published), so both sides evaluate one nonlinear function at one point. What is
#: left is the kernel's fp32 arithmetic and the rounding of its bf16 result (`U_BF16` of the slot): the family budget `BF16_RTOL`
#: on the result over the absolute floor this output already carries (`ENTMAX_GRAD_ATOL`, the fp32 form's), and no envelope: the
#: coefficients of a gradient are a nonlinear function of the stored values, not a multilinear sum of terms. Derived, not measured;
#: the device run's margins (`--oracle-margins`) are to be recorded. The error against the oracle's gradient at the unrounded values is reported.
ENTMAX_GRADIENTS_FROM_STORED = Output("entmax gradients from stored values", clauses=((BF16_RTOL, ENTMAX_GRAD_ATOL),))
STORED_ROUTE = Output("stored route", clauses=((0.0, 2 * STORED_ROUTE_ATOL), (STORED_ROUTE_RTOL, STORED_ROUTE_ATOL)))

#: EVERY OUTPUT KIND, for the test that holds each to declaring its clauses and their measurement.
OUTPUTS = (DECODE_Y, DECODE_STATE, PROJECTION_LOGITS, ENTMAX_VALUES, ENTMAX_GRADIENTS, ENTMAX_STORED_VALUES,
           ENTMAX_GRADIENTS_FROM_STORED,
           STORED_LEVELS, STORED_ROUTE)

#: Where the per-slot check records its margins: `--oracle-margins`, set by `tests/conftest.py`; None records nothing.
MARGINS_FILE = None

"""THE AUDITOR (``THUNDER_RUN_SPEC.md`` §6 P4): both deciders, one ledger.

Under the end-state ruling the runtime no longer runs a cost model, so nothing at
runtime can tell you whether selection was right. What replaces it is this: an OFFLINE
job that runs BOTH deciders on recorded cells and either asserts agreement or writes
the disagreement down with its modelled regret.

    closed form   `rola.planner.select.select_config`   -- what ships
    exact argmin  `planner_calibration.cost_model`      -- what is retired

**THE REGRET IS DENOMINATED IN THE RETIRED CURRENCY, AND THAT IS THE POINT.** "The
closed form picked worse" is only a statement in a currency that prices both arms, and
the only such currency is the cost model being retired. That is precisely why the
shadow belongs here and not in the runtime: a runtime shadow would keep alive exactly
the object the ruling deletes.

**WHAT THE AUDITOR CAN AND CANNOT FALSIFY** (§2's approximation contract). It compares
two MODELS. It cannot tell you the model is right about the machine -- that is the
eight measured argmins' job, and they are recorded rows, not model outputs. What it
CAN tell you is that a constant moved without a re-fit, that a rule stopped tracking
the model it was derived from, or that a placeholder is doing more than it was supposed
to. Every disagreement carries its MARGIN as well as its regret, because §2's validity
domain is "decisions with margins large vs residuals": a disagreement at a 30 % margin
is a different finding from one at 1 %.
"""
from __future__ import annotations

from dataclasses import dataclass

from rola.planner.calibration.cost_model import (
    TileUnionStats,
    plan_cost_per_token,
    plan_waves,
)
from rola.planner.schedule import TOKEN_WORD
from rola.planner.select import (
    ConfigChoice,
    HardwareFacts,
    StaticPlanInputs,
    select_config_with_evidence,
)
from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology


def _topology(widths: tuple[int, ...], d_v: int) -> Topology:
    """A Topology carrying only what the cost model reads off one: widths and ``d_v``.

    ``plan_cost_per_token`` uses the topology as a PRECONDITION (the owner rectangle
    must be admissible at this ``BC``) and not as a price -- the ``iso-gc-*`` cells
    measured the rectangle's shape at EXACTLY zero cost -- so the routing attached to
    each level is immaterial here and a single activation is used rather than the cell's
    own. Stated because a reader may reasonably wonder whether it should be threaded
    through.
    """
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                write_activation=SoftmaxActivation())
    return Topology(levels=tuple(LevelConfig(width=w, routing=routing) for w in widths),
                    d_v=d_v)


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One cell's verdict. ``regret`` is ``None`` exactly when the two deciders agree."""

    cell: str
    decay: bool
    closed: ConfigChoice
    exact_BT: int
    margin: float
    regret: float | None

    @property
    def agreed(self) -> bool:
        # ``ConfigChoice`` has no ``BT`` field (P26 killed the runtime BT decision
        # entirely -- the executor always runs at the fixed 32-token ``TOKEN_WORD``,
        # not a per-launch choice). "The closed form's BT" is therefore the constant
        # every launch actually uses, not a field to read off `closed`; the audit
        # question this property answers is unchanged -- does the calibration-time
        # exact argmin still prefer the value the runtime is structurally pinned to.
        return self.exact_BT == TOKEN_WORD

    def as_ledger_entry(self) -> tuple[str, int, int, float]:
        """The row shape the gate's ``_SELECTOR_DISAGREEMENTS`` table asserts.

        The regret is rounded to 4 decimals so the LEDGER is a stable artefact: it is a
        record of WHICH decisions differ and by roughly how much, and asserting it to
        machine precision would make the gate fail on a float reassociation rather than
        on a decision moving. Four decimals is far finer than any regret that matters
        (the smallest live row is 0.17) and far coarser than any reassociation.
        """
        return (f"{self.cell}/decay={self.decay}",
                TOKEN_WORD,
                self.exact_BT,
                round(self.regret, 4) if self.regret is not None else 0.0)


def exact_argmin(widths, d_v, BC, decay, stats_by_bt, blocks_by_bt, sm_count):
    """The RETIRED decider, run verbatim: argmin over ``BT`` of cost x waves.

    ``stats_by_bt`` maps ``BT`` to a :class:`TileUnionStats` -- recorded, or measured by
    the exact census on a live routing. ``blocks_by_bt`` maps ``BT`` to the driver's
    CTAs/SM. Nothing here is re-derived from the closed form: this is the function the
    closed form is being audited against, so sharing an expression with it would make
    the agreement gate prove only self-consistency.
    """
    topology = _topology(widths, d_v)
    keys: dict[int, float] = {}
    for BT, stats in stats_by_bt.items():
        cost = plan_cost_per_token(topology, BC, stats, decay=decay)
        keys[BT] = cost * plan_waves(stats.n_cohorts, blocks_by_bt[BT], sm_count)
    best = min(keys, key=keys.get)
    return best, keys


def audit_cell(cell: str, static: StaticPlanInputs, hw: HardwareFacts,
               summaries_by_bt, stats_by_bt, blocks_by_bt) -> AuditRow:
    """Run both deciders on one cell and return the verdict.

    The closed form is given the SAME per-``BT`` statistics the exact argmin sees, so a
    disagreement is attributable to the RULE and never to an input mismatch. That is a
    deliberate choice about what this gate is for: the single-granularity question §4.2
    raises is a separate finding with its own gate, not something to fold in here where
    it would masquerade as a rule defect.
    """
    choice, evidence = select_config_with_evidence(
        static, hw, summaries_by_bt[min(summaries_by_bt)],
        summaries_by_bt=summaries_by_bt)
    exact_BT, keys = exact_argmin(
        static.widths, static.d_v, choice.BC, static.decay, stats_by_bt,
        blocks_by_bt, hw.sm_count)
    regret = None
    if exact_BT != TOKEN_WORD:
        # `choice` (a `ConfigChoice`) carries no `BT` -- the runtime is pinned to
        # `TOKEN_WORD` unconditionally now; see `AuditRow.agreed`.
        regret = (keys[TOKEN_WORD] - keys[exact_BT]) / keys[exact_BT]
    return AuditRow(cell=cell, decay=static.decay, closed=choice, exact_BT=exact_BT,
                    margin=evidence.margin, regret=regret)


def ledger(rows) -> tuple[tuple[str, int, int, float], ...]:
    """The DISAGREEMENT ledger: every row where the two deciders differ, sorted.

    An EXACT set is what the gate asserts, in both directions. A new disagreement fails,
    and so does a silently-fixed one -- because a disagreement that vanishes without a
    re-fit means a constant moved underneath the rule that was derived from it.
    """
    return tuple(sorted(row.as_ledger_entry() for row in rows if not row.agreed))


def recorded_stats(BC: int, BT: int, candidate_frac: float, ksteps: float,
                   kisteps: float, n_cohorts: int) -> TileUnionStats:
    """A :class:`TileUnionStats` from a RECORDED row rather than a live census.

    The fields the cost model reads are exactly ``candidate_frac``,
    ``mean_ksteps_candidate``, ``mean_kisteps_candidate``, ``n_cohorts``, ``BC`` and
    ``BT``; the rest are filled consistently so the object is not half-built, and are
    never read by anything this module calls.
    """
    return TileUnionStats(
        BC=BC, BT=BT, mean_K=candidate_frac * BC,
        saturated_frac=1.0 if candidate_frac == 1.0 else 0.0,
        mean_K_candidate=ksteps * 16, mean_KI_candidate=kisteps * 16,
        mean_ksteps_candidate=ksteps, mean_kisteps_candidate=kisteps,
        candidate_frac=candidate_frac, n_pairs=0, n_cohorts=n_cohorts)


__all__ = ["AuditRow", "audit_cell", "exact_argmin", "ledger", "recorded_stats"]

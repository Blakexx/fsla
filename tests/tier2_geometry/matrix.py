"""TIER 2 — the geometry matrix, as DATA.

Design §6.3. Tier 2 asks one question: *does the kernel compute the same function
at every runtime geometry the planner may choose?* The answer is not a pile of
hand-written comparisons — it is this table, plus three meta-tests that keep the
table honest (``test_meta.py``):

* **M1 CENSUS** — every geometry axis the shipped binary actually carries is
  DECLARED here, and every declared geometry axis has rows. A new instantiation
  axis, or a new value on an existing one, **fails at collection**.
* **M2 NON-VACUITY** — every ``ENGINEERED_IDENTITY`` row is executed on a fixture
  where a broken kernel *would* differ.
* **M3 CLASS DRIFT** — every ``REASSOCIATION_EQUIVALENT`` row is asserted **NOT**
  byte-equal. A reassociation that stops happening is a silent kernel change.

**This module is pure data and pure CPU.** It imports no torch, so M1 runs in the
CPU CI job with no device and no build. Execution lives in ``test_matrix_rows.py``.

WHY A CLASS IS DECLARED AND NEVER INFERRED
------------------------------------------
A pair is class (a) or class (b); there is no "mostly identical". Inferring the
class from observed behaviour is how an engineered identity breaks while hiding
inside a sloppy tolerance. A class change is therefore a reviewed edit to THIS
FILE, in a diff, with a new ``basis`` string — it can never be a side effect of a
kernel edit.

THREE DEVIATIONS FROM THE DESIGN SKETCH, RECORDED
-------------------------------------------------
1. **Rows carry an ``output``.** §6.2's own ``P`` sub-case table splits ``state``
   (class (a) from a zero initial state — no atomic) from ``y`` (class (b) —
   split-K ``atomicAdd``) on the *same* geometry pair. The output is therefore
   part of the row's identity, not a detail of its test.
2. **Rows carry an optional ``restrict``.** §6.4's unwritten-leaf carve-out is
   normative and bitwise *on a subset of the state* while the written leaves next
   to it are legitimately re-associated. A selector name makes that a declared row
   rather than a bespoke test.
3. **The census reads the ratification manifest, not ``BUILD_CONFIG``.** §6.3
   names ``BUILD_CONFIG["instantiations"]``; the generated ``_build_config.py``
   carries no such key — it carries ``manifest_sha256``, the hash of the
   ratification the shipped ``.so`` was gated against. The manifest's 1024 mangled
   names ARE the shipped instantiation set, and M1 pins the file it reads to that
   hash, so the census cannot drift onto a manifest the binary was not built from.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = _REPO / "tools" / "manifests"


# ---------------------------------------------------------------------------
# Contract classes
# ---------------------------------------------------------------------------

class Contract(Enum):
    #: The two geometries are DESIGNED to produce byte-identical output. Tested
    #: with `torch.equal` — no tolerance, no `allclose`, ever. The claim is always
    #: structural and the structure is stated in `basis`.
    ENGINEERED_IDENTITY = "engineered-identity"
    #: The two geometries legitimately reorder floating-point accumulation. Tested
    #: by TWO assertions, both required: an independent oracle anchor per geometry,
    #: and a mutual bound DERIVED from the reassociation (never a chosen number).
    REASSOCIATION_EQUIVALENT = "reassociation-equivalent"


ENGINEERED_IDENTITY = Contract.ENGINEERED_IDENTITY
REASSOCIATION_EQUIVALENT = Contract.REASSOCIATION_EQUIVALENT


class AxisKind(Enum):
    #: Same function, different execution. Tier 2's subject: the planner may choose
    #: either value for the same user-visible problem, so the two must agree.
    GEOMETRY = "geometry"
    #: A different problem, not a different execution of one. `d_v = 32` and
    #: `d_v = 64` do not even produce comparable tensors, so there is no pair.
    PROBLEM_SHAPE = "problem-shape"
    #: A different function, or the same function under a different declared
    #: numeric contract. Anchored in TIER 1 against the fp64 oracle, one tolerance
    #: per arm — never compared geometry-to-geometry here.
    SEMANTIC = "semantic"


class Census(Enum):
    #: Every ADJACENT pair of declared values must have at least one row.
    ADJACENT = "adjacent"
    #: Every value other than the declared reference must be paired WITH the
    #: reference. Used where one value is the definitional baseline and a
    #: value-to-value comparison would not be meaningful on its own.
    VS_REFERENCE = "vs-reference"


# ---------------------------------------------------------------------------
# Axis declarations — the census's left-hand side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisDeclaration:
    name: str
    kind: AxisKind
    #: The values the shipped product carries. For `source="template"` this is
    #: CHECKED against the mangled instantiation names; for `source="runtime"` it
    #: is a declaration, because a host-side split leaves no symbol behind.
    values: tuple[Any, ...]
    source: str          # "template" | "runtime"
    basis: str           # WHY this kind. "" is illegal.
    census: Census | None = None
    reference: Any = None

    def __post_init__(self):
        assert self.basis, f"axis {self.name!r} declares no basis"
        assert self.source in ("template", "runtime")
        if self.kind is AxisKind.GEOMETRY:
            assert self.census is not None, f"geometry axis {self.name!r} declares no census rule"
            if self.census is Census.VS_REFERENCE:
                assert self.reference in self.values, (
                    f"axis {self.name!r} references {self.reference!r}, which is not a value")
        else:
            assert self.census is None, f"non-geometry axis {self.name!r} cannot have rows"

    def required_pairs(self) -> tuple[tuple[Any, Any], ...]:
        """The pairs the census demands rows for."""
        if self.census is Census.ADJACENT:
            v = self.values
            return tuple((v[i], v[i + 1]) for i in range(len(v) - 1))
        if self.census is Census.VS_REFERENCE:
            return tuple((self.reference, v) for v in self.values if v != self.reference)
        return ()


#: The template parameter list of `consumer_forward_kernel`, IN ORDER
#: (`csrc/rola/src/consumer_kernel.cuh`):
#:
#:     template <int BC, int DV, int D, bool DECAY>
#:
#: M1 decodes exactly these from every mangled name in the ratification manifest
#: and requires the decoded axis NAMES and VALUE SETS to match the declarations
#: below in both directions. Adding a template parameter, or a value on one,
#: without declaring what kind of axis it is fails at collection.
TEMPLATE_PARAMETER_ORDER: tuple[str, ...] = ("BC", "DV", "D", "DECAY")

AXIS_DECLARATIONS: tuple[AxisDeclaration, ...] = (
    AxisDeclaration(
        "BC", AxisKind.GEOMETRY, (16, 64), "template", census=Census.ADJACENT,
        basis="the owner tile. `prod(widths)/BC` owners cover the SAME state "
              "columns however they are cut, so this is a tiling of one problem "
              "and not two problems — see the `((8,8),16)` and `((8,8),64)` rows "
              "of the tier-1 rectangle grid, which are the same topology."),
    AxisDeclaration(
        "DV", AxisKind.PROBLEM_SHAPE, (32, 64), "template",
        basis="the value width. `d_v = 32` and `d_v = 64` produce differently "
              "shaped `y` and state from different inputs; there is no pair to "
              "compare, only two problems each anchored in tier 1."),
    AxisDeclaration(
        "D", AxisKind.PROBLEM_SHAPE, (1, 2, 3, 4), "template",
        basis="the number of routed levels. A different `D` is a different "
              "topology — a different function of different inputs, gated across "
              "the tier-1 rectangle grid rather than compared to another `D`."),
    AxisDeclaration(
        "DECAY", AxisKind.SEMANTIC, (0, 1), "template",
        basis="LeafMassDecay on/off is a different recurrence, not a different "
              "execution of one. Both arms are tier-1 oracle-gated, including the "
              "`1 - rate` catastrophic-cancellation arm."),
    #: RUNTIME axes. These leave NO symbol in the binary — paging is an
    #: allocation of the state — so the census cannot discover them, and an
    #: undeclared one would be invisible. Listed here precisely so that the
    #: completeness claim covers them.
    AxisDeclaration(
        "paging", AxisKind.GEOMETRY, (False, True), "runtime", census=Census.ADJACENT,
        basis="§7.1's page arena decides WHERE an owner's `BC` state rows live, "
              "never what they are. A residency decision that moved a number "
              "would be a defect, so the pair is in the matrix."),
)

AXIS_BY_NAME: Mapping[str, AxisDeclaration] = {a.name: a for a in AXIS_DECLARATIONS}


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryPair:
    axis: str
    a: Mapping[str, Any]      # geometry A, as overrides on `fixture`
    b: Mapping[str, Any]      # geometry B
    output: str               # "y" | "state"
    contract: Contract
    fixture: str              # a key of test_matrix_rows.FIXTURES
    precondition: str         # asserted in-test; "" is illegal
    basis: str                # WHY. spec cite or derivation. "" is illegal
    bound: str | None = None  # REASSOCIATION_EQUIVALENT only: the derivation's name
    restrict: str | None = None   # a key of test_matrix_rows.SELECTORS
    #: M2's probe: the INPUT a broken kernel would propagate a perturbation from,
    #: for THIS row's output. Declared rather than assumed, because it is not always
    #: `v`: §6.4's carve-out row asserts that an UNWRITTEN leaf's state equals the
    #: INITIAL state, and an unwritten leaf is by definition insensitive to `v` --
    #: probing `v` there would report the row vacuous when it is not.
    nonvacuity_probe: str = "v"

    def __post_init__(self):
        assert self.precondition, f"{self.id} declares no precondition"
        assert self.nonvacuity_probe in ("v", "initial_state"), self.nonvacuity_probe
        assert self.basis, f"{self.id} declares no basis"
        assert self.output in ("y", "state"), self.output
        assert set(self.a) == set(self.b) == {self.axis}, (
            f"{self.id}: a row varies exactly its own axis; anything else is two "
            f"changes in one comparison")
        if self.contract is REASSOCIATION_EQUIVALENT:
            assert self.bound, f"{self.id} is class (b) and names no derived bound"
        else:
            assert self.bound is None, f"{self.id} is class (a); a bound is meaningless"

    @property
    def values(self) -> tuple[Any, Any]:
        return (self.a[self.axis], self.b[self.axis])

    @property
    def id(self) -> str:
        va, vb = self.values
        tag = f"{self.axis}:{va}->{vb}:{self.output}"
        return f"{tag}:{self.restrict}" if self.restrict else tag


GEOMETRY_MATRIX: tuple[GeometryPair, ...] = (
    # -- BC -----------------------------------------------------------------
    GeometryPair(
        "BC", {"BC": 16}, {"BC": 64}, "state", ENGINEERED_IDENTITY,
        fixture="multi_owner_dense",
        precondition="the two BC values tile the SAME `prod(widths)` state columns, and "
                     "no initial state is carried",
        basis="owner tiles PARTITION the state columns — they never overlap — and the "
              "state has no atomic. A column's accumulation therefore walks the same "
              "tokens in the same order whichever owner tile owns it, so cutting the "
              "rectangle differently cannot move a bit. MEASURED byte-equal across "
              "both normalizations and both decay arms (W3 probe, 2026-08-01)."),
    GeometryPair(
        "BC", {"BC": 16}, {"BC": 64}, "y", REASSOCIATION_EQUIVALENT,
        fixture="multi_owner_dense",
        precondition="the two BC values give DIFFERENT owners_total (4 vs 1)",
        basis="`y` sums one element over `owners_total` partials by atomicAdd; at "
              "BC = 64 there is one owner and the sum is a single deposit, at BC = 16 "
              "there are four arriving in scheduler order.",
        bound="split_readout_n2_eps"),

    # -- paging -------------------------------------------------------------
    GeometryPair(
        "paging", {"paging": False}, {"paging": True}, "state", ENGINEERED_IDENTITY,
        fixture="paged_dense",
        precondition="the arena is materialized back to the dense layout before comparing",
        basis="§7.1: a page table changes the ADDRESS of an owner's `BC` state rows and "
              "nothing else. The state has no atomic, so the dense run is exactly "
              "reproducible and paging must not move a bit of it — every paging defect "
              "(a wrong page id, a missed commitment, a mis-ordered admission) lands here.",
    ),
    GeometryPair(
        "paging", {"paging": False}, {"paging": True}, "y", REASSOCIATION_EQUIVALENT,
        fixture="paged_dense",
        precondition="owners_total > 1, so the readout is a split-K atomicAdd",
        basis="paging does not reorder the readout by design, but the readout's atomics "
              "are unordered ANYWAY: `y` differs run-to-run at fixed geometry "
              "(`test_split_k_determinism_is_not_assumed`). Claiming bit-equality here "
              "would be claiming a property of the scheduler.",
        bound="split_readout_n2_eps"),
)


# ---------------------------------------------------------------------------
# The census's right-hand side: what the shipped binary actually carries
# ---------------------------------------------------------------------------

#: `_ZN4rola8consumer23consumer_forward_kernelILi16ELi32ELi1ELb0EE...`
_MANGLED = re.compile(
    r"consumer_forward_kernelI"
    r"Li(\d+)ELi(\d+)ELi(\d+)ELb(\d+)E")

#: The NON-consumer ratified families (coverage-closure item 12): the decode GEMV
#: and the union-entmax forward/backward pair. An EXPLICIT allowlist, so that a
#: manifest entry from any kernel nobody declared still fails the census.
_OTHER_RATIFIED_FAMILIES = re.compile(
    r"decode_gemv_kernelI|union_forward_kernelI|union_backward_kernelI")


def load_manifest(directory: Path = MANIFEST_DIR) -> dict:
    """The ratification, read as ONE record however many files it is split across.

    Per-arch files (§3.7) and the single un-split file are both handled, because
    the census's question -- *what did the assembler actually see?* -- does not
    depend on the filing. Every file must pin the SAME toolchain: a manifest whose
    two halves were ratified under different assemblers is not one ratification.
    """
    files = sorted(p for p in directory.glob("*.json"))
    assert files, f"no ratification manifest in {directory}"
    toolchain: dict | None = None
    shards: dict | None = None
    entries: dict[str, dict] = {}
    for path in files:
        blob = json.loads(path.read_text())
        if toolchain is None:
            toolchain, shards = blob["toolchain"], blob["shards"]
        assert blob["toolchain"] == toolchain, (
            f"{path.name} pins a different assembler than {files[0].name}; the "
            "per-arch split may not become a split TOOLCHAIN")
        assert blob["shards"] == shards, (
            f"{path.name} records a different shard membership than "
            f"{files[0].name}; the per-arch split may not become a split MATRIX")
        for key, value in blob["entries"].items():
            assert key not in entries, f"{key} is ratified twice"
            entries[key] = value
    return {"toolchain": toolchain, "shards": shards, "entries": entries}


def manifest_sha256(directory: Path = MANIFEST_DIR) -> str:
    """The digest ``setup.py`` records in ``_build_config.py``, recomputed.

    Canonical JSON of ``{toolchain, entries, shards}`` with sorted keys -- NOT the
    raw file bytes -- which is what makes the digest a property of the RATIFICATION
    rather than of its formatting or of how many files it is filed in. §3.7's
    per-arch split therefore does not change it, and that invariant is itself
    checked (``tools/ratify.py --self-test``). ``shards`` participates because it
    records WHICH instantiations were measured, which is part of what was ratified
    and not part of how it was filed.
    """
    m = load_manifest(directory)
    blob = json.dumps({"toolchain": m["toolchain"], "entries": m["entries"],
                       "shards": m["shards"]}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def shipped_axis_values(directory: Path = MANIFEST_DIR) -> dict[str, set[Any]]:
    """Decode the template arguments of every ratified instantiation.

    The mangled name is the census's source of truth because it is what the
    ASSEMBLER saw: it cannot disagree with the binary the way a hand-kept list can.
    """
    entries = load_manifest(directory)["entries"]
    found: dict[str, set[Any]] = {n: set() for n in TEMPLATE_PARAMETER_ORDER}
    for key in entries:
        mangled = key.split(":", 1)[1]
        m = _MANGLED.search(mangled)
        if m is None:
            #: The coverage-closure census extension (item 12) ratifies THREE kernel
            #: families, not one: the consumer matrix above, the decode GEMV and the
            #: two union-entmax kernels. This census polices the CONSUMER's template
            #: axes; the other ratified families are RECOGNIZED BY NAME and excluded
            #: from it — recognized, not skipped-on-no-match, so an unparseable
            #: consumer entry (or a fourth family nobody declared here) still fails
            #: exactly as before. Their own gates are the ratification manifest's
            #: six checks and the decode/entmax suites, not this matrix.
            if _OTHER_RATIFIED_FAMILIES.search(mangled):
                continue
            raise AssertionError(
                f"a ratified instantiation does not match the declared template "
                f"parameter list {TEMPLATE_PARAMETER_ORDER}: {mangled!r}. Either the "
                f"kernel gained or lost a template parameter — in which case its "
                f"census axis must be DECLARED in matrix.py — or the mangling changed.")
        for name, value in zip(TEMPLATE_PARAMETER_ORDER, m.groups()):
            found[name].add(int(value))
    return found

"""I6-A: the PHASE-C RENDEZVOUS is structurally safe, read off the SOURCE.

The phase-C counter (``ctl[kCtlGramDone]``) carries exactly one edge: every store
to ``Gm`` happens-before every load of ``Gm``, within a tile.  It is a COUNTER,
not a barrier -- warps ``retire_item`` into it and one ``await_count`` waits for a
target.  That makes it possible to get the edge wrong in a way no bit-identity
comparison can see, because a benign race usually reproduces.

**The invariant this file gates, stated once.**  Let ``S`` be the set of warps
that STORE ``Gm``, ``R`` the set that RETIRE into the counter, and ``T`` the
``await_count`` target.  The release is safe iff

1. ``R`` is a SUPERSET of ``S`` -- every storing warp announces its store; and
2. ``T >= |R|`` -- the waiter cannot reach its target on retirements alone,
   i.e. it cannot be satisfied while a member of ``S`` has not yet retired.
   (Without all of ``S`` retired at most ``|R| - 1`` retirements exist, so
   ``T >= |R|`` is exactly "unreachable until every writer has published".)

Both the shipped 8-wide form (``R`` = all warps, ``T = kWarps``) and the narrowed
form (``R = S = {mixed && warp < kGramWarps}``, ``T = kGramWarps``) satisfy it.
The DEFECT this gate exists to reject is the one
``LATENCY_COVERAGE_DESIGN.md:129-132`` names: lowering ``T`` to ``kGramWarps``
while leaving the retirement UNCONDITIONAL.  There ``|R| = kWarps = 8`` and
``T = 2``, so a waiter can be released by retirements from warps 5, 6, 7 -- which
store no ``Gm`` -- while warps 0-1 are still storing it.  Condition 2 catches it.

**Why the SOURCE and not the manifest.**  ``test_ring_depth_agreement.py`` reads
its macro off ``tools/manifests/sm_86.json``'s toolchain blob, which means a
measurement build that changed the header is invisible to it (``ring_depth.md``
recorded that weakness).  This gate does not repeat it: it parses
``consumer_kernel.cuh`` itself, so the thing checked is the thing compiled.

**The gate asserts its own non-vacuity.**  Every check below is re-run against
three SYNTHETIC mutant sources built by textual surgery on the real header, and
each mutant must be REJECTED.  A gate that cannot fail is not a gate.
"""

from __future__ import annotations

import pathlib
import re

KERNEL = (pathlib.Path(__file__).resolve().parents[2]
          / "csrc" / "rola" / "src" / "consumer_kernel.cuh")

#: The instantiation matrix's structural axes, for the numeric discharge of the
#: symbolic warp-set bounds.  ``kWarps = kConsumerThreads / 32`` and
#: ``NM = BT / 16`` are read from the header rather than assumed (below).
_BT_VALUES = (16, 32)


# --------------------------------------------------------------------------
# A brace-aware scope reader.  Nothing here is regex-over-C: comments and string
# literals are removed first, then `if (cond)` regions are found by matching
# parens and braces, so an enclosing guard is EXACT rather than "the nearest
# line that looked like an if".
# --------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    """Blank out comments and string/char literals, PRESERVING every offset.

    Offsets must be preserved because the caller reports source lines, and a
    length-changing strip would make every reported line a lie.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if src[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            for k in range(i, j):
                if src[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def _match_paren(src: str, i: int) -> int:
    """Index just past the ``)`` matching the ``(`` at ``i``."""
    depth = 0
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise AssertionError("unbalanced parenthesis in consumer_kernel.cuh")


def _match_brace(src: str, i: int) -> int:
    depth = 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise AssertionError("unbalanced brace in consumer_kernel.cuh")


def _if_regions(clean: str) -> list[tuple[str, int, int]]:
    """Every ``if (cond)`` in the file as ``(cond, body_start, body_end)``.

    Both the braced and the single-statement forms are handled; ``if constexpr``
    is included (its condition is a compile-time fork and it guards code exactly
    the same way).

    ``else`` BODIES ARE REGISTERED TOO, under the condition ``!(cond)``.  That is
    not decoration: the phase-C release is now a compile-time fork
    (``if constexpr (kNarrowGate) { narrow } else { wide }``), and a checker blind
    to ``else`` would see two retirements and two waits with no way to say which
    belongs with which — it would have to guess a pairing, and a gate that guesses
    is not a gate.  With the ``else`` arms named, each side of the fork carries its
    own ``kNarrowGate`` / ``!(kNarrowGate)`` tag and the invariant is discharged
    once per arm.
    """
    regions = []
    for m in re.finditer(r"\bif\s*(?:constexpr\s*)?\(", clean):
        lp = clean.index("(", m.start())
        rp = _match_paren(clean, lp)
        cond = clean[lp + 1:rp - 1]
        j = rp
        while j < len(clean) and clean[j] in " \t\n":
            j += 1
        # the ELSE arm, if any, guarded by the negation
        def _register_else(body_end: int, cond: str = cond) -> None:
            k = body_end
            while k < len(clean) and clean[k] in " \t\n":
                k += 1
            if not clean.startswith("else", k):
                return
            k += 4
            while k < len(clean) and clean[k] in " \t\n":
                k += 1
            if k >= len(clean):
                return
            if clean[k] == "{":
                e = _match_brace(clean, k)
            elif clean.startswith("if", k):
                return          # `else if` — the chained `if` is found on its own
            else:
                e = clean.find(";", k)
                e = len(clean) if e < 0 else e + 1
            regions.append((f"!({' '.join(cond.split())})", k, e))

        if j < len(clean) and clean[j] == "{":
            end = _match_brace(clean, j)
            _register_else(end)
        else:
            end = clean.find(";", j)
            end = len(clean) if end < 0 else end + 1
            _register_else(end)
        regions.append((" ".join(cond.split()), rp, end))
    return regions


def _guards_at(regions, off: int) -> list[str]:
    """Conditions of every ``if`` region containing ``off``, outermost first."""
    return [c for c, a, b in sorted(regions, key=lambda r: r[1]) if a <= off < b]


def _for_regions(clean: str) -> list[tuple[int, int]]:
    """Every braced ``for (...) { ... }`` body as ``(start, end)``."""
    out = []
    for m in re.finditer(r"\bfor\s*\(", clean):
        lp = clean.index("(", m.start())
        rp = _match_paren(clean, lp)
        j = rp
        while j < len(clean) and clean[j] in " \t\n":
            j += 1
        if j < len(clean) and clean[j] == "{":
            out.append((j, _match_brace(clean, j)))
    return out


#: The ``NM >= 2`` Gram-store fork bounds the storing warp set with a LOOP BREAK
#: rather than an ``if``: ``gt = warp + q * kWarps`` and ``if (gt >= X) break;``
#: (``consumer_kernel.cuh:3110-3111``).  Since ``q`` starts at 0, the smallest
#: ``gt`` a warp ever forms is ``warp`` itself, so the warps that store at all are
#: exactly ``warp < X``.  Recognising this is REQUIRED, not cosmetic: without it
#: the fork's stores read as unbounded (``kWarps``) and a correctly narrowed
#: retirement would be reported as incomplete.
_STRIDE_RE = re.compile(r"int\s+(\w+)\s*=\s*warp\s*\+\s*\w+\s*\*\s*kWarps\s*;")


def _loop_break_bound(clean: str, for_regions, off: int) -> str | None:
    """The ``warp <`` bound a ``gt = warp + q * kWarps`` / ``break`` loop imposes."""
    enclosing = [(a, b) for a, b in for_regions if a <= off < b]
    for a, b in sorted(enclosing, key=lambda r: r[1] - r[0]):  # innermost (smallest) first
        body = clean[a:b]
        m = _STRIDE_RE.search(body)
        if m is None:
            continue
        var = re.escape(m.group(1))
        brk = re.search(
            rf"if\s*\(\s*{var}\s*>=\s*([^)]+(?:\([^)]*\)[^)]*)*)\)\s*break\s*;", body)
        if brk:
            return " ".join(brk.group(1).split())
    return None


_WARP_CLAUSE = re.compile(r"^warp\s*<\s*(.+)$")


def _warp_set(guards: list[str]) -> tuple[str, bool, list[str]]:
    """Reduce enclosing guards to ``(upper_bound_expr, requires_mixed, others)``.

    A guard set constrains the warp index only through ``warp < X`` clauses and
    the CTA-uniform ``mixed`` bit; everything else (``w + 1 == kNWin``, the
    ``NM`` constexpr fork, ...) is CTA-uniform in ``warp`` and is returned
    separately so the caller can assert on it rather than have it silently
    dropped.
    """
    bounds, mixed, others = [], False, []
    for g in guards:
        for clause in re.split(r"&&", g):
            clause = clause.strip().strip("()").strip()
            if not clause:
                continue
            if clause == "mixed":
                mixed = True
                continue
            m = _WARP_CLAUSE.match(clause)
            if m:
                bounds.append(" ".join(m.group(1).split()))
                continue
            if "warp" in clause:
                others.append(clause)
    bound = bounds[-1] if bounds else "kWarps"
    return bound, mixed, others


def _eval_bound(expr: str, BT: int, kWarps: int) -> int:
    """Numeric value of a symbolic warp bound, over the instantiation axes."""
    NM = BT // 16
    kGramWarps = min(NM * (BT // 8), kWarps)
    env = {"BT": BT, "NM": NM, "kWarps": kWarps, "kGramWarps": kGramWarps}
    return int(eval(expr, {"__builtins__": {}}, env))  # noqa: S307 - closed env


# --------------------------------------------------------------------------
# Site extraction
# --------------------------------------------------------------------------

def _line_of(src: str, off: int) -> int:
    return src.count("\n", 0, off) + 1


def _sites(source: str) -> dict:
    """Locate the phase-C counter's release/acquire/store sites in a source."""
    clean = _strip_comments(source)
    regions = _if_regions(clean)
    loops = _for_regions(clean)

    retire = [m for m in re.finditer(r"retire_item\s*\(\s*&ctl\[kCtlGramDone\]", clean)]
    await_ = [m for m in re.finditer(
        r"await_count\s*\(\s*&ctl\[kCtlGramDone\]\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", clean)]
    #: `gm_put(Gm[` -- the STORE INTO THE GRAM, not the two overload DEFINITIONS at
    #: `:1125-1127`, which a bare `gm_put(` would also match and which carry no warp
    #: guard at all (they would read as unbounded stores and manufacture a false
    #: incompleteness against any narrowed retirement).
    gm_put = [m for m in re.finditer(r"\bgm_put\s*\(\s*Gm\[", clean)]
    read_await = [m for m in re.finditer(
        r"await_count\s*\(\s*&ctl\[kCtlReadDone\]\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", clean)]

    return dict(
        clean=clean, regions=regions,
        retire=[(m.start(), _guards_at(regions, m.start())) for m in retire],
        await_=[(m.start(), m.group(1), _guards_at(regions, m.start())) for m in await_],
        gm_put=[(m.start(), _guards_at(regions, m.start()),
                 _loop_break_bound(clean, loops, m.start())) for m in gm_put],
        read_await=[(m.start(), m.group(1)) for m in read_await],
        lines=lambda off: _line_of(source, off),
        source=source,
    )


# --------------------------------------------------------------------------
# THE CHECK.  Returns a list of violation strings; empty == safe.
# --------------------------------------------------------------------------

#: A site's ARM is the set of enclosing COMPILE-TIME fork conditions naming
#: ``kNarrowGate``.  The phase-C release is scoped to the ``bf16`` precision arms
#: (the ``tf32x3`` arms keep the 8-wide rendezvous, because narrowing them moves
#: register pressure into loop-resident spill on already-starved instantiations),
#: so the tree carries TWO retirements and TWO waits.  Pairing them by position
#: would be a guess; pairing them by arm tag is a fact off the source.
_ARM_RE = re.compile(r"kNarrowGate")


def _arm(guards: list[str]) -> frozenset:
    return frozenset(g for g in guards if _ARM_RE.search(g))


def _arm_name(arm: frozenset) -> str:
    return " & ".join(sorted(arm)) if arm else "<unforked>"


def check_phase_c_release(source: str) -> list[str]:
    """Discharge the release invariant ONCE PER ARM of the ``kNarrowGate`` fork.

    Returns a list of violation strings; empty == safe.
    """
    s = _sites(source)
    bad: list[str] = []

    if not s["retire"]:
        return ["found no retire into kCtlGramDone"]
    if not s["await_"]:
        return ["found no await on kCtlGramDone"]
    if not s["gm_put"]:
        return ["found no gm_put store site -- the parser has lost the Gram stores"]

    kwarps = re.search(r"constexpr int kWarps\s*=\s*kConsumerThreads\s*/\s*32", s["clean"])
    if kwarps is None:
        bad.append("kWarps is no longer `kConsumerThreads / 32`; the bound arithmetic "
                   "below is derived from that definition and must be re-derived")
    KW = 8  # kConsumerThreads == 256, asserted separately below

    retires = {}
    for off, guards in s["retire"]:
        retires.setdefault(_arm(guards), []).append((off, guards))
    awaits = {}
    for off, target, guards in s["await_"]:
        awaits.setdefault(_arm(guards), []).append((off, target, guards))

    if set(retires) != set(awaits):
        return [f"the retirement arms {sorted(map(_arm_name, retires))} and the wait "
                f"arms {sorted(map(_arm_name, awaits))} do not correspond; one side of "
                f"the compile-time fork retires or waits without the other"]
    for arm in retires:
        if len(retires[arm]) != 1 or len(awaits[arm]) != 1:
            return [f"arm {_arm_name(arm)} has {len(retires[arm])} retirement(s) and "
                    f"{len(awaits[arm])} wait(s); want exactly one of each"]

    for arm in sorted(retires, key=_arm_name):
        tag = f"[arm {_arm_name(arm)}] "
        (r_off, r_guards), = retires[arm]
        (a_off, a_target, a_guards), = awaits[arm]
        r_bound, r_mixed, r_other = _warp_set(r_guards)
        _a_bound, a_mixed, _ = _warp_set(a_guards)

        # --- HAZARD (design §3.4): the retire predicate must mirror the gm_put STORE
        # guard, never the GEMM guard.  The GEMM at the `NM == 1` fork carries an
        # extra `kw > 0` (the causal-prefix early exit) that the stores do NOT -- a
        # warp with `kw <= 0` still stores zeros.  A retirement derived from the GEMM
        # guard would skip that warp and the wait would HANG.
        if any("kw" in g for g in r_guards):
            bad.append(f"{tag}the retirement at line {_line_of(source, r_off)} is guarded "
                       f"by a condition mentioning `kw` ({r_guards}); it must mirror the "
                       f"gm_put store guard, not the GEMM guard -- a `kw <= 0` warp still "
                       f"stores zeros and would never retire, hanging the wait")
        if r_other:
            bad.append(f"{tag}the retirement's guard constrains `warp` in a form this "
                       f"gate cannot reduce: {r_other}")

        # --- CONDITION 2: T >= |R|.  With R = {mixed? && warp < r_bound}, |R| = r_bound.
        for BT in _BT_VALUES:
            try:
                n_r = _eval_bound(r_bound, BT, KW)
                n_t = _eval_bound(a_target, BT, KW)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{tag}cannot evaluate warp bound {r_bound!r}/{a_target!r} "
                           f"at BT={BT}: {exc}")
                break
            if n_t < n_r:
                bad.append(
                    f"{tag}UNSAFE RELEASE at BT={BT}: await_count target `{a_target}` = "
                    f"{n_t} is BELOW the retiring-warp count `{r_bound}` = {n_r} (retire "
                    f"at line {_line_of(source, r_off)}, await at line "
                    f"{_line_of(source, a_off)}). The wait can be satisfied by {n_t} "
                    f"retirements from warps that store no `Gm`, releasing a `Gm` "
                    f"consumer before the producer has published -- "
                    f"LATENCY_COVERAGE_DESIGN.md:129-132's named defect.")

        # --- CONDITION 1: R superset of S, over the gm_put store sites.  The stores
        # sit OUTSIDE the fork, so every store site applies to every arm.
        for off, guards, brk in s["gm_put"]:
            g_bound, g_mixed, _ = _warp_set(guards)
            if brk is not None and g_bound == "kWarps":
                g_bound = brk   # the NM >= 2 fork bounds by loop break, not by `if`
            for BT in _BT_VALUES:
                try:
                    n_g, n_r = _eval_bound(g_bound, BT, KW), _eval_bound(r_bound, BT, KW)
                except Exception:  # noqa: BLE001, S110
                    continue
                if n_g > n_r:
                    bad.append(
                        f"{tag}INCOMPLETE RELEASE at BT={BT}: the gm_put store at line "
                        f"{_line_of(source, off)} runs on warps < {g_bound} = {n_g} but "
                        f"only warps < {r_bound} = {n_r} retire; a storing warp never "
                        f"announces its store")
            if r_mixed and not g_mixed:
                bad.append(f"{tag}the retirement requires `mixed` but the gm_put store at "
                           f"line {_line_of(source, off)} does not -- a store outside the "
                           f"retiring set")

        # --- The wait's own guard may only ever be WEAKER than the retirement's, or
        # the CTA deadlocks: a warp that waits while the retirers skipped the retire.
        if a_mixed and not r_mixed:
            bad.append(f"{tag}the await is `mixed`-guarded but the retirement is not")
        if r_mixed and not a_mixed:
            bad.append(f"{tag}DEADLOCK: the retirement at line {_line_of(source, r_off)} "
                       f"is `mixed`-guarded but the await at line "
                       f"{_line_of(source, a_off)} is not -- on a `!mixed` tile nobody "
                       f"retires and every warp waits forever")

    return bad


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_the_phase_c_release_is_safe_in_the_shipped_source():
    """The invariant, on the tree as it stands.  Holds for the 8-wide form and for
    the narrowed one; it is a property of the release, not of a spelling."""
    violations = check_phase_c_release(KERNEL.read_text())
    assert not violations, "phase-C release is UNSAFE:\n  " + "\n  ".join(violations)


#: Mutant surgery, as PRIMITIVES rather than as literal string swaps.  The gate has
#: to reject the same defects whether the tree carries the 8-wide form or the
#: narrowed one, and a literal swap silently degrades to a no-op on the form it was
#: not written against -- which is how a must-fail arm quietly stops failing.  Each
#: primitive ASSERTS that it changed the source.
_RETIRE_STMT = re.compile(
    r"(?:if\s*\([^)]*\)\s*)?retire_item\(&ctl\[kCtlGramDone\], lane\);")
_AWAIT_STMT = re.compile(
    r"(?:if\s*\([^)]*\)\s*)?await_count\(&ctl\[kCtlGramDone\],\s*\w+\);")


def _narrow_index(sites: list, key) -> int:
    """Index of the site on the NARROWED arm of the ``kNarrowGate`` fork.

    The mutants are defects OF THE NARROWING, so they must be applied to the arm
    that narrows.  Mutating the wide arm would produce a source the checker rightly
    accepts, and the must-fail arm would silently stop failing -- the exact failure
    mode that already bit this file once.  With no fork present there is one site
    and it is the one.
    """
    if len(sites) == 1:
        return 0
    narrow = [i for i, site in enumerate(sites)
              if any(_ARM_RE.search(g) and not g.startswith("!") for g in key(site))]
    assert len(narrow) == 1, (
        f"cannot identify the narrowed arm among {len(sites)} sites: {narrow}")
    return narrow[0]


def _splice(src: str, rx, idx: int, stmt: str) -> str:
    """Replace the ``idx``-th match of ``rx`` in ``src`` with ``stmt``.

    Matching runs over the COMMENT-STRIPPED text (whose offsets are preserved by
    construction) so a statement quoted in a comment cannot be mutated by mistake,
    and the splice is applied to the original bytes at those offsets.
    """
    hits = list(rx.finditer(_strip_comments(src)))
    assert 1 <= len(hits) <= 2, f"surgery matched {len(hits)} sites, want 1 or 2"
    m = hits[idx]
    return src[:m.start()] + stmt + src[m.end():]


def _set_retire(src: str, stmt: str) -> str:
    """Force the NARROW arm's phase-C retirement to be exactly ``stmt``.  Setting it
    to what it already says is legitimate when COMPOSING a mutant, so only the site
    count is asserted here; each mutant separately asserts that the composed source
    differs from the tree."""
    sites = _sites(src)["retire"]
    return _splice(src, _RETIRE_STMT, _narrow_index(sites, lambda x: x[1]), stmt)


def _set_await(src: str, stmt: str) -> str:
    sites = _sites(src)["await_"]
    return _splice(src, _AWAIT_STMT, _narrow_index(sites, lambda x: x[2]), stmt)


def _mutated(src: str, mutant: str, name: str) -> str:
    assert mutant != src, f"mutant {name} composed to the UNCHANGED tree source -- it "\
                          f"would be testing the shipped form, not a defect"
    return mutant


def test_the_gate_is_not_vacuous_it_rejects_the_named_defect():
    """MUST-FAIL ARM.  Five synthetic mutants, each built by surgery on the real
    header, each of which the checker has to reject.  This is what distinguishes a
    gate from a comment (`head_major_landing.md:138-159`)."""
    src = KERNEL.read_text()

    # M-A -- THE DESIGN'S NAMED DEFECT (`LATENCY_COVERAGE_DESIGN.md:129-132`):
    # the target is lowered to `kGramWarps` while the retirement stays
    # UNCONDITIONAL, so warps 5-7 -- which store no `Gm` -- can release the waiter.
    m_a = _set_await(_set_retire(src, "retire_item(&ctl[kCtlGramDone], lane);"),
                     "await_count(&ctl[kCtlGramDone], kGramWarps);")
    bad = check_phase_c_release(_mutated(src, m_a, 'M-A'))
    assert any("UNSAFE RELEASE" in b for b in bad), \
        f"M-A (target lowered, retirement unconditional) was ACCEPTED: {bad}"

    # M-B -- the retirement derived from the GEMM guard rather than the store guard:
    # a `kw <= 0` warp still STORES ZEROS but would not retire, and the wait hangs.
    m_b = _set_retire(src, "if (mixed && warp < BT / 8 && kw > 0) "
                           "retire_item(&ctl[kCtlGramDone], lane);")
    bad = check_phase_c_release(_mutated(src, m_b, 'M-B'))
    assert any("kw" in b for b in bad), \
        f"M-B (retirement taken from the GEMM guard, `kw > 0`) was ACCEPTED: {bad}"

    # M-C -- the retirement `mixed`-guarded while the wait is not: on a `!mixed`
    # tile nobody retires and every warp waits forever.
    m_c = _set_await(
        _set_retire(src, "if (mixed && warp < kGramWarps) "
                         "retire_item(&ctl[kCtlGramDone], lane);"),
        "await_count(&ctl[kCtlGramDone], kGramWarps);")
    bad = check_phase_c_release(_mutated(src, m_c, 'M-C'))
    assert any("DEADLOCK" in b for b in bad), \
        f"M-C (retirement `mixed`-guarded, wait not) was ACCEPTED: {bad}"

    # M-D -- the skip taken on the WAIT only, leaving the retirement narrowed but
    # unconditional in `mixed`: the counter is then incremented on `!mixed` tiles
    # and, being reset per tile, that is benign -- but the target `kGramWarps`
    # against a retiring set that no longer requires `mixed` is not what the
    # comment claims.  The check that catches it is the `mixed` asymmetry.
    m_d = _set_await(_set_retire(src, "if (warp < kGramWarps) "
                                      "retire_item(&ctl[kCtlGramDone], lane);"),
                     "if (mixed) await_count(&ctl[kCtlGramDone], kGramWarps);")
    bad = check_phase_c_release(_mutated(src, m_d, 'M-D'))
    assert bad, "M-D (wait `mixed`-guarded, retirement not) was ACCEPTED"

    # M-E -- the WIDE arm narrowed too, i.e. the scope erased.  This is the mutant
    # the per-arm discharge exists for: it leaves the narrow arm perfectly correct,
    # so a checker that examined only "the" retirement and "the" wait -- or that
    # paired them by position -- would pass it.  It is also the shape a future
    # editor produces by "simplifying away" the fork.
    m_e = src.replace("      await_count(&ctl[kCtlGramDone], kWarps);",
                      "      await_count(&ctl[kCtlGramDone], kGramWarps);")
    bad = check_phase_c_release(_mutated(src, m_e, 'M-E'))
    assert any("UNSAFE RELEASE" in b and "!(kNarrowGate)" in b for b in bad), \
        f"M-E (the wide arm narrowed, scope erased) was ACCEPTED: {bad}"


def test_the_parser_actually_found_the_sites():
    """Anti-vacuity for the PARSER: a checker that located nothing would return no
    violations and read as a pass."""
    s = _sites(KERNEL.read_text())
    assert len(s["retire"]) == len(s["await_"]) >= 1, (s["retire"], s["await_"])
    assert len(s["gm_put"]) == 8, f"{len(s['gm_put'])} gm_put(Gm[..]) stores located, want 8"
    assert len(s["read_await"]) == 1, s["read_await"]
    #: The two forks must be seen as BOUNDED, each by its own mechanism -- an `if`
    #: at `NM == 1` and a loop break at `NM >= 2`.  A store site whose warp set
    #: reduced to the unbounded `kWarps` would make CONDITION 1 unfalsifiable.
    for off, guards, brk in s["gm_put"]:
        bound, mixed, _ = _warp_set(guards)
        if bound == "kWarps":
            bound = brk
        assert bound is not None and bound != "kWarps", (
            f"the gm_put store at line {_line_of(KERNEL.read_text(), off)} reduced to an "
            f"UNBOUNDED warp set; CONDITION 1 cannot fail against it")
        assert mixed, f"the gm_put store at line {_line_of(KERNEL.read_text(), off)} is not "\
                      f"`mixed`-guarded; the skip in §3.3 assumes it is"
    src = KERNEL.read_text()
    assert re.search(r"constexpr int kConsumerThreads\s*=\s*256", _strip_comments(src)), \
        "kConsumerThreads is no longer 256; kWarps == 8 is assumed by this gate"


def test_the_narrow_gate_scope_is_a_precision_axis_and_both_arms_exist():
    """The scope is a FORMAT axis the dispatch already forks on, and BOTH arms are
    really compiled — not one arm plus a comment.

    Item 1 is bit-identical everywhere but it is not free everywhere: on the
    `tf32x3` arms the extra guards re-schedule `ptxas` on already-starved
    instantiations and the 1,216-entry ratchet measured in-loop `LDL`/`STL` growth
    on 12 of them.  So the narrowing is scoped to the `bf16` arms.  This pins the
    scope to `stage_is_narrow<READ_PREC, FOLD_PREC>()` — a compile-time precision
    axis — and refuses a size- or occupancy-shaped one, which would be
    branch-on-threshold.
    """
    src = KERNEL.read_text()
    clean = _strip_comments(src)
    assert re.search(
        r"constexpr bool kNarrowGate\s*=\s*stage_is_narrow<\s*READ_PREC\s*,\s*"
        r"FOLD_PREC\s*>\(\)\s*;", clean), (
        "kNarrowGate is not defined as `stage_is_narrow<READ_PREC, FOLD_PREC>()`; the "
        "phase-C scope must be the existing compile-time PRECISION axis (the one "
        "`StageT`, `GmT` and `GMP` already fork on), never a size or occupancy test")
    for forbidden in ("BT ==", "BT >", "BT <", "BC ==", "DV ==", "n_owners", "occupancy"):
        assert forbidden not in _scope_expr(clean), (
            f"the phase-C scope mentions {forbidden!r} -- that is a size/threshold "
            f"branch, not a structural axis")

    s = _sites(src)
    arms_r = {_arm_name(_arm(g)) for _o, g in s["retire"]}
    arms_a = {_arm_name(_arm(g)) for _o, _t, g in s["await_"]}
    assert arms_r == arms_a == {"kNarrowGate", "!(kNarrowGate)"}, (
        f"expected both arms of the compile-time fork to carry a retirement and a "
        f"wait; retirements on {sorted(arms_r)}, waits on {sorted(arms_a)}")

    # The WIDE arm must be the untouched 8-wide reference: unconditional retirement,
    # target `kWarps`.  If it ever drifts, the "tf32x3 keeps what it had" claim that
    # justifies the scope stops being true and the byte-identity proof goes stale.
    (_off, target, _g), = [a for a in s["await_"]
                           if _arm_name(_arm(a[2])) == "!(kNarrowGate)"]
    assert target == "kWarps", (
        f"the WIDE (tf32x3) arm waits on `{target}`, not `kWarps` -- that arm is the "
        f"correctness reference and is supposed to be byte-identical to the shipped "
        f"kernel")
    (_off, guards), = [r for r in s["retire"]
                       if _arm_name(_arm(r[1])) == "!(kNarrowGate)"]
    bound, mixed, _ = _warp_set(guards)
    assert bound == "kWarps" and not mixed, (
        f"the WIDE arm's retirement is guarded (`{bound}`, mixed={mixed}); it must be "
        f"unconditional, exactly as the shipped kernel has it")


def _scope_expr(clean: str) -> str:
    m = re.search(r"constexpr bool kNarrowGate\s*=([^;]*);", clean)
    return m.group(1) if m else ""


def test_the_read_done_counter_is_untouched_and_still_cta_wide():
    """`kCtlReadDone` is NOT narrowable (design §3.4): its dependence is the readout
    reading `S` before the fold writes it, and at `DV = 64` all eight warps run a
    readout tile.  It is genuinely 8-wide and this pins it."""
    s = _sites(KERNEL.read_text())
    (_off, target), = s["read_await"]
    assert target == "kWarps", (
        f"the kCtlReadDone wait's target is `{target}`, not `kWarps`; that counter is "
        f"genuinely CTA-wide (all NT_DV = 8 warps run a readout tile at DV = 64) and "
        f"narrowing it is a different, unargued change")

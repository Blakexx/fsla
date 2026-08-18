"""WHAT is timed: one entry per custom kernel, plus the two compositions.

**THE TAXONOMY (ruling 2026-08-14).** The UNIT subjects mirror the oracle-test roster one
for one -- the chunk forward, the stats pass, the block bits, the decode step, the entmax
solves, the divide, the backward seeds and the two reverse scans -- because a bench
roster that does not match the correctness roster is a roster with kernels nobody
measures. The INTEGRATION subjects
are compositions the unit rows cannot add up to: the whole layer step, and the
prefill->decode continuation across one carried state.

A subject is a factory: it takes a built cell fixture and returns a
:class:`bench.pairing.Arm` whose call issues exactly the launch being priced. Everything
the launch does NOT pay for -- the union table, the packed sides, the state plane, the
routing -- is the fixture's :class:`~rola.engine.plan.ChunkPlan`, built once by the chunk
DAG at `Cell.build()` and read off here: outside every timed callable, as before, and
now identical by construction to what a real call hands the same entries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import rola
from bench.pairing import Arm
from rola.engine.dags.decode_dag import decode_forward
from rola.engine.rules.arm import CHUNK_TOKENS
from rola.engine.types import Mask
from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.routing.entmax.production import production_routing_factor_levels


def _widths(fx) -> list:
    return list(fx["plan"].widths)


def chunk_forward(fx) -> Arm:
    """The chunk consumer's own launch, with the union table PREBUILT.

    `chunk_forward_raw` rather than the fused entry: the table and the divide are the
    other two subjects on this roster, and a row that contains all three cannot be
    attributed to any of them.
    """
    m, plan = extension(), fx["plan"]
    tbl, widths, v = plan.union_table, _widths(fx), fx["v_kernel"]
    c, bc = fx["C"], fx["BC"]

    def call():
        return m.chunk_forward_raw(plan.read_plane, plan.write_plane, plan.g_write, v,
                                   tbl, c, bc, widths, plan.run_table, None, None, None,
                                   None)

    return Arm(name="chunk_forward_raw", call=call)


def chunk_facts(fx) -> Arm:
    """The facts pass: the union table and the atom-grain write bitmap, one pass.

    Priced under the STATEFUL call class with the bitmap asked for -- the most a
    prefill call pays here. The stateless class is the same launch with the
    atom-grain emission compiled out.
    """
    m, plan, widths = extension(), fx["plan"], _widths(fx)
    pr, pw, c, bc = plan.read_plane, plan.write_plane, fx["C"], fx["BC"]

    def call():
        return m.chunk_facts(pr, pw, c, bc, widths, Mask.M2, True)

    return Arm(name="chunk_facts", call=call)


def chunk_block_bits(fx) -> Arm:
    """The block-grain liveness reduction the sparse grid launches over."""
    m, widths = extension(), _widths(fx)
    tbl, bc, tokens = fx["plan"].union_table, fx["BC"], fx["routes"].tokens

    def call():
        return m.chunk_block_bits(tbl, bc, widths, tokens, None)

    return Arm(name="chunk_block_bits", call=call)


def _numerator(fx):
    """One forward's `(num, den)`, for the two readout subjects to consume."""
    m, plan = extension(), fx["plan"]
    return m.chunk_forward_raw(plan.read_plane, plan.write_plane, plan.g_write,
                               fx["v_kernel"], plan.union_table, fx["C"], fx["BC"],
                               _widths(fx), plan.run_table, None, None, None, None)


def divide_pass(fx) -> Arm:
    """The readout ratio: one elementwise pass that also narrows and transposes."""
    m, read_mass = extension(), fx["plan"].read_mass
    num, den = _numerator(fx)
    out_dtype = fx["cell"].torch_dtype

    def call():
        return m.chunk_divide_pass(num, den, read_mass, READOUT_EPS, out_dtype)

    return Arm(name="chunk_divide_pass", call=call)


def backward_seeds(fx) -> Arm:
    """The readout's backward seeds -- the one reverse-pass entry that is wired."""
    m, read_mass = extension(), fx["plan"].read_mass
    num, den = _numerator(fx)
    y = m.chunk_divide_pass(num, den, read_mass, READOUT_EPS, torch.float32)
    dy = torch.randn_like(y)

    def call():
        return m.chunk_backward_seeds(dy, y, den, read_mass, READOUT_EPS)

    return Arm(name="chunk_backward_seeds", call=call)


def _seeds(fx):
    """The readout seeds, built once OUTSIDE every timed callable.

    Both scans read `dnum` and `dden`, and neither pays for them: they are the reverse
    pass' own first launch and have a subject of their own.
    """
    m, plan = extension(), fx["plan"]
    num, den = _numerator(fx)
    y = m.chunk_divide_pass(num, den, plan.read_mass, READOUT_EPS, torch.float32)
    dy = torch.randn_like(y)
    return m.chunk_backward_seeds(dy, y, den, plan.read_mass, READOUT_EPS)[:2]


def _scan_operands(fx):
    plan, (dnum, dden) = fx["plan"], _seeds(fx)
    return (plan.read_plane, plan.write_plane, plan.g_write, fx["v_kernel"], dnum, dden,
            plan.union_table, fx["C"], fx["BC"], _widths(fx))


def scan_a(fx) -> Arm:
    """SCAN A's launch: the ascending rebuild and the read side's digit gradient.

    The shipped reverse pass is two independent launches, so it is two subjects: a row
    containing both could be attributed to neither.
    """
    m, operands = extension(), _scan_operands(fx)

    def call():
        return m.chunk_scan_a_probe(*operands)

    return Arm(name="scan_a", call=call)


def scan_b(fx) -> Arm:
    """SCAN B's launch: the descending adjoint -- `dv`, `d gw` and `d pw`."""
    m, operands = extension(), _scan_operands(fx)

    def call():
        return m.chunk_scan_b_probe(*operands)

    return Arm(name="scan_b", call=call)


def entmax_solve(fx) -> Arm:
    """The producer's batched multi-level solve, on this cell's own routing.

    The logits are drawn once from the cell's seed and held: what is priced is the SOLVE,
    not the projection that produced them.
    """
    producer = fx["layer"].routes
    routing = producer.routing
    cell = fx["cell"]
    g = torch.Generator(device="cuda").manual_seed(cell.seed + 900)
    shape = (cell.B * cell.H, cell.T, routing.packed_logit_width)
    read_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g)
    write_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=g)

    def call():
        return production_routing_factor_levels(read_logits, write_logits, routing)

    return Arm(name="entmax_solve", call=call)


def decode_step(fx) -> Arm:
    """ONE carried single-token step, on a state the prefill already populated.

    The state is MUTATED by the call, which is what a decode step is; repeated timing
    therefore walks the sequence forward. The arena's growth saturates within the first
    few steps at these widths, so the timed reps price a steady-state step and not the
    admission that precedes it -- which the warmup launches are what guarantee.
    """
    cell, producer = fx["cell"], fx["layer"].routes
    if cell.decode_steps == 0:
        raise ValueError(f"{cell.name} is a prefill-only cell; it has no decode subject")
    state = rola.state()
    with torch.no_grad():
        _, state = rola.rola_op(producer(fx["x"][:, :cell.T]), fx["v_prefill"], state=state)
    token = fx["x"][:, cell.T:cell.T + 1]
    v_token = fx["v"][:, cell.T:cell.T + 1].contiguous()

    def call():
        with torch.no_grad():
            y, _ = decode_forward(producer(token), v_token, state)
        return y

    return Arm(name="decode_step", call=call)


def layer_step(fx) -> Arm:
    """The whole layer forward: projections, producer, kernel, readout, output projection.

    The backend is asserted once, outside the timed callable: a row that silently measured
    a non-CUDA arm is a row about a different program.
    """
    layer, x = fx["layer"], fx["x"][:, :fx["cell"].T]

    def call():
        with torch.no_grad():
            return layer(x)

    call()
    if layer.last_execution_backend != "cuda":
        raise SystemExit(f"REFUSING TO MEASURE: {fx['cell'].name} reached the "
                         f"{layer.last_execution_backend!r} arm, not the CUDA kernel.")
    return Arm(name="layer_step", call=call)


def prefill_decode(fx) -> Arm:
    """The continuation: one prefill and `decode_steps` carried steps, from a FRESH state.

    The state is rebuilt inside the callable because that is the composition being
    priced -- a continuation that reuses a warm state measures the steps only, which is
    the `decode_step` subject and is already on the roster.
    """
    cell, producer = fx["cell"], fx["layer"].routes
    if cell.decode_steps == 0:
        raise ValueError(f"{cell.name} is a prefill-only cell; it has no continuation")
    x, v = fx["x"], fx["v"]

    def call():
        with torch.no_grad():
            state = rola.state()
            y, state = rola.rola_op(producer(x[:, :cell.T]), fx["v_prefill"], state=state)
            for step in range(cell.decode_steps):
                token = slice(cell.T + step, cell.T + step + 1)
                y, state = decode_forward(producer(x[:, token]), v[:, token].contiguous(),
                                          state)
        return y

    return Arm(name="prefill_decode", call=call)


#: THE UNIT ROSTER: one entry per custom kernel, mirroring the oracle tests.
UNIT = {
    "chunk_forward": chunk_forward,
    "chunk_facts": chunk_facts,
    "chunk_block_bits": chunk_block_bits,
    "divide_pass": divide_pass,
    "backward_seeds": backward_seeds,
    "entmax_solve": entmax_solve,
    "decode_step": decode_step,
    "scan_a": scan_a,
    "scan_b": scan_b,
}

#: THE INTEGRATION ROSTER: compositions the unit rows cannot be summed into.
INTEGRATION = {
    "layer_step": layer_step,
    "prefill_decode": prefill_decode,
}

SUBJECTS = {**UNIT, **INTEGRATION}

#: THE BUILT ARM LISTS, READ FROM THE GENERATOR THAT OWNS THEM. The reverse pass'
#: matrix is narrower than the forward's, so a cell outside it carries no scan launch to
#: price. Taken from `tools/gen_shards.py` rather than restated here: a second copy of an
#: arm list is the drift shape this project has already been bitten by, and that module
#: is pure host Python with no device in it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import gen_shards  # noqa: E402


#: The arm rule's own preference order (`rola.engine.rules.arm`), applied to the built
#: matrix: one chunk length, the pinned block width where the topology has it and the
#: largest built one otherwise.
def _arm_of(cell) -> tuple[int, int] | None:
    depth, width = len(cell.widths), cell.widths[0]
    bcs = [bc for (c, bc, d, b) in gen_shards.CHUNK_ARMS
           if d == depth and b == width and c == CHUNK_TOKENS]
    if not bcs:
        return None
    return (CHUNK_TOKENS, 128 if 128 in bcs else max(bcs))


#: Which subjects a cell can carry. The decode subjects need a cell that continues; the
#: scan subjects need an arm their own matrix is built for. Every other subject runs on
#: every cell.
def applicable(cell) -> list:
    decode_only = ("decode_step", "prefill_decode")
    scan_only = ("scan_a", "scan_b")
    arm = _arm_of(cell)
    has_scan = arm is not None and (*arm, len(cell.widths), cell.widths[0]) in gen_shards.SCAN_ARMS
    out = []
    for name in SUBJECTS:
        if name in decode_only and cell.decode_steps == 0:
            continue
        if name in scan_only and not has_scan:
            continue
        out.append(name)
    return out

"""WHAT is timed: one entry per custom kernel, plus the two compositions.

**THE TAXONOMY (ruling 2026-08-14).** The UNIT subjects mirror the oracle-test roster one
for one -- the chunk forward, the stats pass, the block bits, the decode step, the entmax
solves, the divide and the backward seeds -- because a bench roster that does not match
the correctness roster is a roster with kernels nobody measures. The INTEGRATION subjects
are compositions the unit rows cannot add up to: the whole layer step, and the
prefill->decode continuation across one carried state.

A subject is a factory: it takes a built cell fixture and returns a
:class:`bench.pairing.Arm` whose call issues exactly the launch being priced. Everything
the launch does NOT pay for -- the union table, the packed sides, the state plane, the
routing -- is built here, once, outside the timed callable.
"""
from __future__ import annotations

import torch

import rola
from bench.pairing import Arm
from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.ops.decode import decode_forward
from rola.routing.entmax.production import production_routing_factor_levels


def _widths(fx) -> list:
    return [level.width for level in fx["routes"].topology.levels]


def _union_table(fx):
    packed = fx["packed"]
    return extension().chunk_union_table(packed.pr, packed.pw, fx["v_kernel"],
                                         fx["C"], fx["BC"], _widths(fx))


def chunk_forward(fx) -> Arm:
    """The chunk consumer's own launch, with the union table PREBUILT.

    `chunk_forward_raw` rather than the fused entry: the table and the divide are the
    other two subjects on this roster, and a row that contains all three cannot be
    attributed to any of them.
    """
    m, packed = extension(), fx["packed"]
    tbl, widths, v = _union_table(fx), _widths(fx), fx["v_kernel"]
    c, bc = fx["C"], fx["BC"]

    def call():
        return m.chunk_forward_raw(packed.pr, packed.pw, packed.gw, v, tbl, c, bc,
                                   widths, None, None, None, None)

    return Arm(name="chunk_forward_raw", call=call)


def chunk_stats(fx) -> Arm:
    """The atom-grain write bitmap: the page plan's input, taken before an arm exists."""
    m, packed, widths = extension(), fx["packed"], _widths(fx)

    def call():
        return m.chunk_stats(packed.pw, widths)

    return Arm(name="chunk_stats", call=call)


def chunk_block_bits(fx) -> Arm:
    """The block-grain liveness reduction the sparse grid launches over."""
    m, widths = extension(), _widths(fx)
    tbl, bc, tokens = _union_table(fx), fx["BC"], fx["routes"].tokens

    def call():
        return m.chunk_block_bits(tbl, bc, widths, tokens, None)

    return Arm(name="chunk_block_bits", call=call)


def _numerator(fx):
    """One forward's `(num, den)`, for the two readout subjects to consume."""
    m, packed = extension(), fx["packed"]
    return m.chunk_forward_raw(packed.pr, packed.pw, packed.gw, fx["v_kernel"],
                               _union_table(fx), fx["C"], fx["BC"], _widths(fx),
                               None, None, None, None)


def divide_pass(fx) -> Arm:
    """The readout ratio: one elementwise pass that also narrows and transposes."""
    m, packed = extension(), fx["packed"]
    num, den = _numerator(fx)
    out_dtype = fx["cell"].torch_dtype

    def call():
        return m.chunk_divide_pass(num, den, packed.rm, READOUT_EPS, out_dtype)

    return Arm(name="chunk_divide_pass", call=call)


def backward_seeds(fx) -> Arm:
    """The readout's backward seeds -- the one reverse-pass entry that is wired."""
    m, packed = extension(), fx["packed"]
    num, den = _numerator(fx)
    y = m.chunk_divide_pass(num, den, packed.rm, READOUT_EPS, torch.float32)
    dy = torch.randn_like(y)

    def call():
        return m.chunk_backward_seeds(dy, y, den, packed.rm, READOUT_EPS)

    return Arm(name="chunk_backward_seeds", call=call)


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
    cell, gateway = fx["cell"], fx["layer"].routes
    if cell.decode_steps == 0:
        raise ValueError(f"{cell.name} is a prefill-only cell; it has no decode subject")
    state = rola.state()
    with torch.no_grad():
        _, state = rola.rola_op(gateway(fx["x"][:, :cell.T]), fx["v_prefill"], state=state)
    token = fx["x"][:, cell.T:cell.T + 1]
    v_token = fx["v"][:, cell.T:cell.T + 1].contiguous()

    def call():
        with torch.no_grad():
            y, _ = decode_forward(gateway(token), v_token, state)
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
    cell, gateway = fx["cell"], fx["layer"].routes
    if cell.decode_steps == 0:
        raise ValueError(f"{cell.name} is a prefill-only cell; it has no continuation")
    x, v = fx["x"], fx["v"]

    def call():
        with torch.no_grad():
            state = rola.state()
            y, state = rola.rola_op(gateway(x[:, :cell.T]), fx["v_prefill"], state=state)
            for step in range(cell.decode_steps):
                token = slice(cell.T + step, cell.T + step + 1)
                y, state = decode_forward(gateway(x[:, token]), v[:, token].contiguous(),
                                          state)
        return y

    return Arm(name="prefill_decode", call=call)


#: THE UNIT ROSTER: one entry per custom kernel, mirroring the oracle tests.
UNIT = {
    "chunk_forward": chunk_forward,
    "chunk_stats": chunk_stats,
    "chunk_block_bits": chunk_block_bits,
    "divide_pass": divide_pass,
    "backward_seeds": backward_seeds,
    "entmax_solve": entmax_solve,
    "decode_step": decode_step,
}

#: THE INTEGRATION ROSTER: compositions the unit rows cannot be summed into.
INTEGRATION = {
    "layer_step": layer_step,
    "prefill_decode": prefill_decode,
}

SUBJECTS = {**UNIT, **INTEGRATION}

#: Which subjects a cell can carry. The decode subjects need a cell that continues; every
#: other subject runs on every cell.
def applicable(cell) -> list:
    decode_only = ("decode_step", "prefill_decode")
    return [name for name in SUBJECTS
            if cell.decode_steps > 0 or name not in decode_only]

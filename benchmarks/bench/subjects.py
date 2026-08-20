"""WHAT is timed: one entry per surviving custom kernel.

**THE TAXONOMY (ruling 2026-08-14).** The UNIT subjects mirror the oracle-test
roster one for one -- the facts pass, the block bits, the decode step and the
entmax solves -- because a bench roster that does not match the correctness
roster is a roster with kernels nobody measures. The consumer subjects and the
two compositions over them left with the K31 deletion batch (baseline = tag
`baseline/pre-k31`); the R1 rebuild re-registers its own.

A subject is a factory: it takes a built cell fixture and returns a
:class:`bench.pairing.Arm` whose call issues exactly the launch being priced. Everything
the launch does NOT pay for -- the union table, the packed sides, the state plane, the
routing -- is the fixture's :class:`~rola.engine.plan.ChunkPlan`, built once by the chunk
DAG at `Cell.build()` and read off here: outside every timed callable, as before, and
now identical by construction to what a real call hands the same entries.
"""
from __future__ import annotations

import torch

import rola
from bench.pairing import Arm
from rola.engine.dags.decode_dag import decode_forward
from rola.engine.types import Mask
from rola.ops._ext import extension
from rola.routing.entmax.production import production_routing_factor_levels


def _widths(fx) -> list:
    return list(fx["plan"].widths)


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
    #: K31: the prefill arm that populated this state is deleted (baseline = tag
    #: `baseline/pre-k31`); the seed binds through the facade's own entry and
    #: commits exactly the pages the prefill routing writes -- a decode step's
    #: cost is a function of residency and routing, not of the numbers stored.
    from rola.engine.facts import planes

    state = rola.state()
    with torch.no_grad():
        routes = fx["plan"].routes
        bits = planes.atom_bits(planes.pack_side(routes.write), fx["cell"].widths)
        _s, arena = state._kernel_entry(routes, bits, d_v=fx["layer"].d_v,
                                        paging=True, BC=fx["BC"])
        arena.wait()
    token = fx["x"][:, cell.T:cell.T + 1]
    v_token = fx["v"][:, cell.T:cell.T + 1].contiguous()

    def call():
        with torch.no_grad():
            y, _ = decode_forward(producer(token), v_token, state)
        return y

    return Arm(name="decode_step", call=call)



#: THE UNIT ROSTER: one entry per surviving custom kernel, mirroring the oracle
#: tests.  The consumer subjects (`chunk_forward`, `divide_pass`,
#: `backward_seeds`, `scan_a`, `scan_b`) and the compositions over them
#: (`layer_step`, `prefill_decode`) left with the K31 deletion batch
#: (baseline = tag `baseline/pre-k31`); the R1 rebuild re-registers its own.
SUBJECTS = {
    "chunk_facts": chunk_facts,
    "chunk_block_bits": chunk_block_bits,
    "entmax_solve": entmax_solve,
    "decode_step": decode_step,
}

#: Which subjects a cell can carry. The decode subject needs a cell that
#: continues; every other subject runs on every cell.
def applicable(cell) -> list:
    return [name for name in SUBJECTS
            if name != "decode_step" or cell.decode_steps != 0]

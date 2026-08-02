"""The consumer benchmark harness, split by ROLE (design §6.9).

Four modules, because the file they came from had four jobs in it and only one
of them is the methodology:

* :mod:`bench.cells` -- WHAT is measured. Frozen, named, seeded fixtures plus the
  committed statistics manifest. A cell is its constructor and its seed.
* :mod:`bench.pairing` -- HOW it is measured. **The interleaved paired-arm engine.
  This is THE methodology of every number this repository publishes**, and its
  module docstring is where it is written down.
* :mod:`bench.instruments` -- what is measured ABOUT it beyond wall time: the
  gather's realized extents, the decay preamble's cost.
* :mod:`bench.drivers` -- one thin function per QUESTION, holding no machinery of
  its own. A driver that grows machinery is machinery that escaped review.
"""

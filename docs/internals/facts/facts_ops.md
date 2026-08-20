# `csrc/rola/src/facts/facts_ops.cuh` — the facts kernels' device operations

The subset of the retired consumer baseline's operation catalogue
(`chunk/ops.cuh`, deleted with the K31 batch; baseline = tag
`baseline/pre-k31`) that the union-table pass and the two bitmap reductions
use: the gathered-address discipline (`gather_ptr`, a base formed once and
displaced by a zero-extended 32-bit byte offset; `pin_address`, a base ptxas
cannot rematerialize), the lockstep unit's vote (`lane_ballot`,
`kBallotLanes`), the CTA barrier (`rendezvous`), and the zero-run test
(`or_run` over `kBf16MagnitudeMask` — a bf16 run is entirely zero exactly when
the OR of its storage words has no magnitude bit set). Bodies are verbatim from
the consumer's catalogue: the extraction must not move a single SASS body, and
the per-instantiation `sass_sha256` rows in `tools/manifests/` are the check.

# The chunk family's DAG

Mirror of `rola/dags/chunk_dag.py` and the chunk half of `rola/facts/nodes.py`.
Framework vocabulary — masks, seeds, the fact bag, the Join — is
[the facts engine's page](facts.md); the band-2 folds this tuple walks are
[the rule layer's](rules.md).

Symbols: `C` chunk tokens, `BC` owner block, `N` leaf count, `BH = B*H`, atom = 16
consecutive leaves in canonical order (the page granule).

## <a id="the-two-masks"></a>1. M1 and M2 are one tuple

```
paging_preference   -> paging                        @host  RULE
topology            -> widths, d_v                   @host
classify_call       -> call_class                    @host
mask                -> mask                          @host  RULE
manifest            -> manifest                      @host
arm                 -> arm                           @host  RULE
pack_planes         -> read/write plane, g_write, read_mass   @main
atom_bits           -> atom_bits                     @main   [M2]
residency           -> state_in, arena               @side   [M2] host_sync
pack_v              -> v_packed                      @main
join                                                 @main   [M2] waits
backing             -> page_table, state_out         @host
union_table         -> union_table                   @main
block_bits          -> block_bits                    @main
build_plan          -> plan                          @host
```

M1 (prefill-stateless) is M2 (prefill-stateful) with the residency band masked out:
`atom_bits`, `residency` and `join` do not run and their facts are null. The state
plane is not allocated and the kernel variant with no state I/O compiled into it is
the one that runs, so a stateless call pays nothing for the feature.

The critical path is `pack_planes -> union_table -> block_bits -> execute`. The three
rules, the topology, the manifest and the call class are host-only and sit on the
parallel front — which is why band 2 costs nothing: no rule is ever on the critical
path.

The mask is a FACT the MaskRule produces, and the runner reads it rather than the call
class: the band a node belongs to is a decision, and the walk consults the decision.
Every node before that rule runs under every mask by construction.

## <a id="the-absent-arm-edge"></a>2. `atom_bits` has no `arm` edge

The atom bitmap is keyed to the 16-leaf quantum and to nothing else, so it is a
property of the routing alone and is legitimately taken BEFORE an arm is chosen.
That is what makes residency an OUTPUT of the plan rather than a consequence of it,
and it is why the admission can run concurrently with the arm's own work. **A DAG
that gave `atom_bits` an `arm` edge would silently destroy that overlap**, so the
absence is asserted at import rather than left to a reader's care.

## <a id="the-join"></a>3. The Join sits before the block bitmap

`residency` runs the admission — the page-table write and the slot zero-init — on the
arena's side stream, and records the one event. `join` waits on it.

**Where the wait sits is a correctness fact.** `block_bits` READS the page table: the
`S` bit is a residency fact, so the block bitmap is downstream of the admission's
TABLE WRITE and not merely of the launch that follows. A wait placed immediately
before the consumer launch would leave a race between the side-stream table write and
the block bitmap's residency read. The Join is therefore upstream of `block_bits`,
never between it and `execute`.

## <a id="host-order"></a>4. The host order of a stateful call

The DAG's order IS the host order, and the two things that used to make it delicate
are now declarations rather than conventions:

* `residency` is the only node permitted a device-to-host read. `plan_exact`'s single
  size read is the only host synchronization in the sequence, and it is the read that
  also SIZES the admission. A fresh state plans `fresh=True` (call-scoped, the
  unpaged "allocate zeros every call" semantics); a continuation plans `fresh=False`,
  so residency becomes the UNION, which is what continuing a sequence means.
* `BC` reaches the state's extent policy from the plan's own `arm`, so the refusal
  check, the allocator policy and the launch cannot disagree about which arm this
  call runs. It is never stored on the state: it is allocator policy, and every
  ADDRESS is `BC`-invariant by the atom re-key
  ([`state.md` §4](../state.md#backing-policy)).

## <a id="entering-without-a-state"></a>5. Driving the arm without a state object

`ChunkContext` takes `state_in` and `arena` directly as an alternative to a
`RoLAState`. The `residency` node hands those straight back when no state is engaged,
which is how a test or a bring-up driver reaches the kernel arm without the facade's
state lifecycle — through the same DAG, the same plan and the same launch.

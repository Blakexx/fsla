# `device_probe`

Mirrors `measure/lab/experiments/device_probe.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does this device report for every fact the contract registry's schema reads from the device -- its compute capability, registers and shared
memory per SM, shared memory per CTA (opt-in), threads per SM, SM count and L2 bytes -- and do the runtime's two routes to each
(`cudaGetDeviceProperties` and `cudaDeviceGetAttribute`) and the device's own registers (`%nsmid`, `%total_smem_size`) agree?

THE FORM. No timing. The host reads each fact by its property and by its attribute; one CTA of one warp reads the two special registers and stores them; each
fact is recorded as a row of its own name (`cc` is major x 100 + minor x 10, as the schema says), beside the bus, slot and domain the driver reports. Three checks: the two
routes are equal, the registers were read, and the device's SM count is within the register's SM identifiers. The kernel declares its special-register read.

THE CANDIDATE. `lab page device_probe --instance LABEL` turns the record into `hardware/parts/device/<id>.candidate.json`: each fact the schema reads
from the device with its value and the row that measured it, `tensor` and `instructions` from the architecture table's row of the device's compute capability
(`documented_by`: no lab experiment measures the set, and none is re-measured here), and the key `unestablished` naming what the table has no row for. The schema gains `sm_count` and `l2_bytes`, which the candidate already carries, with the first reviewed part. The registry
refuses a candidate until a reviewer renames it to a part ([`registry.md#probe`](../internals/hardware/registry.md#probe)).

NOT YET MEASURED: it runs on the next H100 session.

Record: none yet; `lab run device_probe --record` writes it.

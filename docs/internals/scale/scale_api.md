# `csrc/rola/src/scale/scale_api.cuh` — the scale pass's host entry

The one entry the pybind unit binds (`csrc/rola/rola_api.cpp`'s `scale_forward`); `rola.ops.scale.scale` is its Python
caller. What the pass computes is [`scale.md`](scale.md).

<a id="scale-forward"></a>

## `scale_forward`

`scale_forward(read_rows, write_rows, fm, mass, bytes, widths, level_s, heads, inv_factor)`: the bundle's two levels of
rows a side (as `RouteFactors.amp_read` / `amp_write` hold them, at their padded widths), the facts pass's `fm`, a
`[BH, windows, width_S]` fp32 workspace and the `[BH heads, windows W]` int8 bytes out (the window grid whole), level S, the query heads a write head
and the reciprocal of the carry's factor. Three launches on the current stream; the workspace's contents after the call
are the windows' prefix masses, read by nothing else.

<a id="refusals"></a>

## The refusals

Each by name: a depth other than two; level S not 0 or 1; a level's rows not `[B, T, H, width]` u8 or f16 with digits
contiguous and every row and stride on 16 bytes (a lane's eight digits are one 8- or 16-byte load); a width not a
multiple of 8 digits, or level S's masses over 48 KB of shared memory (12,288 digits); a window over 1,024 tokens or not a
multiple of 32; more than 65,535 query batch-heads (the grid's y); `fm`, the workspace or the output not the contiguous
tensors above. `rola.ops.scale.serves` names the depth refusal before the launch.

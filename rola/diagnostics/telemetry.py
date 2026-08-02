"""RoLA sparsity/collapse telemetry (remaining-tasks-spec.md TASK #2).

A lightweight, opt-in, throttled JSONL sink called from inside the one compact route
descriptor producer. The producer passes the already-solved descriptor, so a disabled
or throttled call performs no descriptor unpack or leaf fold work.

Activation (either):
  * env  ROLA_TELEMETRY=/path/to/out.jsonl        (read once at import, module singleton)
  * code TELEMETRY.enable("/path/to/out.jsonl")
Throttle: fires every N calls (per sink), default 100, env ROLA_TELEMETRY_EVERY=N or
`enable(path, every=N)`.

What one JSON line holds (per firing, per producer call = per layer step):
  * per side (r/w), per level d: exact-zero fraction (THE entmax sparsity number),
    support-size histogram over 0..b + mean/p10/p50/p90, dual collapse entropies
    (entropy of the MEAN gate distribution AND mean PER-TOKEN entropy — both low = the
    dangerous premature-collapse quadrant; high mean-dist + low per-token = healthy
    specialization), top-branch share p_max.
  * reverse-tile support at every production Scan A/Scan B candidate BC, reduced per BH
    and across the batch without constructing or retaining a tile bitmap;
  * leaf level (folded ONLY when a firing happens - never on the hot path): per-BH/state
    write-mass dead bitsets for this window, cumulative `ever_written` tracking, the
    read-of-never-written mass fraction, and alive-leaf fraction.
    Cumulative tracking is keyed by an explicit `layer_idx` or the producer's route
    parameter identity; records with neither identity report it as unavailable.

Grad-path safety: the driver calls the producer with grad ENABLED — every reduction here
runs on `.detach()`-ed tensors under `torch.no_grad()`; nothing is retained in the graph
and the sink never mutates its inputs. Safe under `torch.no_grad()` / inference_mode too.

The leaf fold below is a local reimplementation of the canonical Kronecker-of-levels fold,
using bounded per-level radix index vectors derived from the descriptor metadata. It is
telemetry-only and never feeds the engine.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

__all__ = ['TELEMETRY', 'RolaTelemetry', 'record_route_descriptor']

_ENV_PATH = 'ROLA_TELEMETRY'
_ENV_EVERY = 'ROLA_TELEMETRY_EVERY'
_DEFAULT_EVERY = 100
# Production carry ownership is one 16-leaf cohort. Candidate reverse-scan
# geometry belonged to the removed checkpoint/backward engine and must not
# remain in forward evidence.
_FORWARD_COHORT_WIDTH = 16


def _unpack_descriptor_values(values, descriptor):
    """Unpack compact tile-major values into a tuple of per-level tensors."""
    tiles = values.shape[1]
    padded = values.detach().float().permute(0, 1, 3, 2).reshape(values.shape[0], tiles * 32, -1)
    return tuple(
        padded[:, :descriptor.token_count, start:end]
        for start, end in zip(descriptor.level_offsets, descriptor.level_offsets[1:])
    )


def _radix_indices(descriptor):
    """Derive one bounded state-index vector per level from canonical strides."""
    state_count = math.prod(descriptor.level_widths)
    state_ids = torch.arange(state_count, device=descriptor.read_values.device, dtype=torch.long)
    return tuple(
        (state_ids // stride) % width
        for stride, width in zip(descriptor.radix_strides, descriptor.level_widths)
    )


def _fold_state_block(levels, indices, state_lo, state_hi):
    """Fold one compact state block, producing at most ``[BH, L, BC]`` values."""
    folded = None
    for factors, index in zip(levels, indices):
        selected = factors.index_select(-1, index[state_lo:state_hi])
        folded = selected if folded is None else folded * selected
    return folded


def _fold_level_chunks(levels, indices, nc, chunk_size=16):
    """Yield compact leaf folds without constructing a ``[BH, L, nc]`` tensor."""
    for lo in range(0, nc, chunk_size):
        hi = min(lo + chunk_size, nc)
        yield lo, hi, _fold_state_block(levels, indices, lo, hi)


def _entropy(p, dim=-1, eps=1e-12):
    """Shannon entropy (nats) of a (possibly unnormalized, nonnegative) distribution along
    `dim`; renormalizes defensively (entmax/softmax levels already sum to 1)."""
    s = p.sum(dim=dim, keepdim=True).clamp_min(eps)
    q = p / s
    return -(q * (q.clamp_min(eps)).log()).sum(dim=dim)


def _quantiles(x):
    """(mean, p10, p50, p90) of a flattened float tensor, as python floats."""
    xf = x.reshape(-1).float()
    if xf.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    qs = torch.quantile(xf, torch.tensor([0.10, 0.50, 0.90], device=xf.device))
    return float(xf.mean()), float(qs[0]), float(qs[1]), float(qs[2])


class RolaTelemetry:
    """Opt-in throttled JSONL routing-telemetry sink. NO-OP when disabled: `record_descriptor`
    returns after a single `if not self._enabled: return` — zero tensor work, no file I/O."""

    def __init__(self, path=None, every=None):
        env_path = os.environ.get(_ENV_PATH)
        self._path = path if path is not None else (env_path if env_path else None)
        if every is None:
            every = int(os.environ.get(_ENV_EVERY, _DEFAULT_EVERY))
        self._every = max(1, int(every))
        self._enabled = self._path is not None
        self._calls = 0
        self._fired = 0
        # cumulative never-written tracking: (layer_key, nc, BH) -> CPU bool [BH,nc]
        self._ever_written = {}

    # -- lifecycle -------------------------------------------------------------------
    def enable(self, path, every=None):
        """Explicit activation (alternative to the ROLA_TELEMETRY env var)."""
        self._path = str(path)
        if every is not None:
            self._every = max(1, int(every))
        self._enabled = True
        return self

    def disable(self):
        self._enabled = False
        return self

    def reset(self):
        """Reset call counter + cumulative ever-written state (e.g. between runs)."""
        self._calls = 0
        self._fired = 0
        self._ever_written = {}
        return self

    @property
    def enabled(self):
        return self._enabled

    @property
    def every(self):
        return self._every

    @property
    def calls(self):
        return self._calls

    @property
    def fired(self):
        return self._fired

    # -- the one producer-side entry point ---------------------------------------------
    def record_descriptor(self, descriptor, layer_idx=None, step=None, union_levels=None,
                          read_logits=None, write_logits=None, _source=None):
        """Record one already-solved compact descriptor only when telemetry fires."""
        if not self._enabled:
            return
        self._calls += 1
        if (self._calls - 1) % self._every != 0:
            return
        self._fired += 1
        source_key = None if layer_idx is not None else self._source_key(_source)
        read_levels = _unpack_descriptor_values(descriptor.read_values, descriptor)
        write_levels = _unpack_descriptor_values(descriptor.write_values, descriptor)
        indices = _radix_indices(descriptor)
        rec = self._compute(
            descriptor, read_levels, write_levels, indices, layer_idx, step, source_key)
        if union_levels and read_logits is not None and write_logits is not None:
            rec['union'] = self._union_stats(
                read_levels, write_levels, read_logits.detach(), write_logits.detach(),
                union_levels, descriptor.level_widths)
        self._write(rec)

    # -- stats ---------------------------------------------------------------------------
    @staticmethod
    def _source_key(source):
        """Return stable-in-process metadata for a producer route-parameter tensor."""
        if source is None:
            return None
        return (
            source.device.type,
            source.device.index,
            str(source.dtype),
            tuple(source.shape),
            tuple(source.stride()),
            int(source.storage_offset()),
            int(source.untyped_storage().data_ptr()),
        )

    @torch.no_grad()
    def _compute(self, descriptor, read_levels, write_levels, indices, layer_idx, step, source_key):
        D = len(descriptor.level_widths)
        branches = list(descriptor.level_widths)
        nc = math.prod(branches)
        rec = {
            'ts': time.time(),
            'step': step,
            'layer_idx': layer_idx,
            'call': self._calls,
            'descriptor_version': descriptor.topology_stamp[0],
            'token_origin': descriptor.token_origin,
            'token_count': descriptor.token_count,
            'D': D, 'branch_counts': branches, 'nc': nc,
            'r': self._side_stats(read_levels, branches),
            'w': self._side_stats(write_levels, branches),
        }
        rec['tile'] = self._tile_stats(read_levels, write_levels, indices, nc)
        rec['leaf'] = self._leaf_stats(
            read_levels, write_levels, indices, layer_idx, nc=nc, source_key=source_key)
        return rec

    @staticmethod
    def _side_stats(levels, branches):
        """Per-level + aggregate stats for compact per-level factor tensors."""
        real = [f.float() for f in levels]
        D = len(real)
        nz = [fd != 0 for fd in real]                            # exact entmax zeros only
        zero_per_level = [float(1.0 - nd.float().mean()) for nd in nz]
        support = [nd.sum(-1) for nd in nz]                      # D x [BH,L]
        hist = []
        for d in range(D):
            hist.append(torch.bincount(support[d].reshape(-1), minlength=branches[d] + 1).tolist())
        support_all = torch.stack([x.float() for x in support], dim=-1)
        s_mean, s_p10, s_p50, s_p90 = _quantiles(support_all)
        # dual collapse entropies + p_max, per level
        ent_mean_dist, ent_per_token, p_max = [], [], []
        for d in range(D):
            fd = real[d]                                        # [BH, L, b_d]
            mean_dist = fd.mean(dim=tuple(range(fd.dim() - 1)))  # [b]
            ent_mean_dist.append(float(_entropy(mean_dist, dim=-1)))
            ent_per_token.append(float(_entropy(fd, dim=-1).mean()))
            p_max.append(float(fd.max(dim=-1).values.mean()))
        return {
            'zero_frac': float(1.0 - torch.cat(
                [nd.reshape(-1) for nd in nz]).float().mean()),
            'zero_frac_per_level': zero_per_level,
            'support_hist_per_level': hist,                     # [D][b+1] counts, sizes 0..b
            'support_mean': s_mean, 'support_p10': s_p10,
            'support_p50': s_p50, 'support_p90': s_p90,
            'entropy_mean_dist_per_level': ent_mean_dist,       # low here + low below = collapse
            'entropy_per_token_per_level': ent_per_token,       # low here alone = confident routing
            'p_max_per_level': p_max,
            'max_entropy_per_level': [math.log(x) for x in branches],
        }

    @staticmethod
    @torch.no_grad()
    def _union_stats(read_levels, write_levels, read_logits, write_logits,
                     union_levels, branches, near_half=0.05):
        """Report Union split scales using solved values and producer-owned logits."""
        out = {}
        for level in union_levels:
            width = int(branches[level])
            zr = read_logits[..., level, :width].float()
            zw = write_logits[..., level, :width].float()
            support = (read_levels[level] != 0) | (write_levels[level] != 0)
            sigma = torch.sigmoid(zr - zw)[support]
            mean, p10, p50, p90 = _quantiles(sigma)
            support_size = support.sum(-1)
            out[str(level)] = {
                'branch_count': width,
                'logit_scale_r': float(zr.std(unbiased=False)),
                'logit_scale_w': float(zw.std(unbiased=False)),
                'support_hist': torch.bincount(
                    support_size.reshape(-1), minlength=width + 1).tolist(),
                'support_mean': float(support_size.float().mean()),
                'sigma_mean': mean, 'sigma_p10': p10,
                'sigma_p50': p50, 'sigma_p90': p90,
                'sigma_near_half': float(((sigma - 0.5).abs() < near_half).float().mean())
                                   if sigma.numel() else 1.0,
            }
        return out

    @staticmethod
    def _tile_stats(read_levels, write_levels, indices, nc, token_tile=32):
        """Count exact forward-cohort support for the production state plane.

        State cohorts are folded one at a time, so active storage is bounded by
        ``[BH, ceil(L/32)*32, 16]`` plus compact counters. No tile plane is
        retained, and the counters cross to the host once after reduction.
        """
        bh, length = read_levels[0].shape[:2]
        token_tiles = (length + token_tile - 1) // token_tile
        padded_length = token_tiles * token_tile
        counts = torch.zeros(bh, 3, device=read_levels[0].device, dtype=torch.int64)

        for state_lo in range(0, nc, _FORWARD_COHORT_WIDTH):
            state_hi = min(state_lo + _FORWARD_COHORT_WIDTH, nc)
            read = _fold_state_block(read_levels, indices, state_lo, state_hi)
            write = _fold_state_block(write_levels, indices, state_lo, state_hi)
            if padded_length != length:
                read = F.pad(read, (0, 0, 0, padded_length - length))
                write = F.pad(write, (0, 0, 0, padded_length - length))
            width = state_hi - state_lo
            read_live = read.view(bh, token_tiles, token_tile, width).ne(0).any(dim=(2, 3))
            write_live = write.view(bh, token_tiles, token_tile, width).ne(0).any(dim=(2, 3))
            read_dead = ~read_live
            write_dead = ~write_live
            counts.add_(torch.stack(
                (read_dead, write_dead, read_dead & write_dead), dim=-1
            ).to(torch.int64).sum(dim=1))

        # This is the tile reducer's one post-reduction device-to-host boundary; there is no
        # per-tile sync.
        counts = counts.cpu()

        def count_record(values, total):
            read_dead, write_dead, both_dead = (int(value) for value in values)
            denominator = total if total else 1
            return {
                'count': total,
                'read_dead_count': read_dead,
                'write_dead_count': write_dead,
                'both_dead_count': both_dead,
                'read_dead_frac': read_dead / denominator if total else 0.0,
                'write_dead_frac': write_dead / denominator if total else 0.0,
                'both_dead_frac': both_dead / denominator if total else 0.0,
            }

        tiles_per_bh = token_tiles * ((nc + _FORWARD_COHORT_WIDTH - 1) // _FORWARD_COHORT_WIDTH)
        per_bh = [
            {'bh': index, **count_record(counts[index], tiles_per_bh)}
            for index in range(bh)
        ]
        return {
            'scope': 'logical production forward-cohort geometry',
            'token_tile': token_tile,
            'cohort_width': _FORWARD_COHORT_WIDTH,
            'tile_count_per_bh': tiles_per_bh,
            'fold_storage': 'bounded [BH,ceil(L/32)*32,16]; no retained tile plane',
            'per_bh': per_bh,
            'batch': count_record(counts.sum(dim=0), bh * tiles_per_bh),
        }

    def _leaf_stats(self, read_levels, write_levels, indices, layer_idx, nc=None, source_key=None):
        """Per-BH leaf support and cumulative never-written telemetry.

        Chunk folds are reduced entirely on-device. Persistent active storage is
        ``O(BH*nc)`` mass/counter state, never ``O(BH*L*nc)``.
        """
        nc = math.prod(factors.shape[-1] for factors in write_levels) if nc is None else nc
        bh, length = read_levels[0].shape[:2]
        write_mass = torch.zeros(bh, nc, device=read_levels[0].device, dtype=torch.float32)
        read_mass = torch.zeros_like(write_mass)
        active_r = torch.zeros(bh, device=read_levels[0].device, dtype=torch.int64)
        active_w = torch.zeros_like(active_r)
        for lo, hi, gr in _fold_level_chunks(read_levels, indices, nc):
            read_mass[:, lo:hi] = gr.sum(dim=1)
            active_r.add_(gr.ne(0).sum(dim=(1, 2)))
        for lo, hi, gw in _fold_level_chunks(write_levels, indices, nc):
            write_mass[:, lo:hi] = gw.sum(dim=1)
            active_w.add_(gw.ne(0).sum(dim=(1, 2)))

        # Pack all leaf reductions before this reducer's one post-reduction host boundary.
        # This keeps chunk traversal asynchronous and avoids a separate transfer for each
        # accumulator. A full record separately transfers the tile reducer's counters.
        leaf_summary = torch.cat(
            (read_mass, write_mass,
             active_r.to(read_mass.dtype).unsqueeze(-1),
             active_w.to(read_mass.dtype).unsqueeze(-1)), dim=-1).cpu()
        read_mass = leaf_summary[:, :nc]
        write_mass = leaf_summary[:, nc:2 * nc]
        active = leaf_summary[:, 2 * nc:].to(torch.int64)
        written_now = write_mass.ne(0)
        dead_now = ~written_now
        if layer_idx is not None:
            scope = 'layer_idx'
            key = ('layer_idx', layer_idx, nc, bh)
        elif source_key is not None:
            scope = 'route_parameter'
            key = ('route_parameter', source_key, nc, bh)
        else:
            scope = 'unavailable'
            key = None

        if key is None:
            never = None
        else:
            ever = self._ever_written.get(key)
            if ever is None:
                ever = torch.zeros(bh, nc, dtype=torch.bool)
            ever = ever | written_now
            self._ever_written[key] = ever
            never = ~ever

        def _hex(bits):
            value = 0
            for index in torch.nonzero(bits, as_tuple=False).reshape(-1).tolist():
                value |= 1 << index
            return format(value, 'x')

        population_per_bh = length * nc
        per_bh = []
        for index in range(bh):
            total_read = float(read_mass[index].sum())
            read_never_mass = (float(read_mass[index][never[index]].sum())
                               if never is not None else None)
            per_bh.append({
                'bh': index,
                'leaf_active_frac_r': (int(active[index, 0]) / population_per_bh
                                       if population_per_bh else 0.0),
                'leaf_active_frac_w': (int(active[index, 1]) / population_per_bh
                                       if population_per_bh else 0.0),
                'dead_write_count': int(dead_now[index].sum()),
                'dead_write_frac': (float(dead_now[index].float().mean()) if nc else 0.0),
                'dead_write_bitset_hex': _hex(dead_now[index]),
                'ever_written_scope': scope,
                'never_written_count': (int(never[index].sum()) if never is not None else None),
                'never_written_bitset_hex': (_hex(never[index]) if never is not None else None),
                'read_never_written_frac': (
                    read_never_mass / total_read
                    if read_never_mass is not None and total_read > 0 else None),
            })

        population = bh * population_per_bh
        total_read = float(read_mass.sum())
        read_never_mass = (float(read_mass[never].sum()) if never is not None else None)
        return {
            'nc': nc,
            'bh': bh,
            'population_per_bh': population_per_bh,
            'ever_written_scope': scope,
            'per_bh': per_bh,
            'batch': {
                'leaf_active_frac_r': int(active[:, 0].sum()) / population if population else 0.0,
                'leaf_active_frac_w': int(active[:, 1].sum()) / population if population else 0.0,
                'dead_write_count': int(dead_now.sum()),
                'dead_write_frac': float(dead_now.float().mean()) if bh * nc else 0.0,
                'never_written_count': (int(never.sum()) if never is not None else None),
                'never_written_frac': (
                    float(never.float().mean()) if never is not None and bh * nc else None),
                'read_never_written_frac': (
                    read_never_mass / total_read
                    if read_never_mass is not None and total_read > 0 else None),
            },
        }

    # -- sink ----------------------------------------------------------------------------
    def _write(self, rec):
        try:
            d = os.path.dirname(self._path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self._path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
        except OSError:
            # telemetry must never take down training; drop the record, disable the sink
            self._enabled = False


# Module singleton — the producer calls the descriptor-aware entry point below so that the
# disabled path is one attribute check and one function call.
TELEMETRY = RolaTelemetry()


def record_route_descriptor(descriptor, layer_idx=None, step=None, union_levels=None,
                            read_logits=None, write_logits=None, _source=None):
    """Producer-side convenience: forwards to the module singleton."""
    TELEMETRY.record_descriptor(
        descriptor, layer_idx=layer_idx, step=step, union_levels=union_levels,
        read_logits=read_logits, write_logits=write_logits, _source=_source)

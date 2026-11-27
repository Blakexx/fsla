# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTERNAL FORMAT: scaled blocks in their storage forms, and the straight-through view.

A SCALED BLOCK is unit-max rows in a storage FORM and each row's max as its fp32 scale (docs/CONSTITUTION.md#r23): a
routing level's rows on either side, a token's values -- one type, :class:`ScaledBlock`. A :class:`Form` states the
storage type and whether an entry may be negative, and quantizing into it and dequantizing out of it are written once, on it. The two forms: ``ROWS`` (u8,
the row max at 255, a zero a zero -- every routing level, dense or sparse) and ``VALUES`` (bf16 in ``[-1, 1]`` -- the
values). A routing form is nonnegative, because the den, the liveness words and the decay's shares need it, and
:meth:`Form.quantize` refuses an entry its form does not hold. ``straight_through`` hands a consumer the DEQUANTIZED
values with an identity gradient, which is how the format trains (docs/api.md section 2.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "AMP_LEVELS",
    "ROWS",
    "ROW_ALIGN",
    "ROW_DTYPE",
    "VALUES",
    "Form",
    "ScaledBlock",
    "aligned_rows",
    "refuse_unheld",
    "row_slot",
    "straight_through",
]

#: u8 coefficient levels: the row max at 255.
AMP_LEVELS = 255.0
#: the stored coefficient form: u8 at every depth.
ROW_DTYPE = torch.uint8
#: THE ROW ALIGNMENT LAW: every level's rows start on, and step batch, token and head by, multiples of this many bytes
#: (docs/internals/facts/liveness_contract.md#alignment).
ROW_ALIGN = 32
_TINY = 1e-30


def row_slot(width: int) -> int:
    """A level's bytes in a packed row: its width in bytes rounded up to the alignment law's grain."""
    return -(-width // ROW_ALIGN) * ROW_ALIGN


def aligned_rows(B: int, T: int, H: int, width: int, device, dtype: torch.dtype = ROW_DTYPE) -> torch.Tensor:
    """``[B, T, H, width]`` zeros of ``dtype`` under the alignment law: a view of storage whose rows are
    ``row_slot`` of their bytes."""
    size = torch.empty((), dtype=dtype).element_size()
    return torch.zeros((B, T, H, row_slot(width * size) // size), dtype=dtype, device=device)[..., :width]


@dataclass(frozen=True, slots=True)
class Form:
    """A SCALED BLOCK'S STORAGE FORM: the stored type, whether an entry may be negative, the stored value of a row's max
    (``top``) and whether its rows lie under the row alignment law (the routing form: a kernel reads them in
    32-byte grains)."""

    name: str
    dtype: torch.dtype
    signed: bool
    top: float
    aligned: bool

    def unheld(self, x: torch.Tensor) -> torch.Tensor:
        """Whether ``x`` has an entry this form does not hold -- a negative one in a nonnegative form, a non-finite one
        in any -- as a 0-dim bool on ``x``'s device, read by no one yet (:func:`refuse_unheld` reads it)."""
        unfinite = ~torch.isfinite(x).all()
        return unfinite if self.signed else unfinite | (x < 0).any()

    def quantize(self, x: torch.Tensor, *, checked: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rows [..., w], scale fp32 [...])``: each row against its own max, rounded once into the form (u8 to the
        nearest digit, bf16 to the nearest value); an all-zero row stays zero. Refuses
        an entry the form does not hold (:func:`refuse_unheld`), unless the caller ``checked`` the block already."""
        exact = x.detach().float()
        if not checked:
            refuse_unheld([(self, exact)])
        scale = exact.abs().amax(dim=-1)
        unit = exact / scale.clamp_min(_TINY).unsqueeze(-1)
        stored = torch.round(unit * self.top) if self.dtype is ROW_DTYPE else unit * self.top
        if not self.aligned:
            return stored.to(self.dtype), scale
        rows = aligned_rows(*exact.shape, exact.device, self.dtype)
        rows.copy_(stored.to(self.dtype))
        return rows, scale

    def dequantize(self, rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """The block's values, fp32: each stored entry over ``top`` times its row's scale."""
        return rows.float() / self.top * scale.unsqueeze(-1)


def refuse_unheld(blocks) -> None:
    """Refuse the first ``(form, x)`` of ``blocks`` with an entry its form does not hold: ONE host read for them all, so
    a factory over many blocks waits on the device once."""
    blocks = list(blocks)
    flags = [form.unheld(x.detach().float()) for form, x in blocks]
    if flags and bool(torch.stack([f.to(flags[0].device) for f in flags]).any()):
        form, _ = next(block for block, flag in zip(blocks, flags) if bool(flag))
        raise ValueError(f"the {form.name} form holds {'finite' if form.signed else 'nonnegative, finite'} entries; "
                         "this block has one it does not")


#: a routing level's rows, dense or sparse: u8, the row max at 255, a zero a zero
ROWS = Form("u8 rows", ROW_DTYPE, signed=False, top=AMP_LEVELS, aligned=True)
#: the values: bf16 in [-1, 1] against the token's largest channel
VALUES = Form("bf16 values", torch.bfloat16, signed=True, top=1.0, aligned=False)


def straight_through(exact: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
    """The quantized VALUES, fp32, with ``exact``'s gradient: the rounding is stepped through as identity."""
    wide = exact.float()
    return wide + (dequantized.float() - wide).detach()


def _aligned(rows: torch.Tensor) -> bool:
    """THE ROW ALIGNMENT LAW on ``[B, T, H, width]`` rows: the first row on the grain, and every stride of a dimension
    longer than one a multiple of it, in bytes."""
    size = rows.element_size()
    return rows.data_ptr() % ROW_ALIGN == 0 and all(
        rows.stride(d) * size % ROW_ALIGN == 0 for d in range(3) if rows.shape[d] > 1)


def _described(t) -> str:
    return f"{getattr(t, 'dtype', type(t).__name__)} {tuple(getattr(t, 'shape', ()))}"


@dataclass(frozen=True, slots=True)
class ScaledBlock:
    """ONE SCALED BLOCK: ``data`` ``[B, T, H, width]`` unit-max rows in ``form``'s storage type, ``scale`` fp32
    ``[B, T, H]`` each row's max, and ``source`` the floating original, held ONLY when it requires a gradient (the
    straight-through anchor of :meth:`dequantized`), ``None`` otherwise. A routing level's rows on either side (``ROWS``)
    and a token's values (``VALUES``) are each one. ``__post_init__`` checks the structure -- dtype, shapes, one device,
    the alignment law where the form states it -- and never a value: :meth:`quantize` is what refuses an entry the form
    does not hold."""

    data: torch.Tensor
    scale: torch.Tensor
    form: Form
    source: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if type(self.form) is not Form:
            raise TypeError(f"ScaledBlock.form is a fsla.routing.quant.Form (ROWS, VALUES), got {self.form!r}")
        data = self.data
        if not isinstance(data, torch.Tensor) or data.ndim != 4 or data.dtype is not self.form.dtype:
            raise ValueError(f"ScaledBlock.data is the {self.form.name} form's {self.form.dtype} [B, T, H, width] rows, "
                             f"got {_described(data)}")
        if (not isinstance(self.scale, torch.Tensor) or self.scale.dtype is not torch.float32
                or tuple(self.scale.shape) != tuple(data.shape[:3])):
            raise ValueError(f"ScaledBlock.scale is the rows' maxima, fp32 [B, T, H]={tuple(data.shape[:3])}, "
                             f"got {_described(self.scale)}")
        if self.source is not None and tuple(self.source.shape) != tuple(data.shape):
            raise ValueError(f"ScaledBlock.source is the rows' floating original, {tuple(data.shape)}, "
                             f"got {_described(self.source)}")
        for name in ("scale", "source"):
            t = getattr(self, name)
            if t is not None and t.device != data.device:
                raise ValueError(f"ScaledBlock.{name} is on {t.device} but the rows are on {data.device}")
        if self.form.aligned and not _aligned(data):
            raise ValueError(f"the {self.form.name} form's rows start on and step by multiples of {ROW_ALIGN} bytes "
                             f"(strides {tuple(data.stride())}, address {data.data_ptr() % ROW_ALIGN} past the grain; "
                             "docs/internals/facts/liveness_contract.md#alignment)")

    @classmethod
    def quantize(cls, x: torch.Tensor, form: Form, *, checked: bool = False) -> ScaledBlock:
        """A floating ``[B, T, H, width]`` in, its block in ``form`` out (:meth:`Form.quantize`), the original held as
        ``source`` where it requires a gradient. ``checked`` as :meth:`Form.quantize` takes it."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or not x.dtype.is_floating_point:
            raise ValueError(f"a scaled block quantizes a floating [B, T, H, width] tensor, got {_described(x)}")
        data, scale = form.quantize(x, checked=checked)
        source = x if torch.is_grad_enabled() and x.requires_grad else None
        return cls(data=data, scale=scale, form=form, source=source)

    def dequantized(self) -> torch.Tensor:
        """The block as fp32, ``data / top x scale``, computed at the call; with a ``source`` the original's gradient
        steps straight through the rounding. A consumer still computing on floats calls this at its seam."""
        exact = self.form.dequantize(self.data, self.scale)
        return exact if self.source is None else straight_through(self.source, exact)

    @property
    def batch(self) -> int:
        return self.data.shape[0]

    @property
    def tokens(self) -> int:
        return self.data.shape[1]

    @property
    def heads(self) -> int:
        return self.data.shape[2]

    @property
    def width(self) -> int:
        return self.data.shape[3]

    @property
    def device(self) -> torch.device:
        return self.data.device

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the block holds, for the gradient-reachability predicate."""
        return tuple(t for t in (self.data, self.scale, self.source) if t is not None)

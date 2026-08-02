import inspect
import pathlib
import sys
from unittest.mock import patch

import pytest
import torch

# `benchmarks/` on the path, so the bench PACKAGES import by their own names -- the
# repository's existing convention (`benchmarks/bench_regression.py` inserts the same
# directory and imports `bench.cells`). It is needed here because
# `THUNDER_RUN_SPEC.md` §6 P4 relocated the planner's exact census, its cost model and
# its `(BT, compact)` argmin to `benchmarks/planner_calibration/`, and the gates that
# bind those constants necessarily still call them. The runtime does not: `rola`
# imports nothing from here, which is what the relocation exists to achieve.
_BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks"
if _BENCHMARKS.is_dir() and str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

try:
    from torch.compiler import is_compiling
except ImportError:
    def is_compiling():
        return False

# The device's torch submodule (`torch.cuda`), resolved without depending on the
# fork's `fla.utils`: `rola` supports NVIDIA CUDA only, and the ratification gate
# is what decides which CUDA devices at that.
device_torch_lib = torch.cuda

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_ORIGINAL_EMPTY = torch.empty
_ORIGINAL_EMPTY_LIKE = torch.empty_like
_ORIGINAL_NEW_EMPTY = torch.Tensor.new_empty


def _is_called_from_fla():
    """Check if the call is from the `rola` package."""
    frame = inspect.currentframe()
    try:
        # Skip the current frame and go up the call stack
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            if hasattr(frame, 'f_code') and hasattr(frame.f_code, 'co_filename'):
                filename = frame.f_code.co_filename
                # Skip conftest.py frames (where the guarded functions are defined)
                if 'conftest.py' in filename:
                    continue
                # Check if this frame is from a test file
                # Look for 'tests/' or 'test_' in the file path
                if 'tests/' in filename or 'test_' in filename:
                    return False

            module = inspect.getmodule(frame)
            if module and hasattr(module, '__name__'):
                # If call is from the `rola` package, apply guard
                if module.__name__.split('.')[0] == 'rola':
                    return True
    finally:
        del frame
    # Default to not guarding if we can't determine
    return False


def _poison(result):
    """Fill a scratch tensor with NaN. Skip requires_grad leaves: inductor's lazy init (e.g. pad_mm)
    allocates grad-leaf scratch via torch.empty, and an in-place fill_ on a leaf-that-requires-grad
    raises — so with torch.compile defaulted on, poisoning those would break compilation, not catch a
    real uninitialized-read bug. (Those tensors are inductor's, not fla buffers.)"""
    if result.requires_grad:
        return result
    if result.is_floating_point():
        result.fill_(float('nan'))
    elif result.is_complex():
        result.fill_(complex(float('nan'), float('nan')))
    return result


def _guarded_empty(*args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    dtype = kwargs.get('dtype') or torch.get_default_dtype()

    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY(*args, **kwargs)

    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY(*args, **kwargs)

    return _poison(_ORIGINAL_EMPTY(*args, **kwargs))


def _guarded_empty_like(input, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = input.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    return _poison(_ORIGINAL_EMPTY_LIKE(input, **kwargs))


def _guarded_new_empty(self, *args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = self.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    return _poison(_ORIGINAL_NEW_EMPTY(self, *args, **kwargs))


@pytest.fixture(scope="function", autouse=True)
def poison_torch_memory(request):
    # Only apply the guard to ops and modules tests
    path = str(request.node.fspath)
    if 'tests/ops/' not in path and 'tests/modules/' not in path:
        yield
        return

    with patch('torch.empty', new=_guarded_empty), \
            patch('torch.empty_like', new=_guarded_empty_like), \
            patch('torch.Tensor.new_empty', new=_guarded_new_empty):
        yield
        if hasattr(device_torch_lib, 'synchronize'):
            device_torch_lib.synchronize()

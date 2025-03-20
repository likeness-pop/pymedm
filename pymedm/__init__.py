import contextlib
from importlib.metadata import PackageNotFoundError, version

from .batch import batch_solve
from .diagnostics import moe_fit_rate
from .pmedm import (
    PMEDM,
    compute_allocation,
    compute_hessian_matrix,
    processor_access,
    processor_availability,
    simulate_allocation_matrix,
)
from .puma import puma

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("pymedm")

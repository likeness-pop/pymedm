import numpy as np
import pandas as pd
from jax.lib import xla_bridge
from multiprocess import cpu_count, pool

from .pmedm import PMEDM

available_cpus = cpu_count() - 1
cpu = xla_bridge.get_backend().platform == "cpu"


def batch_solve(
    mpu, include_cg0=True, n_reps=0, n_procs=available_cpus, build_only=False
):
    """Solves multiple P-MEDM problems with parallelization.

    Parameters
    ----------
    mpu : dict
        A dictionary representing PUMAs in the area of interest,
        with keys ``fips`` and values ``livelike.acs.puma``.
    include_cg0 : bool = True
        Whether to include Level 0 (PUMA) constraints.
    n_reps : int = 0
        Number of allocation matrix replicates to generate.
    n_procs : int
        The number of processes to use for parallel processing
        (defaults to the number of CPUs - 1).
    build_only : bool = True
        Only build the ``PMEDM`` instances, do not solve them. This is used for
        testing purposes only. See GL#30.

    Returns
    -------
    pmds : dict
        A dictionary with keys ``fips`` and values ``pymedm.PMEDM``.
    """

    pumas = list(mpu.keys())

    pmd_workers = pool.ThreadPool(n_procs)
    pmds = pmd_workers.map(
        lambda p: PMEDM(
            serial=mpu[p].est_ind.index,
            year=mpu[p].year,
            wt=mpu[p].wt,
            cind=mpu[p].est_ind,
            cg1=mpu[p].est_g1,
            cg2=mpu[p].est_g2,
            sg1=mpu[p].se_g1,
            sg2=mpu[p].se_g2,
            include_cg0=include_cg0,
            topo=mpu[p].topo,
            n_reps=n_reps,
            random_state=int(p),
        ),
        pumas,
    )
    pmd_workers.close()

    if not build_only:
        if cpu:
            solvers = pool.ThreadPool(n_procs)
            solvers.map(lambda p: p.solve(), pmds)
            solvers.close()

        else:
            [p.solve() for p in pmds]

    pmds = dict(zip(mpu.keys(), pmds, strict=True))

    return pmds

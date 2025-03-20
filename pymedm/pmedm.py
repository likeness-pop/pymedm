import os
import sys
import time

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import pandas as pd
from jax import jit
from jax.experimental import sparse
from scipy import linalg
from scipy import sparse as sps

jax.config.update("jax_enable_x64", True)

python_path = sys.executable.split("/bin/python")[0]
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={python_path}"


class PMEDM:
    """P-MEDM class."""

    def __init__(
        self,
        year,
        serial,
        wt,
        cind,
        cg1,
        cg2,
        sg1,
        sg2,
        include_cg0=True,
        q=None,
        lam=None,
        topo=None,
        verbose=False,
        tol=1e-6,
        keep_solver=True,
        allocation_matrix=True,
        n_reps=0,
        random_state=1,
    ):
        """

        Parameters
        ----------
        year : int
            ACS vintage year.
        serial : numpy.ndarray
            Response IDs.
        wt : numpy.ndarray
            Sample weights.
        cind : pandas DataFrame
            Individual-level constraints.
        cg1 : pandas.DataFrame
            Aggregate-level constraints.
        cg2 : pandas.DataFrame
            Target-level constraints.
        sg1 : pandas.DataFrame
            Aggregate-level standard errors.
        sg2 : pandas.DataFrame
            Target-level standard errors.
        include_cg0 : bool = True
            Whether to include Level 0 (PUMA) constraints.
        q : None | numpy.ndarray = None
            A vector of probabilities (individual x location).
        lam : None | numpy.ndarray = None
            A vector of parameters.
        topo : None | pandas.DataFrame = None
            Target-aggregate zone topology (crosswalk).
        verbose : bool = False
            Flag for printing solver output.
        tol : float = 1e-6
            Tolerance level for solver termination.
        keep_solver : bool
            Flag for appending solver results.
        allocation_matrix : bool = True
            Flag for computing allocation matrix when P-MEDM solves.
        n_reps : int = 0
            Number of allocation matrix replicates to generate. Setting ``n_reps>=1``
            triggers the replicate generation workflow, which runs
            ``compute_hessian_matrix()``. The Hessian Matrix functionality within
            that function is highly experimental and not mature. Use with caution.
            Results are not guaranteed.
        random_state : int = 1
            Random state for generating allocation matrix replicates.

        Attributes
        ----------
        serial : numpy.ndarray
            Response IDs.
        wt : numpy.ndarray
            Sample weights.
        N : int | jaxlib.xla_extension.Array
            Total population.
        n : int
            Total responses.
        Y1 : numpy.ndarray
            Level-1 (aggregate) constraints.
        Y2 : numpy.ndarray
            Level-2 (target) constraints.
        Y_vec : numpy.array | jaxlib.xla_extension.Array | array-like
            Vectorized constraints.
        V1 : numpy.ndarray
            Level-1 (aggregate) error variances.
        V2 : numpy.ndarray
            Level-2 (target) error variances.
        V_vec : numpy.array | jaxlib.xla_extension.Array | array-like
            Vectorized error variances.
        sV : jax.experimental.sparse.BCOO
            Diagonal matrix of error variances.
        topo : pandas.DataFrame
            Geographic crosswalk.
        n_topo : int
            Total units in ``topo``.
        A1 : numpy.array
            Geographic crosswalk.
        A2 : numpy.array
            Level-2 (target) identity matrix.
        X : jax.experimental.sparse.BCOO
            Solution space.
        q : numpy.array | jaxlib.xla_extension.Array | array-like
            Occurrence probabilities.
        lam : numpy.array | jaxlib.xla_extension.Array | array-like
            Parameters.
        """

        # ACS vintage year
        self.year = year

        # Response IDs
        self.serial = serial

        # Sample weights
        self.wt = wt

        # Total population
        self.N = np.sum(self.wt)

        # Total responses
        self.n = cind.shape[0]

        # Level 0 (PUMA) constraints
        if include_cg0:
            cg0 = pd.DataFrame(cg2.sum(axis=0).values[:, None].T, columns=cg2.columns)
            sg0 = sg2.apply(lambda x: np.sqrt(np.sum(np.square(x))), axis=0)

            # Tighten SEs to near-zero to preserve total pops
            # TODO: explore additional scaling factors
            sg0 *= 0.1

        # Geographic constraints
        self.Y0 = cg0.values.flatten("F") if include_cg0 else None
        self.Y1 = cg1.values.flatten("F")
        self.Y2 = cg2.values.flatten("F")
        # See GL#29 & GL!34 for info on the numpy vs. jax arrays/operations
        self.Y_vec = jnp.array([])
        if self.Y0 is not None:
            self.Y_vec = jnp.concatenate([self.Y_vec, self.Y0])
        self.Y_vec = jnp.concatenate([self.Y_vec, self.Y1, self.Y2])
        self.Y_vec = self.Y_vec / self.N

        # Geographic constraint error variances
        self.V0 = (
            np.square(sg0.astype("float").values).flatten("F") if include_cg0 else None
        )
        self.V1 = np.square(sg1.astype("float").values).flatten("F")
        self.V2 = np.square(sg2.astype("float").values).flatten("F")

        self.V_vec = jnp.array([])
        if self.V0 is not None:
            self.V_vec = jnp.concatenate([self.V_vec, self.V0])
        self.V_vec = jnp.concatenate([self.V_vec, self.V1, self.V2])
        self.V_vec = self.V_vec * (self.n / self.N**2)

        sV = np.diag(self.V_vec)
        self.sV_sps = sps.csr_matrix(sV)  # used to compute Hessian inv
        self.sV = sparse.BCOO.from_scipy_sparse(self.sV_sps)

        # Geographic topology
        if topo is None:
            _ix = cg2.index.values
            self.topo = pd.DataFrame({"g2": _ix, "g1": [str(i)[:-1] for i in _ix]})
        else:
            self.topo = topo
        self.n_topo = self.topo.shape[0]

        # Set up geographic crosswalk and level 2 identity matrix
        self.A0 = np.ones((self.n_topo,)).astype("int") if include_cg0 else None
        topo_g1 = self.topo.g1.values
        self.A1 = np.array([1 * (topo_g1 == G) for G in np.unique(topo_g1)])
        self.A2 = np.identity(self.n_topo).astype("int")

        # Solution space
        _cindT = cind.values.astype("float").transpose()
        X0 = sps.kron(_cindT, self.A0) if self.A0 is not None else None
        X1 = sps.kron(_cindT, self.A1)
        X2 = sps.kron(_cindT, self.A2)

        # initialize as scipy.sparse
        # to have handy for computing Hessian inv
        self.X_sps = sps.vstack([X1, X2])
        if X0 is not None:
            self.X_sps = sps.vstack([X0, self.X_sps])
        self.X_sps = self.X_sps.transpose()
        self.X = sparse.BCOO.from_scipy_sparse(self.X_sps)

        # Initial probabilities
        if q is None:
            self.q = jnp.repeat(wt, self.A1.shape[1], axis=0)
            self.q = self.q / np.sum(self.q)
        else:
            self.q = q

        # Initial parameters
        if lam is None:
            self.lam = jnp.zeros((len(self.Y_vec),))
        else:
            self.lam = lam

        # for storing solver results
        self.res = None

        # for storing allocation matrix
        self.almat = None

        # for storing allocation matrix replicates
        self.almat_reps = None

        # print output?
        self.verbose = verbose

        # gradient tolerance
        self.tol = tol

        # keep P-MEDM solver results?
        self.keep_solver = keep_solver

        # compute allocation matrix?
        self.allocation_matrix = allocation_matrix

        # number of allocation matrix replicates
        self.n_reps = n_reps
        self.random_state = random_state

    @staticmethod
    @jit
    def f(lam, **kwargs):
        """P-MEDM objective function."""
        q = kwargs["q"]
        X = kwargs["X"]
        Y_vec = kwargs["Y_vec"]
        sV = kwargs["sV"]

        qXl = compute_allocation(q, X, lam)

        sVl = sparse.bcoo_dot_general(
            sV, lam, dimension_numbers=(((1,), (0,)), ((), ()))
        )
        lvl = jnp.matmul(lam, sVl)

        return jnp.matmul(Y_vec, lam) + jnp.log(np.sum(qXl)) + (0.5 * lvl)

    def solve(self):
        # pass the other required variables as kwargs
        # so that only ``lam`` is updated
        solve_kws = dict(q=self.q, X=self.X, Y_vec=self.Y_vec, sV=self.sV)

        # setup problem and solve
        solver = jaxopt.LBFGS(fun=self.f, tol=self.tol, jit=True)

        # execute solver
        if self.verbose:
            print("Initializing P-MEDM solver...")
            solve_start_time = time.time()

        res = solver.run(init_params=self.lam, **solve_kws)

        if self.verbose:
            solve_elapsed_time = np.round(time.time() - solve_start_time, 4)
            print(f"P-MEDM completed in {solve_elapsed_time} seconds.")

        # update params
        self.lam = res.params

        # append results
        if self.keep_solver:
            self.res = res

        # compute allocation matrix
        if self.allocation_matrix:
            self.almat = compute_allocation(
                q=self.q,
                X=self.X,
                lam=self.lam,
                prob=True,
                counts=True,
                reshape=True,
                N=self.N,
                n_obs=self.n,
                n_geo=self.n_topo,
            )

        # simulate allocation matrix replicates
        if self.n_reps > 0:
            if self.verbose:
                print(
                    f"\nGenerating {self.n_reps} replicates of the "
                    "allocation matrix. The Hessian Matrix functionality here "
                    "is highly experimental and not mature. Use with caution.\n"
                )
                reps_start_time = time.time()

            self.almat_reps = simulate_allocation_matrix(
                self, nsim=self.n_reps, seed=self.random_state
            )

            if self.verbose:
                reps_elapsed_time = np.round(time.time() - reps_start_time, 4)
                print(f"Replicates generated in {reps_elapsed_time} seconds.")


def compute_allocation(
    q, X, lam, prob=False, counts=False, reshape=False, N=None, n_obs=None, n_geo=None
):
    """Compute P-MEDM allocations.

    Parameters
    ----------
    q : numpy.array | jaxlib.xla_extension.Array | array-like
        Occurrence probabilities.
    X : jax.experimental.sparse.BCOO
        Solution space.
    lam : numpy.array | jaxlib.xla_extension.Array | array-like
        A vector of parameters.
    prob : bool = True
        Calculate probability from raw counts.
    counts : bool = True
        Multiply probabilities by original total population counts.
    reshape : bool = False
        If ``True``, reshape ``al`` to ``(n_obs, n_geo)``.
    N : None | int | jaxlib.xla_extension.Array = None
        Total population.
    n_obs : None | int = None
        Total responses.
    n_geo : None | int = None
        Total units in ``PMEDM.topo`` -- the geographic crosswalk.

    Returns
    -------
    numpy.array, jaxlib.xla_extension.Array, array-like
        Calculated allocations.
    """

    Xl = sparse.bcoo_dot_general(-X, lam, dimension_numbers=(((1,), (0,)), ((), ())))
    qXl = q * jnp.exp(Xl)

    if prob:
        al = qXl / np.sum(qXl)

        if counts:
            if not N:
                raise ValueError(
                    "A value for ``N`` must be provided when ``counts==True``."
                )

            al = al * N

    else:
        al = qXl

    if reshape:
        if not n_obs or not n_geo:
            raise ValueError(
                "Values for both ``n_obs`` and ``n_geo`` must be provided"
                "when ``reshape==True``."
            )
        al = al.reshape(n_obs, n_geo)

    return al


def compute_hessian_matrix(pmd):
    """The Hessian Matrix functionality found here is highly experimental
    and not mature. Use with caution. Results are not guaranteed.

    Parameters
    ----------
    pmd : PMEDM
        A ``PMEDM`` object.

    Returns
    -------
    numpy.array | jaxlib.xla_extension.Array | array-like
        Computed Hessian matrix.
    """

    p = compute_allocation(pmd.q, pmd.X, pmd.lam, prob=True)

    # we need to use the scipy.sparse versions of ``X`` and
    # ``sV`` stored in our ``pmd`` here because the JAX
    # implementation results in an OOM error

    # first term
    h1a = pmd.X_sps.transpose() * p
    h1b = p * pmd.X_sps
    h1 = h1a[:, None] * h1b

    # second term
    pix = np.array(list(range(0, len(p))))
    dp = sps.coo_matrix((p, (pix, pix)), shape=(len(p), len(p)))
    h2 = (pmd.X_sps.transpose() * dp) * pmd.X_sps

    return -h1 + h2 + pmd.sV_sps


def simulate_allocation_matrix(pmd, nsim=1, seed=1):
    """The Hessian Matrix functionality found here is highly experimental
    and not mature. Use with caution. Results are not guaranteed.

    Parameters
    ----------
    pmd : PMEDM
        A ``PMEDM`` object.
    nsim : int = 1
        Number of allocation matrix replicates to generate.
        See also  the ``n_reps`` keyword argument in ``PMEDM.``
    seed : int = 1
        Random state for generating allocation matrix replicates.
        See also  the ``random_state`` keyword argument in ``PMEDM.``

    Returns
    -------
    list[numpy.array | jaxlib.xla_extension.Array | array-like]
        Simulated allocation matrices.
    """

    # Hessian
    H = compute_hessian_matrix(pmd)

    # inverse Hessian
    inv_H = linalg.inv(H)

    # parameter covariances
    cov_lam = inv_H / pmd.N

    # simulate paramaters
    rng = np.random.default_rng(seed=seed)
    sim_lam = rng.multivariate_normal(
        mean=pmd.lam, cov=cov_lam, size=nsim, tol=1e-3, method="cholesky"
    )

    # compute replicate allocation matrices
    sim_almats = []

    for lam_rep in sim_lam:
        _almat = compute_allocation(
            q=pmd.q,
            X=pmd.X,
            lam=lam_rep,
            prob=True,
            counts=True,
            reshape=True,
            N=pmd.N,
            n_obs=pmd.n,
            n_geo=pmd.n_topo,
        )

        sim_almats.append(_almat)

    return sim_almats


GPU_ERROR = "Unknown backend: 'gpu' requested"
TPU_ERROR = "Backend 'tpu' failed to initialize"


def processor_access(processor: str = "gpu") -> bool:
    """Convenience flagger to determine access to system GPUs, TPUs, etc.
    See gl:pymedm#82

    Parameters
    ----------
    processor : str = 'gpu'
        Processor type in question.

    Returns
    -------
    bool
        Flag indicating presence of & access to ``processor``.
    """

    try:
        ones = jax.numpy.ones(1)
        devices = jax.devices(processor)[0]
        _ = jax.device_put(ones, device=devices)
        return True
    except RuntimeError as e:
        if str(e).startswith((GPU_ERROR, TPU_ERROR)):
            return False
        else:
            raise e


def processor_availability(indent_print: None | int = None) -> pd.DataFrame:
    """Helper to determine processor availability through JAX.

    Parameters
    ----------
    indent_print : None | int = None
        Tab count to include for printing.

    Returns
    -------
    process_info : pandas.DataFrame
        Processor availability table.
    """

    process_info = pd.DataFrame(columns=["device", "available", "count"])
    for ix, processor in enumerate(["cpu", "gpu", "tpu"]):
        if processor_access(processor=processor):
            record = [processor, "yes", jax.device_count()]
        else:
            record = [processor, "no", 0]
        process_info.loc[ix] = record

    if isinstance(indent_print, int):
        indent = "\t" * indent_print
        print(indent + "Processors")
        df_pattern = "{:>8} | {:>10} | {:>6}"
        hbar = indent + ("-" * 31)
        print(hbar)
        print(indent + df_pattern.format(*process_info.columns))
        print(hbar)
        for _, r in process_info.iterrows():
            print(indent + df_pattern.format(r["device"], r["available"], r["count"]))
        print(hbar)

    return process_info

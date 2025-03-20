import numpy as np
import pandas as pd


def moe_fit_rate(cind, cg2, sg2, almat):
    """
    Simple validation based on the proportion of synthetic constraints
    fitting ACS 90% Margins of Error (the 'MOE fit rate') at the target
    geography.

    Parameters
    ----------
    cind : pandas.DataFrame
        Individual-level constraints.
    cg2 : pandas.DataFrame
        Target zone population constraints.
    sg2 : pandas.DataFrame
        Target zone population constraint standard errors.
    almat : numpy.array
        P-MEDM allocation matrix.

    Returns
    -------
    dict
        A dictionary containing a comparison of the published and
        synthetic estimates (``Ycomp``) and the proportion of synthetic
        estimates occurring within the published ACS 90% Margins
        of Error (MOEs) (``moe_fit_rate``).
    """

    # ACS block group constraints (published)
    Y = pd.melt(cg2, value_name="acs", ignore_index=False)

    # synthetic constraints
    Yhat = cind.apply(lambda x: np.sum(x.values[:, None] * np.array(almat), axis=0))

    # published 90% MOEs
    moe = pd.melt(sg2 * 1.645, value_name="moe", ignore_index=False)

    Yhat.index = cg2.index.values
    Yhat = pd.melt(Yhat, value_name="pmedm", ignore_index=False)

    # compare estimates
    Ycomp = pd.concat([Y, Yhat.pmedm], axis=1)
    Ycomp["err"] = np.abs(Ycomp.acs - Ycomp.pmedm)
    Ycomp["moe"] = moe.reset_index().moe.values
    Ycomp["in_moe"] = Ycomp.err < Ycomp.moe

    # proportion synthetic estimates within 90% ACS MOE
    moe_fit_rate = Ycomp.in_moe.sum() / Ycomp.shape[0]

    return {"Ycomp": Ycomp, "moe_fit_rate": moe_fit_rate}

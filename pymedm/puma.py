import numpy as np
import pandas as pd


class puma:
    """Simple PUMA representation.

    Parameters
    ----------
    serial : array-like, str
        Response IDs.
    year : int
        ACS vintage year.
    wt : array-like, int
        Sample weights.
    cind : pandas.DataFrame
        Individual-level constraints.
    cg1 : pandas.DataFrame
        Aggregate zone population constraints.
    cg2 : pandas.DataFrame
        Target zone population constraints.
    sg1 : pandas.DataFrame
        Aggregate zone population constraint standard errors.
    sg2 : pandas.DataFrame
        Target zone population constraint standard errors.
    """

    def __init__(self, serial, year, wt, cind, cg1, cg2, sg1, sg2):
        self.serial = serial
        self.year = year
        self.wt = wt
        self.est_ind = cind
        self.est_g1 = cg1
        self.est_g2 = cg2
        self.se_g1 = sg1
        self.se_g2 = sg2

        self.topo = pd.DataFrame(
            {"g2": cg2.index.values, "g1": [str(i)[:-1] for i in cg2.index.values]}
        )

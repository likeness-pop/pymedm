import warnings

import numpy
import pandas
import pytest

from pymedm import PMEDM, compute_allocation, simulate_allocation_matrix
from pymedm.diagnostics import moe_fit_rate

################################################################################
########################## Process input params ################################
################################################################################
gid = "GEOID"
pwt = "PERWT"
ser = "SERIAL"

# individual constraints ------------------------------------
cind = pandas.read_csv("data/toy_constraints_ind.csv")
# response IDs
serial = cind[ser].values
# sample weights
wt = cind[pwt].values
# drop response IDs & sample weights from indiv constraints
cind = cind.drop([ser, pwt], axis=1)

# geographic constraints ------------------------------------
cbg0 = pandas.read_csv("data/toy_constraints_bg.csv")
cbg0.index = cbg0[gid].values

# separate ests and standard errors
cols = cbg0.columns
is_se_col = [k.endswith("s") for k in cols]
se_cols = cols[numpy.where(is_se_col)]
is_est_col = [(k not in se_cols) & (k != gid) for k in cols]
est_cols = cols[numpy.where(is_est_col)]
cbg = cbg0[est_cols]
sbg = cbg0[se_cols]

# tract constraints -----------------------------------------
ctrt0 = cbg0.copy()
ctrt0.set_index(gid, inplace=True)
ctrt0.index = ctrt0.index.astype(str).str[0]

# separate ests and standard errors
ctrt = ctrt0[est_cols]
strt = ctrt0[se_cols]

# aggregate ests
ctrt = ctrt.groupby(ctrt.index).aggregate("sum")

# aggregate SE's
calc_std_err = lambda x: numpy.sqrt(numpy.sum(numpy.square(x)))
strt = strt.groupby(strt.index).aggregate(calc_std_err)


################################################################################
###################### Basic test from toy_example #############################
################################################################################


class TestSynthPMEDMBasicSetUp:
    def setup_method(self):
        # generate and solve a PMEDM instance
        self.pmd = PMEDM(
            year=2019,
            serial=serial,
            wt=wt,
            cind=cind,
            cg1=ctrt,
            cg2=cbg,
            sg1=strt,
            sg2=sbg,
            include_cg0=False,
            verbose=False,
        )

    def test_N(self):
        observed = self.pmd.N
        known = 100
        assert observed == known

    def test_n(self):
        observed = self.pmd.n
        known = 5
        assert observed == known

    def test_Y_vec(self):
        observed = self.pmd.Y_vec
        known = numpy.array(
            [
                0.48,
                0.52,
                0.28,
                0.37,
                0.38,
                0.37,
                0.27,
                0.38,
                0.05,
                0.05,
                0.33,
                0.47,
                0.13,
                0.17,
                0.28,
                0.2,
                0.3,
                0.22,
                0.15,
                0.13,
                0.18,
                0.19,
                0.25,
                0.13,
                0.22,
                0.15,
                0.15,
                0.12,
                0.2,
                0.18,
                0.03,
                0.02,
                0.03,
                0.02,
                0.16,
                0.17,
                0.27,
                0.2,
                0.05,
                0.08,
                0.07,
                0.1,
            ]
        )
        numpy.testing.assert_array_almost_equal(observed, known)

    def test_V_vec(self):
        observed = self.pmd.V_vec
        known = numpy.array(
            [
                0.004,
                0.004,
                0.00265625,
                0.0030625,
                0.0050625,
                0.0050625,
                0.0065,
                0.004,
                0.00025,
                0.00025,
                0.0075625,
                0.0075625,
                0.001,
                0.001,
                0.002,
                0.002,
                0.002,
                0.002,
                0.00153125,
                0.001125,
                0.00153125,
                0.00153125,
                0.00253125,
                0.00253125,
                0.00253125,
                0.00253125,
                0.0045,
                0.002,
                0.002,
                0.002,
                0.000125,
                0.000125,
                0.000125,
                0.000125,
                0.00378125,
                0.00378125,
                0.00378125,
                0.00378125,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
            ]
        )
        numpy.testing.assert_array_almost_equal(observed, known)

    def test_V_vec_eq_sV_data(self):
        numpy.testing.assert_array_equal(self.pmd.V_vec, self.pmd.sV.data)

    def test_topo(self):
        observed = self.pmd.topo
        known = pandas.DataFrame(dict(g2=[10, 11, 20, 21], g1=["1", "1", "2", "2"]))
        pandas.testing.assert_frame_equal(observed, known)

    def test_A1(self):
        observed = self.pmd.A1
        known = numpy.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        numpy.testing.assert_array_equal(observed, known)

    def test_A2(self):
        observed = self.pmd.A2
        known = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        numpy.testing.assert_array_equal(observed, known)

    def test_X(self):
        observed = self.pmd.X.data
        known = numpy.array(
            [
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )
        numpy.testing.assert_array_equal(observed, known)

    def test_q(self):
        observed = self.pmd.q
        known = numpy.array(
            [
                0.125,
                0.125,
                0.125,
                0.125,
                0.05,
                0.05,
                0.05,
                0.05,
                0.025,
                0.025,
                0.025,
                0.025,
                0.0125,
                0.0125,
                0.0125,
                0.0125,
                0.0375,
                0.0375,
                0.0375,
                0.0375,
            ]
        )
        numpy.testing.assert_array_equal(observed, known)

    def test_lam(self):
        observed = self.pmd.lam
        known = numpy.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        numpy.testing.assert_array_equal(observed, known)


class TestSynthPMEDMBasicSolve:
    def test_solve(self):
        # generate and solve a PMEDM instance
        pmd = PMEDM(
            year=2019,
            serial=serial,
            wt=wt,
            cind=cind,
            cg1=ctrt,
            cg2=cbg,
            sg1=strt,
            sg2=sbg,
            include_cg0=False,
            verbose=False,
            n_reps=1,
        )
        pmd.solve()

        # test objective value
        observed = pmd.res.state.value
        known = -2.0709795976648397
        assert observed == pytest.approx(known)

        # test resultant params
        observed = pmd.res.params
        known = numpy.array(
            [
                -1.418414684442366,
                1.418414684442362,
                0.6587797050032959,
                -1.5894223233357294,
                -4.0375871168081385,
                5.871322913292904,
                5.955664218302378,
                -7.30733134261312,
                3.3923924140201294,
                -4.188647080028564,
                -0.32978706039367156,
                0.6983678441556238,
                1.1094717646679515,
                -2.99748675354803,
                -4.077872745517934,
                1.2409877369180866,
                -8.244794596978371,
                11.081679605578175,
                2.6203968589276987,
                -2.0112579044210066,
                8.563354390749248,
                -11.742105529733927,
                -9.19622073084843,
                1.121036802133184,
                4.730788420662936,
                7.011842505215445,
                10.22871574769762,
                -3.6587958509426195,
                1.8327068515588565,
                -16.447294968213164,
                5.54355170330812,
                1.2421317351792995,
                -13.524245046033908,
                5.145810773463697,
                -1.7933220331959863,
                1.1337994225295605,
                -7.845844418591098,
                9.242568229199197,
                5.714280984192479,
                -3.4953252494075473,
                15.37468699956549,
                -21.370039969531152,
            ]
        )
        numpy.testing.assert_array_almost_equal(observed, known, decimal=3)

        # test MOE fit rate
        kwargs = dict(cind=cind, cg2=cbg, sg2=sbg, almat=pmd.almat)
        observed = moe_fit_rate(**kwargs)["moe_fit_rate"]
        known = 1.0
        assert observed == pytest.approx(known)

        # test replicate allocation matrix
        observed = pmd.almat_reps[0]
        known = numpy.array(
            [
                [6.94501275, 6.36350405, 14.56527004, 18.9929887],
                [5.44174333, 3.84979836, 6.01050686, 11.29194287],
                [1.12368488, 3.26265652, 4.73436497, 0.69702675],
                [2.30533822, 0.73172614, 0.15842991, 0.7619354],
                [7.87033544, 1.25808615, 3.55059396, 0.08505469],
            ]
        )
        numpy.testing.assert_array_almost_equal(observed, known, decimal=3)


class TestSynthPMEDMBasicSolveSupplement:
    def test_solve_with_params(self):
        # synthetic topology
        topo = pandas.DataFrame(dict(g2=[10, 11, 20, 21], g1=["1", "1", "2", "2"]))

        # synthetic predetermined probabilities
        numpy.random.seed(1982)
        q = numpy.random.uniform(low=0, high=20, size=20)
        q = q / q.sum()
        # -------- values of q --------
        # array([0.07710486, 0.11018122, 0.04908866, 0.05371031, 0.03715731,
        #       0.00512023, 0.01071222, 0.0511158 , 0.1077138 , 0.01584758,
        #       0.00906746, 0.06804444, 0.05258866, 0.05621154, 0.04100186,
        #       0.01836935, 0.04218958, 0.08200221, 0.04195901, 0.07081388])

        # synthetic predetermined parameters
        numpy.random.seed(2001)
        lam = numpy.random.uniform(low=-10, high=10, size=42)
        # -------- values of lam --------
        # array([-6.65748714, -8.38003926,  9.2045148 ,  8.76900488,  2.41717708,
        #       -1.48796065,  3.7956258 , -5.15930757, -2.6679873 , -7.16097946,
        #       -5.7961552 ,  8.15889034,  2.88101433, -5.85455585, -1.28068735,
        #        9.12714175,  3.9557546 ,  0.59784864,  1.62201477, -3.20913444,
        #        6.34991783, -8.36797361,  3.0843141 ,  2.89048547,  8.08506144,
        #        6.49940518, -4.72534732, -1.36896954, -0.02055179, -6.99398474,
        #       -1.87152906, -8.86251785,  2.87026949,  3.15444909, -8.6786926 ,
        #        2.9071884 , -7.57589112,  0.0588423 , -2.16059708, -7.4612199 ,
        #       -8.5646566 , -3.3232808 ])

        # generate and solve a PMEDM instance
        pmd = PMEDM(
            year=2019,
            serial=serial,
            wt=wt,
            cind=cind,
            cg1=ctrt,
            cg2=cbg,
            sg1=strt,
            sg2=sbg,
            include_cg0=False,
            topo=topo,
            q=q,
            lam=lam,
            verbose=False,
        )
        pmd.solve()

        # test objective value
        observed = pmd.res.state.value
        known = -2.50151283
        assert observed == pytest.approx(known)

        # test resultant params
        observed = pmd.res.params
        known = numpy.array(
            [
                -1.35597485,
                1.35598803,
                0.7389849,
                -1.79731003,
                -4.25459788,
                5.72079,
                5.74717729,
                -7.54340623,
                4.41748458,
                -3.4600354,
                -0.76487256,
                0.3583358,
                0.28810132,
                -3.52664037,
                -4.03525174,
                1.32325473,
                -8.29384304,
                11.00581875,
                2.7085312,
                -1.94179667,
                8.54977104,
                -12.14434635,
                -9.60612479,
                1.09689663,
                4.51409056,
                6.92746783,
                9.96529846,
                -3.74358747,
                1.5352176,
                -16.62203591,
                6.72457548,
                2.10938581,
                -13.15243224,
                6.2338906,
                -2.11437188,
                0.5846161,
                -8.31052428,
                9.02717697,
                5.78405598,
                -5.20740728,
                14.18156325,
                -21.23503356,
            ]
        )

        # test MOE fit rate
        kwargs = dict(cind=cind, cg2=cbg, sg2=sbg, almat=pmd.almat)
        observed = moe_fit_rate(**kwargs)["moe_fit_rate"]
        known = 0.9642857142857143
        assert observed == pytest.approx(known)

    def test_solve_without_params(self):
        # generate and solve a PMEDM instance
        pmd = PMEDM(
            year=2019,
            serial=serial,
            wt=wt,
            cind=cind,
            cg1=ctrt,
            cg2=cbg,
            sg1=strt,
            sg2=sbg,
            include_cg0=False,
            keep_solver=False,
            allocation_matrix=False,
            verbose=False,
        )
        pmd.solve()

        # result state not kept
        observed = pmd.res
        known = None
        assert observed == known

        # allocation matrix not kept
        observed = pmd.almat
        known = None
        assert observed == known

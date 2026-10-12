"""
Revisiting the policy-period dummies (post-paper revision).

Question: do the 2003, 2012 and 2017 period dummies measure the effect of CBA
changes, or something else?

Checks
  A. Team FE: nominal payroll + dummies vs relative payroll + dummies.
  B. Do the dummies survive a linear time trend / year effects?
  C. Year-by-year cross-sectional payroll-wins slope (relative payroll):
     is the efficiency of spending stable across CBA eras?
  D. Interrupted time series on payroll dispersion (coefficient of variation)
     with HAC errors: is there a level or slope break at each CBA year?
  E. Descriptive event-study plot of dispersion around each CBA year.

Everything here is descriptive evidence about timing. A policy that applies to
all 30 teams at once has no untreated comparison group, so none of this
identifies a causal effect on its own.

Run:  python python/policy_revisited.py
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

d = pd.read_csv(ROOT / "data" / "team_panel_1985_2023.csv")
d["rel"] = d.payroll / d.groupby("year").payroll.transform("mean")
d["trend"] = d.year - d.year.min()
x = d.set_index(["team", "year"])
G = d.team.nunique()
POL = ["policy2003", "policy2012", "policy2017"]


def fe(cols, label, time_effects=False):
    m = PanelOLS(x.winpct, x[cols], entity_effects=True,
                 time_effects=time_effects).fit(cov_type="clustered", cluster_entity=True)
    se = m.std_errors * np.sqrt(G / (G - 1))   # Stata xtreg small-sample scaling
    t = m.params / se
    p = pd.Series(2 * stats.t.sf(np.abs(t), G - 1), index=t.index)
    return pd.DataFrame({"model": label, "term": m.params.index, "coef": m.params.values,
                         "se": se.values, "t": t.values, "p": p.values,
                         "within_r2": m.rsquared_within})


# ---- A/B: fixed effects specifications --------------------------------------
A = pd.concat([
    fe(["payroll"] + POL, "A1 nominal payroll + dummies"),
    fe(["rel"] + POL, "A2 relative payroll + dummies"),
    fe(["rel", "paid_luxury_tax"] + POL, "A3 relative payroll + tax + dummies"),
    fe(["rel", "paid_luxury_tax"], "A4 relative payroll + tax"),
    fe(["payroll", "trend"], "B1 nominal payroll + linear trend"),
    fe(["rel"], "B2 relative payroll, year effects", time_effects=True),
])
A.to_csv(RES / "py_policy_revisited_fe.csv", index=False)

# ---- C: year-by-year cross-sectional slope -----------------------------------
rows = []
for yr, g in d.groupby("year"):
    X = sm.add_constant(g.rel)
    m = sm.OLS(g.winpct, X).fit()
    rows.append((yr, m.params["rel"], m.bse["rel"], m.rsquared, len(g)))
C = pd.DataFrame(rows, columns=["year", "slope", "se", "r2", "n"])
C.to_csv(RES / "py_yearly_payroll_win_slope.csv", index=False)
era = pd.cut(C.year, [1984, 2002, 2011, 2016, 2023], labels=["1985-02", "2003-11", "2012-16", "2017-23"])
era_tbl = C.groupby(era).agg(mean_slope=("slope", "mean"), mean_r2=("r2", "mean"), years=("year", "size"))
era_tbl.to_csv(RES / "py_slope_by_era.csv")

# ---- D: interrupted time series on dispersion --------------------------------
disp = d.groupby("year").agg(cv_payroll=("payroll", lambda s: s.std() / s.mean()),
                             sd_win=("winpct", "std")).reset_index()
disp["t"] = disp.year - disp.year.min()
out = []
for outcome in ["cv_payroll", "sd_win"]:
    for brk in (2003, 2012, 2017):
        z = disp.copy()
        z["post"] = (z.year >= brk).astype(int)
        z["post_t"] = z.post * (z.year - brk)
        m = sm.OLS(z[outcome], sm.add_constant(z[["t", "post", "post_t"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        for term in ("post", "post_t"):
            out.append((outcome, brk, term, m.params[term], m.bse[term], m.pvalues[term]))
D = pd.DataFrame(out, columns=["outcome", "break_year", "term", "coef", "se", "p"])
D.to_csv(RES / "py_its_dispersion.csv", index=False)

# joint model: all three breaks at once (level + slope change at each CBA year)
J = []
for outcome in ["cv_payroll", "sd_win"]:
    z = disp.copy()
    cols = ["t"]
    for brk in (2003, 2012, 2017):
        z[f"lvl{brk}"] = (z.year >= brk).astype(int)
        z[f"slp{brk}"] = z[f"lvl{brk}"] * (z.year - brk)
        cols += [f"lvl{brk}", f"slp{brk}"]
    m = sm.OLS(z[outcome], sm.add_constant(z[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    # joint Wald test that all six break terms are zero
    R = np.zeros((6, len(m.params))); names = list(m.params.index)
    for i, c in enumerate(cols[1:]):
        R[i, names.index(c)] = 1
    w = m.wald_test(R, scalar=True)
    for c in cols:
        J.append((outcome, c, m.params[c], m.bse[c], m.pvalues[c], float(w.pvalue), m.rsquared))
J = pd.DataFrame(J, columns=["outcome", "term", "coef", "se", "p", "joint_wald_p_all_breaks", "r2"])
J.to_csv(RES / "py_its_dispersion_joint.csv", index=False)
print("\nJoint ITS:\n", J.round(4).to_string(index=False))

# ---- figures -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(C.year, C.slope, yerr=1.96 * C.se, fmt="o", ms=3, lw=0.8, color="#1f4e79")
for b in (2003, 2012, 2017):
    ax[0].axvline(b - 0.5, color="gray", ls="--", lw=0.8)
ax[0].axhline(C.slope.mean(), color="#c00000", lw=1)
ax[0].set_title("Wins per unit of relative payroll, by season")
ax[0].set_ylabel("slope of win % on payroll / league mean"); ax[0].set_xlabel("season")
ax[1].plot(disp.year, disp.cv_payroll, color="#1f4e79")
for b in (2003, 2012, 2017):
    ax[1].axvline(b - 0.5, color="gray", ls="--", lw=0.8)
ax[1].set_title("Payroll dispersion (coefficient of variation)")
ax[1].set_xlabel("season")
fig.tight_layout(); fig.savefig(FIG / "policy_revisited.png", dpi=160)

print(A.round(4).to_string(index=False))
print("\nEra table (yearly slope):\n", era_tbl.round(4))
print("\nITS on dispersion:\n", D.round(4).to_string(index=False))

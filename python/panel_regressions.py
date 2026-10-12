"""Python replication of the four panel specifications, plus robustness checks.

Run from the repo root:  python python/panel_regressions.py
"""
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

df = pd.read_csv("data/team_panel_1985_2023.csv")
x = df.set_index(["team", "year"])
cl = df["team"].astype("category").cat.codes
rows = []


def add(label, params, tstats, nobs, r2):
    for k in params.index:
        if k in ("Intercept",):
            continue
        rows.append({"model": label, "term": k, "coef": params[k], "t": round(tstats[k], 2), "nobs": nobs, "r2": round(r2, 4)})


# Models as in the paper (conventional SEs, as in the original Stata output)
m1 = PanelOLS(x.winpct, x[["payroll"]], entity_effects=True).fit()
add("1. Team FE", m1.params, m1.tstats, m1.nobs, m1.rsquared_within)
r2 = smf.ols("winpct ~ payroll + paid_luxury_tax", df).fit()
add("2. OLS + luxury tax dummy", r2.params, r2.tvalues, r2.nobs, r2.rsquared)
r3 = smf.ols("winpct ~ post2003 + payroll", df).fit()
add("3. OLS + post-2003", r3.params, r3.tvalues, r3.nobs, r3.rsquared)
m4 = PanelOLS(x.winpct, x[["payroll", "policy2003", "policy2012", "policy2017"]], entity_effects=True).fit()
add("4. Team FE + policy periods", m4.params, m4.tstats, m4.nobs, m4.rsquared_within)

# Robustness: standard errors clustered by team
for label, f in [("2c. OLS + lux tax, clustered", "winpct ~ payroll + paid_luxury_tax"),
                 ("3c. OLS + post-2003, clustered", "winpct ~ post2003 + payroll")]:
    r = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": cl})
    add(label, r.params, r.tvalues, r.nobs, r.rsquared)
m4c = PanelOLS(x.winpct, x[["payroll", "policy2003", "policy2012", "policy2017"]], entity_effects=True).fit(
    cov_type="clustered", cluster_entity=True)
add("4c. Team FE + policy, clustered", m4c.params, m4c.tstats, m4c.nobs, m4c.rsquared_within)

# Robustness: nominal payroll trends upward over time, so also use payroll relative to the league average
df["rel_payroll"] = df["payroll"] / df.groupby("year")["payroll"].transform("mean")
x = df.set_index(["team", "year"])
m5 = PanelOLS(x.winpct, x[["rel_payroll"]], entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
add("5. Team FE, payroll relative to league mean, clustered", m5.params, m5.tstats, m5.nobs, m5.rsquared_within)
r6 = smf.ols("winpct ~ payroll + C(team) + C(year)", df).fit(cov_type="cluster", cov_kwds={"groups": cl})
add("6. Two-way FE (team + year), clustered", r6.params[["payroll"]], r6.tvalues[["payroll"]], r6.nobs, r6.rsquared)

out = pd.DataFrame(rows)
out["coef"] = out["coef"].map(lambda v: float(f"{v:.4g}"))
out.to_csv("results/py_panel_models.csv", index=False)
print(out.to_string(index=False))
print("\nSample:", len(df), "team-seasons,", df.team.nunique(), "teams,", df.year.min(), "-", df.year.max(),
      "| seasons per team:", df.groupby("team").size().min(), "to", df.groupby("team").size().max())
print("Team-seasons paying the tax:", int(df.paid_luxury_tax.sum()))

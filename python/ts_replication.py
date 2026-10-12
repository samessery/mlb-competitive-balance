"""Python replication of the time-series analysis (ADF, ARIMA, VAR, Granger).

Run from the repo root:  python python/ts_replication.py
"""
import warnings

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

df = pd.read_csv("data/sd_series_1985_2023.csv").set_index("year")
payroll, winpct = df["payroll_sd"], df["winpct_sd"]

# --- Stationarity (ADF with AIC lag selection; results are lag-sensitive, see README)
rows = []
for name, s in [("payroll_sd", payroll), ("winpct_sd", winpct),
                ("diff(payroll_sd)", payroll.diff().dropna()), ("diff(winpct_sd)", winpct.diff().dropna())]:
    stat, p, lags, *_ = adfuller(s, regression="c", autolag="AIC")
    rows.append({"series": name, "adf_stat": round(stat, 3), "p_value": round(p, 3), "lags": lags})
pd.DataFrame(rows).to_csv("results/py_adf_tests.csv", index=False)
print(pd.DataFrame(rows))

# --- ARIMA specs from the paper plus an AIC grid
out = []
for name, s, order in [("payroll_sd", payroll, (0, 1, 1)), ("winpct_sd", winpct, (1, 1, 0))]:
    fit = ARIMA(s.values, order=order).fit()
    lb = acorr_ljungbox(fit.resid[1:], lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    out.append({"series": name, "order": str(order), "param": fit.params[0].round(4),
                "se": fit.bse[0].round(4), "aic": round(fit.aic, 1), "ljung_box_p_lag10": round(lb, 3)})
pd.DataFrame(out).to_csv("results/py_arima_fits.csv", index=False)
print(pd.DataFrame(out))

grid = []
for name, s in [("payroll_sd", payroll), ("winpct_sd", winpct)]:
    for p in range(3):
        for q in range(3):
            try:
                grid.append({"series": name, "order": f"({p},1,{q})", "aic": round(ARIMA(s.values, order=(p, 1, q)).fit().aic, 1)})
            except Exception:
                pass
grid = pd.DataFrame(grid).sort_values(["series", "aic"])
grid.to_csv("results/py_arima_aic_grid.csv", index=False)
print(grid.groupby("series").head(3))

# --- VAR on standardized first differences
d = pd.DataFrame({"Payroll_sd": payroll.diff(), "WinPct_sd": winpct.diff()}).dropna()
z = (d - d.mean()) / d.std()
model = VAR(z)
print("\nLag order chosen by information criteria:", model.select_order(3).selected_orders)
res = model.fit(1)
coefs = pd.concat({"coef": res.params, "p_value": res.pvalues}, axis=1).round(3)
coefs.to_csv("results/py_var1_coefficients.csv")
print(coefs)
g = pd.DataFrame({
    "test": ["Payroll_sd -> WinPct_sd", "WinPct_sd -> Payroll_sd"],
    "p_value": [res.test_causality("WinPct_sd", "Payroll_sd").pvalue,
                res.test_causality("Payroll_sd", "WinPct_sd").pvalue]}).round(3)
g.to_csv("results/py_granger.csv", index=False)
print(g)
print("Portmanteau (whiteness) p-value:", round(res.test_whiteness(nlags=10).pvalue, 3))

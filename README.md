# Payroll Dispersion, the Luxury Tax, and Competitive Balance in MLB

Time-series and panel analysis of whether payroll inequality and MLB's luxury tax (Competitive Balance Tax) are linked to competitive balance, 1985 to 2023. Written as a term paper at Providence College (Spring 2025) and presented at the college's 16th Annual Celebration of Student Scholarship.

The repo holds the cleaned data, the R code (path-fixed), a Python replication, and a plain account of what the numbers do and do not support.

## Questions

1. **League level:** How have the cross-team standard deviation of payroll (`payroll_sd`) and of winning percentage (`winpct_sd`) evolved, and does one predict the other over time? (ARIMA, VAR, Granger causality)
2. **Team level:** Is payroll associated with winning, and do luxury-tax periods change that relationship? (Fixed-effects and pooled OLS panels)

## Data

| File | Contents |
|---|---|
| `data/sd_series_1985_2023.csv` | Annual cross-team SD of payroll and of win percentage (39 years) |
| `data/team_panel_1985_2023.csv` | Team-season panel: 1,128 observations, 30 teams, 26 to 39 seasons each (unbalanced); wins, payroll, luxury tax paid, policy-period indicators |

Sources: Baseball Reference (records) and Steve The Ump / Spotrac (payroll, luxury tax). Franchises are coded under their current names. Payroll is nominal dollars.

## Code

```
r/00_scrape_payroll.R      how the payroll table was collected (optional, not run here)
r/01_time_series_arima.R   ADF tests, ARIMA fits, Ljung-Box, AIC grid, forecasts   (tested)
r/02_var_granger.R         VAR(1) and Granger tests (needs the 'vars' package)      (not run here)
python/ts_replication.py   same time-series analysis in statsmodels                 (tested)
python/panel_regressions.py four panel models plus robustness checks                 (tested)
python/policy_revisited.py  relative payroll, yearly slopes, interrupted time series   (tested)
```

```bash
pip install -r requirements.txt
python python/ts_replication.py
python python/panel_regressions.py
Rscript r/01_time_series_arima.R
```

The Python panel results were checked against Stata xtreg output for the same four models (coefficients and t-statistics). The R and Python ARIMA estimates agree.

## Results

### Panel models (winning percentage on payroll)

| Model | Payroll coef | Other terms |
|---|---|---|
| 1. Team FE | 1.80e-10 (t = 4.39) | within R-squared 0.017 |
| 2. OLS + luxury tax paid | 1.87e-10 (t = 3.93) | luxury tax dummy +0.042 (t = 3.76); 60 team-seasons paid the tax |
| 3. OLS + post-2003 | 4.90e-10 (t = 9.36) | post-2003 -0.033 (t = -5.99) |
| 4. Team FE + policy periods | 4.00e-10 (t = 6.22) | 2003-11: -0.018, 2012-16: -0.030, 2017-23: -0.034 |

Payroll is positively associated with winning in every specification. Robustness checks (`results/py_panel_models.csv`): standard errors clustered by team keep every payroll and policy coefficient significant; the payroll coefficient is stable in a two-way (team and year) fixed-effects model (6.0e-10, t = 5.20) and when payroll is measured relative to the league average that year.

### Time series

- **Payroll dispersion** is a clearly non-stationary, rising series. ARIMA(0,1,1) gives MA(1) = -0.12 (SE 0.17).
- **Win-percentage dispersion** is much flatter. ARIMA(1,1,0) gives AR(1) = -0.11 (SE 0.17).
- **VAR(1)** on standardized first differences finds no significant coefficients. Granger tests: payroll to win dispersion p = 0.65, win to payroll dispersion p = 0.33. Residuals pass the portmanteau test (p = 0.95).

## Revisions since the original paper (inefficiency framing)

The original paper and poster are unchanged. This section records what I would do differently now, and what the corrected evidence says. The organizing idea across my work is **inefficiency**: payroll buys wins at a stable rate, but how well teams convert dollars to wins differs a lot, and the policy period does not change that rate.

Script: `python/policy_revisited.py` (outputs in `results/py_policy_revisited_fe.csv`, `py_yearly_payroll_win_slope.csv`, `py_slope_by_era.csv`, `py_its_dispersion*.csv`; figure `figures/policy_revisited.png`).

**1. The era dummies were measuring inflation, not the CBAs.** With nominal payroll, the 2003, 2012 and 2017 dummies are negative and significant (-0.018, -0.030, -0.034). Average win percentage is 0.50 in every era by construction, while payroll roughly quadrupled, so a model in nominal dollars needs a negative era term to compensate. Measured relative to the league average that season, the dummies are zero (+0.001, p > 0.85) and the payroll effect is much stronger (t = 8.1 vs 4.5; within R-squared 0.071 vs 0.035). The luxury tax paid dummy is also zero once payroll is relative (+0.009, p = 0.52).

**2. The payroll-to-wins rate is stable across eras.** Estimating the slope of win percentage on relative payroll separately for each season, the mean slope is 0.073 (1985-2002), 0.074 (2003-11), 0.053 (2012-16) and 0.075 (2017-23). The 2012-16 dip is small relative to year-to-year noise (see the figure). Spending is not getting more or less efficient at the league level when the rules change.

**3. Where the rules do show up: payroll dispersion.** An interrupted time series on the cross-team coefficient of variation of payroll, with a break at each CBA year (HAC errors, all three breaks estimated jointly, joint test p < 0.001), shows the pre-2003 upward drift in dispersion bending down after 2003 (slope change -0.012 per year, p < 0.001) and again after 2012 (-0.014, p = 0.02), then reversing sharply after 2017 (+0.049, p < 0.001). Win-percentage dispersion steps up after 2012 and 2017 (+0.010, p = 0.03; +0.026, p < 0.001). This is timing evidence. I chose the break years in advance from the CBA calendar, not from the data, but with 39 annual observations and six break terms it is suggestive and not definitive.

**4. On causality.** I believe the CBA changes affected behavior, and the dispersion results above are consistent with that. But a rule that applies to all 30 teams at once has no untreated comparison group, so period dummies alone cannot separate it from other things that changed in the same years (revenue sharing, analytics adoption, tanking incentives, the 2020 season). The strongest causal design available would use the tax thresholds, which only bind for some teams (see the bunching analysis in the companion repo, if present). I therefore describe the results as policy-timed changes, not causal effects.

## Caveats and corrections to the original paper

These are things I found when re-running the work. The qualitative story survives, but several statements in the paper are stronger than the output supports.

1. **Sample.** The paper describes a balanced panel from 1975 to 2024. The estimation sample is unbalanced and covers 1985 to 2023 (1,128 team-seasons).
2. **MA and AR terms are weak.** Neither ARIMA coefficient is statistically distinguishable from zero, so "a strong negative spike at lag one" overstates the evidence. By AIC, a random walk, ARIMA(0,1,0), beats ARIMA(0,1,1) for payroll dispersion, and ARIMA(0,1,2) beats ARIMA(1,1,0) for win dispersion.
3. **No "momentum" in win dispersion.** The paper describes momentum in changes in competitive balance. The estimated AR(1) coefficient is negative, not positive.
4. **Diagnostics are lag-dependent.** The Ljung-Box test on win-dispersion residuals passes at lag 10 (p = 0.31) but fails at lag 24 (p = 0.026). With only 38 observations, lag 24 is also a very long window.
5. **ADF results depend on lag choice.** With AIC lag selection (Python), differenced payroll dispersion looks stationary (p = 0.013). With the tseries default of three lags (R), it does not (p = 0.25). The VAR on first differences assumes stationarity.
6. **VAR lag order.** Information criteria select zero lags; VAR(1) was chosen by judgment. This does not change the null result.
7. **Policy-period dummies are period effects.** The 2003, 2012, and 2017 dummies partition time, so their negative signs also capture other changes across eras (for example, how performance and payroll spreads evolved). They are not clean estimates of the luxury tax's causal effect. The luxury-tax-paid dummy is positive because teams that pay the tax are high spenders who win.
8. **Nominal payroll.** Payroll is in nominal dollars and trends upward. Relative-to-league payroll and year effects are included above as checks.
9. **Poster claim.** The poster states that the luxury tax helped reduce payroll inequality. The analysis here does not test that directly, and the ARIMA results show payroll dispersion still rising through 2023.

## How this repo fits with my other baseball work

The common thread is **inefficiency**: payroll buys wins at a fairly stable rate, but how much teams get for their money varies, and the same win is worth different amounts to different teams.

| Repo | Question |
|---|---|
| `mlb-contract-efficiency` (this one) | Does the labor market pay for the skills that win? |
| `mlb-competitive-balance` | Do payroll dispersion and the luxury tax move competitive balance? Includes revisions to the original paper. |
| `mlb-efficiency-frontier` | How far below the best-practice frontier do teams fall, and does the gap persist or rotate? |
| `mlb-win-forecasting` | Do skill metrics and payroll forecast next year's wins beyond regression to the mean? |
| `mlb-playoff-value` | What is one extra win worth, by team quality and playoff format? |
| `mlb-player-salary-structure` | How does pay respond to production across a career, and who captures the surplus? |

## Author

Samuel Essery, B.S. Quantitative Economics, Providence College.

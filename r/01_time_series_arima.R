# ARIMA analysis of league-wide payroll dispersion and competitive balance, 1985-2023.
#   payroll_sd : cross-team standard deviation of payroll in each season
#   winpct_sd  : cross-team standard deviation of winning percentage in each season
# Run from the repo root:  Rscript r/01_time_series_arima.R
suppressMessages(library(tseries))

dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

df <- read.csv("data/sd_series_1985_2023.csv")
payroll.ts <- ts(df$payroll_sd, start = 1985, frequency = 1)
winpct.ts  <- ts(df$winpct_sd,  start = 1985, frequency = 1)
d_payroll  <- diff(payroll.ts)
d_winpct   <- diff(winpct.ts)

# Stationarity: augmented Dickey-Fuller on levels and first differences
adf <- data.frame(
  series = c("payroll_sd", "winpct_sd", "diff(payroll_sd)", "diff(winpct_sd)"),
  p_value = c(adf.test(payroll.ts)$p.value, adf.test(winpct.ts)$p.value,
              adf.test(d_payroll)$p.value, adf.test(d_winpct)$p.value))
write.csv(adf, "results/r_adf_tests.csv", row.names = FALSE)
print(adf)

# ARIMA fits used in the paper
fit_payroll <- arima(payroll.ts, order = c(0, 1, 1))
fit_winpct  <- arima(winpct.ts,  order = c(1, 1, 0))
cat("\nPayroll SD, ARIMA(0,1,1):\n"); print(fit_payroll)
cat("\nWinPct SD, ARIMA(1,1,0):\n");  print(fit_winpct)

# Residual diagnostics: Ljung-Box at lags 10 and 24
lb <- function(fit, lag) Box.test(residuals(fit), lag = lag, type = "Ljung-Box", fitdf = length(fit$coef))$p.value
diag <- data.frame(model = c("payroll ARIMA(0,1,1)", "winpct ARIMA(1,1,0)"),
                   ljung_box_p_lag10 = c(lb(fit_payroll, 10), lb(fit_winpct, 10)),
                   ljung_box_p_lag24 = c(lb(fit_payroll, 24), lb(fit_winpct, 24)))
write.csv(diag, "results/r_ljung_box.csv", row.names = FALSE)
print(diag)

# Model comparison by AIC over a small grid
grid <- expand.grid(p = 0:2, q = 0:2)
aic_for <- function(x, p, q) tryCatch(AIC(arima(x, order = c(p, 1, q))), error = function(e) NA)
grid$aic_payroll <- mapply(aic_for, list(payroll.ts), grid$p, grid$q)
grid$aic_winpct  <- mapply(aic_for, list(winpct.ts),  grid$p, grid$q)
write.csv(grid, "results/r_arima_aic_grid.csv", row.names = FALSE)

# Five-year forecasts
fc_payroll <- predict(fit_payroll, n.ahead = 5)
fc_winpct  <- predict(fit_winpct,  n.ahead = 5)
write.csv(data.frame(year = 2024:2028, payroll_sd = fc_payroll$pred, payroll_se = fc_payroll$se,
                     winpct_sd = fc_winpct$pred, winpct_se = fc_winpct$se),
          "results/r_forecasts.csv", row.names = FALSE)

plot_fc <- function(x, fc, main, ylab, file) {
  png(file, width = 1400, height = 900, res = 150)
  ts.plot(x, fc$pred, fc$pred + 2 * fc$se, fc$pred - 2 * fc$se,
          col = c("black", "blue", "red", "red"), lty = c(1, 2, 3, 3), main = main, ylab = ylab)
  legend("topleft", legend = c("Actual", "Forecast", "~95% interval"),
         col = c("black", "blue", "red"), lty = c(1, 2, 3), bty = "n")
  dev.off()
}
plot_fc(payroll.ts, fc_payroll, "Payroll SD forecast, ARIMA(0,1,1)", "Payroll SD", "figures/r_payroll_sd_forecast.png")
plot_fc(winpct.ts,  fc_winpct,  "Win% SD forecast, ARIMA(1,1,0)", "Win% SD", "figures/r_winpct_sd_forecast.png")

# VAR(1) on standardized first differences of payroll_sd and winpct_sd, plus Granger tests.
# Requires the 'vars' package:  install.packages("vars")
# Run from the repo root:  Rscript r/02_var_granger.R
suppressMessages(library(vars))

df <- read.csv("data/sd_series_1985_2023.csv")
d_payroll <- diff(ts(df$payroll_sd, start = 1985))
d_winpct  <- diff(ts(df$winpct_sd,  start = 1985))

# Standardize: payroll SD is in millions of dollars, win% SD is below 0.1
z <- cbind(Payroll_sd = as.numeric(scale(d_payroll)), WinPct_sd = as.numeric(scale(d_winpct)))
z <- ts(z, start = 1986)

print(VARselect(z, lag.max = 3, type = "const"))   # information-criteria lag choice
var1 <- VAR(z, p = 1, type = "const")
print(summary(var1))
print(causality(var1, cause = "Payroll_sd")$Granger)
print(causality(var1, cause = "WinPct_sd")$Granger)
print(serial.test(var1, lags.pt = 10, type = "PT.asymptotic"))

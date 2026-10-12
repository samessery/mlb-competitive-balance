# Optional: how the 1975-2023 payroll table was originally collected (Spotrac, spring 2025).
# Not needed to reproduce results (data/ already contains the cleaned series).
# Site layouts and terms change; check the site's terms of use before running.
library(rvest)
library(dplyr)
library(readr)

scrape_payroll <- function(year) {
  url <- paste0("https://www.spotrac.com/mlb/payroll/", year, "/")
  page <- tryCatch(read_html(url), error = function(e) NULL)
  if (is.null(page)) return(NULL)
  tbl <- html_table(page, fill = TRUE)
  if (length(tbl) < 1) return(NULL)
  mutate(tbl[[1]], Year = year)
}

payroll <- bind_rows(lapply(1975:2023, scrape_payroll))
dir.create("data", showWarnings = FALSE)
write_csv(payroll, "data/raw_payroll_scrape.csv")

# R oracle for tests/test_estimation.py (survey 4.5).
#
#   Rscript estimation_r.R \
#     ../../../svy/tests/test_data/svy_synthetic_sample_07082025.csv estimation_r.json
#
# Each estimator drops the rows the Python test drops before building the design.
suppressMessages({library(survey); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], stringsAsFactors = FALSE)
dat$resp2_new <- ifelse(dat$resp2 == 1, 1, ifelse(dat$resp2 == 2, 0, NA))

designs <- list(
  none = list(ids = ~1, strata = NULL),
  strata = list(ids = ~1, strata = ~region),
  psu = list(ids = ~cluster, strata = NULL),
  strata_psu = list(ids = ~cluster, strata = ~region)
)
mk <- function(cols, spec) {
  keep <- complete.cases(dat[, c(cols, "samp_wgt")])
  svydesign(ids = spec$ids, strata = spec$strata, weights = ~samp_wgt, data = dat[keep, ])
}
est_se <- function(est, se) list(est = as.numeric(est), se = as.numeric(se))
by_name <- function(est, se, names) setNames(Map(est_se, est, se), names)

overall <- function(cols, fit) lapply(designs, function(spec) {
  r <- fit(mk(cols, spec))
  if (length(coef(r)) == 1) est_se(coef(r), SE(r))
  else by_name(coef(r), SE(r), sub("^factor\\(resp2_new\\)", "", names(coef(r))))
})
by_educ <- function(cols, fit) {
  r <- fit(mk(c(cols, "educ"), designs$strata_psu))
  by_name(coef(r), SE(r), r$educ)
}

prop_by_educ <- function() {
  r <- svyby(~factor(resp2_new), ~educ, mk(c("resp2_new", "educ"), designs$strata_psu), svymean)
  cell <- function(i, lvl) {
    col <- paste0("factor(resp2_new)", lvl)
    est_se(r[[col]][i], r[[paste0("se.", col)]][i])
  }
  setNames(lapply(seq_along(r$educ), function(i) list(`0` = cell(i, "0"), `1` = cell(i, "1"))), r$educ)
}

golden <- list(
  mean = overall("income", function(d) svymean(~income, d)),
  total = overall("resp2_new", function(d) svytotal(~resp2_new, d)),
  ratio = overall(c("income", "fam_size"), function(d) svyratio(~income, ~fam_size, d)),
  prop = overall("resp2_new", function(d) svymean(~factor(resp2_new), d)),
  mean_by_educ = by_educ("income", function(d) svyby(~income, ~educ, d, svymean)),
  total_by_educ = by_educ("resp2_new", function(d) svyby(~resp2_new, ~educ, d, svytotal)),
  ratio_by_educ = by_educ(c("income", "fam_size"),
                          function(d) svyby(~income, ~educ, d, svyratio, denominator = ~fam_size)),
  prop_by_educ = prop_by_educ()
)
write_json(golden, args[2], auto_unbox = TRUE, digits = I(17), pretty = TRUE)

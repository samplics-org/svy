# R oracle for tests/svy/estimation/test_replication_deff.py (survey 4.5).
#
#   Rscript replication_deff_r.R \
#     ../../../test_data/categorical_rep_30092026.csv replication_deff_r.json
#
# Design as in ../../categorical/golden/categorical_replication_r.R: svy's
# replicate weights, its per-replicate coefficients as rscales. svymean and
# svytotal give deff=TRUE ("wor") and deff="replace" ("wr"); where= is subset()
# and by= is svyby. R's replicate svyratio deffs the linearized total instead of
# the ratio it reports (and fails on "replace"), and svyvar has no deff, so
# ratio, cov and corr are the reported replicate variance over the SRS
# reference written out here.
suppressMessages({library(survey); library(jsonlite)})
options(digits = 15)
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], stringsAsFactors = FALSE)
dat$a <- factor(dat$a)
coefs <- c(brr_ = 1 / 8, fay_ = 1 / (8 * 0.7^2), jkn_ = 1 / 2, jk1_ = 11 / 12, bs_ = 1 / 50)
n_reps <- c(brr_ = 8, fay_ = 8, jkn_ = 12, jk1_ = 12, bs_ = 50)

mk <- function(prefix) {
  svrepdesign(data = dat, repweights = paste0("^", prefix), weights = ~w, type = "other",
              scale = 1, rscales = rep(coefs[[prefix]], n_reps[[prefix]]),
              combined.weights = TRUE, mse = FALSE)
}
refs <- list(wor = TRUE, wr = "replace")

stat <- function(r) list(est = unname(coef(r)), se = unname(SE(r)), deff = unname(deff(r)))
# deff.svyby fails on deff = "replace" (it negates the string), so read the column.
by_stat <- function(r) list(level = as.character(r[[1]]), est = unname(coef(r)),
                            se = unname(SE(r)), deff = unname(r[[grep("^DEff", names(r))]]))

# SRS variance of a weighted mean of z: svyvar's n/(n-1) form over n, times 1 - n/N for wor.
srs_mean <- function(z, w, ref) {
  n <- length(z); N <- sum(w)
  s2 <- n / (n - 1) * sum(w * (z - sum(w * z) / N)^2) / N
  s2 / n * (if (ref == "wor") (N - n) / N else 1)
}
rep_var <- function(des, f) {
  reps <- apply(weights(des, "analysis"), 2, f)
  svrVar(reps, des$scale, des$rscales, mse = des$mse, coef = f(weights(des, "sampling")))
}
wcov <- function(y, x, w) { my <- sum(w * y) / sum(w); mx <- sum(w * x) / sum(w); sum(w * (y - my) * (x - mx)) / sum(w) }

ratio_hand <- function(des, ref) {
  d <- des$variables; w <- weights(des, "sampling")
  r <- svyratio(~y, ~y2, des)
  est <- unname(coef(r)); v <- unname(r$var[1, 1])
  xbar <- sum(w * d$y2) / sum(w)
  list(est = est, se = sqrt(v), deff = v / (srs_mean(d$y - est * d$y2, w, ref) / xbar^2))
}
assoc_hand <- function(des, kind, ref) {
  d <- des$variables; w <- weights(des, "sampling"); n <- nrow(d)
  kish <- n / (n - 1)
  f <- if (kind == "cov") function(wt) kish * wcov(d$y, d$y2, wt)
       else function(wt) wcov(d$y, d$y2, wt) / sqrt(wcov(d$y, d$y, wt) * wcov(d$y2, d$y2, wt))
  est <- f(w); v <- unname(rep_var(des, f))
  a <- d$y - sum(w * d$y) / sum(w); b <- d$y2 - sum(w * d$y2) / sum(w)
  myx <- wcov(d$y, d$y2, w); myy <- wcov(d$y, d$y, w); mxx <- wcov(d$y2, d$y2, w)
  u <- if (kind == "cov") (a * b - myx) * kish else {
    yt <- a / sqrt(myy); xt <- b / sqrt(mxx)
    yt * xt - 0.5 * est * (yt^2 + xt^2)
  }
  list(est = est, se = sqrt(v), deff = v / srs_mean(u, w, ref))
}

out <- list()
for (prefix in names(coefs)) {
  des <- mk(prefix)
  dd <- des[dat$dom == 1, ]
  res <- list()
  for (ref in names(refs)) {
    de <- refs[[ref]]
    hand_by <- function(fun, ...) lapply(c("north", "south"), function(g) fun(des[dat$region == g, ], ...))
    res[[ref]] <- list(
      mean = stat(svymean(~y, des, deff = de)),
      mean_where = stat(svymean(~y, dd, deff = de)),
      mean_by = by_stat(svyby(~y, ~region, des, svymean, deff = de)),
      total = stat(svytotal(~y, des, deff = de)),
      total_by = by_stat(svyby(~y, ~region, des, svytotal, deff = de)),
      prop = stat(svymean(~a, des, deff = de)),
      prop_where = stat(svymean(~a, dd, deff = de)),
      prop_by = lapply(c("north", "south"), function(g) stat(svymean(~a, des[dat$region == g, ], deff = de))),
      factor_total = stat(svytotal(~a, des, deff = de)),
      ratio = ratio_hand(des, ref),
      ratio_where = ratio_hand(dd, ref),
      ratio_by = hand_by(ratio_hand, ref),
      cov = assoc_hand(des, "cov", ref),
      corr = assoc_hand(des, "corr", ref),
      corr_by = hand_by(assoc_hand, "corr", ref)
    )
  }
  out[[sub("_$", "", prefix)]] <- res
}
cat(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), file = args[2])
cat("R", as.character(packageVersion("survey")), "\n")

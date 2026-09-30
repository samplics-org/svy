# R oracle for tests/svy/categorical/test_categorical_replication.py (survey 4.5).
#
#   Rscript categorical_replication_r.R \
#     ../../../test_data/categorical_rep_30092026.csv categorical_replication_r.json
#
# The replicate weights are svy's (create_brr_wgts, fay_coef=0.3, create_jk_wgts
# on stratum/psu and on psu alone, create_bs_wgts(50), create_sdr_wgts(8)).
# R gets svy's per-replicate coefficients as rscales and svy's design df as
# degf, re-set after every subset: svy's replicate df does not shrink on a
# domain, R's `[.svyrep.design` recomputes it.
suppressMessages({library(survey); library(jsonlite)})
options(digits = 15)
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], stringsAsFactors = FALSE)
coefs <- c(brr_ = 1 / 8, fay_ = 1 / (8 * 0.7^2), jkn_ = 1 / 2, jk1_ = 11 / 12, bs_ = 1 / 50, sdr_ = 4 / 8)
n_reps <- c(brr_ = 8, fay_ = 8, jkn_ = 12, jk1_ = 12, bs_ = 50, sdr_ = 8)
dat$a <- factor(dat$a); dat$b <- factor(dat$b); dat$grp <- factor(dat$grp); dat$k3 <- factor(dat$k3)
dfs <- c(brr_ = 6, fay_ = 6, jkn_ = 6, jk1_ = 11, bs_ = 49, sdr_ = 8)

mk <- function(prefix) {
  svrepdesign(data = dat, repweights = paste0("^", prefix), weights = ~w, type = "other",
              scale = 1, rscales = rep(coefs[[prefix]], n_reps[[prefix]]),
              combined.weights = TRUE, mse = FALSE,
              degf = dfs[[prefix]])
}
subd <- function(des, keep, df) { s <- des[keep, ]; s$degf <- df; s }

cells <- function(des, f) {
  m <- svymean(f, des); t <- svytotal(f, des)
  list(names = names(coef(m)), prop = unname(coef(m)), prop_se = unname(SE(m)),
       total = unname(coef(t)), total_se = unname(SE(t)))
}
chisq <- function(des) {
  f <- svychisq(~a + b, des, statistic = "F"); c <- svychisq(~a + b, des, statistic = "Chisq")
  list(f = unname(f$statistic), ndf = unname(f$parameter[1]), ddf = unname(f$parameter[2]),
       f_p = unname(f$p.value), chisq = unname(c$statistic), chisq_p = unname(c$p.value))
}
tt1 <- function(des, h0 = 0) {
  des <- update(des, yc = y - h0)
  r <- svyttest(yc ~ 0, des)
  list(est = unname(r$estimate) + h0, se = unname(r$estimate / r$statistic),
       t = unname(r$statistic), df = unname(r$parameter), p = unname(r$p.value))
}
tt2 <- function(des) {
  r <- svyttest(y ~ grp, des)
  g <- svyby(~y, ~grp, des, svymean)
  list(diff = unname(r$estimate), se = unname(r$estimate / r$statistic), t = unname(r$statistic),
       df = unname(r$parameter), p = unname(r$p.value),
       means = unname(coef(g)), mean_se = unname(SE(g)))
}
rk2 <- function(des, test) {
  r <- svyranktest(y ~ grp, des, test = test)
  list(delta = unname(r$estimate), t = unname(r$statistic), df = unname(r$parameter),
       p = unname(r$p.value))
}
rkk <- function(des, test) {
  r <- svyranktest(y ~ k3, des, test = test)
  list(chisq = unname(r$parameter[1]), ndf = unname(r$statistic), ddf = unname(r$ddf),
       p = unname(r$p.value))
}
sq <- function(r, N) (r / N)^2

out <- list()
for (prefix in names(dfs)) {
  des <- mk(prefix); df <- dfs[[prefix]]
  dd <- subd(des, dat$dom == 1, df)
  res <- list(
    oneway = cells(des, ~a),
    oneway_where = cells(dd, ~a),
    twoway = cells(des, ~interaction(a, b)),
    twoway_where = cells(dd, ~interaction(a, b)),
    chisq = chisq(des),
    chisq_where = chisq(dd),
    tt1 = tt1(des),
    tt1_h0 = tt1(des, 11),
    tt_paired = { d2 <- update(des, yd = y - y2); r <- svyttest(yd ~ 0, d2)
                  list(est = unname(r$estimate), t = unname(r$statistic), p = unname(r$p.value)) },
    tt2 = tt2(des),
    tt2_where = tt2(dd),
    rk_kw = rk2(des, "KruskalWallis"),
    rk_vdw = rk2(des, "vanderWaerden"),
    rk_median = rk2(des, "median"),
    rk_custom = rk2(des, sq),
    rk_where = rk2(dd, "KruskalWallis"),
    rkk_kw = rkk(des, "KruskalWallis"),
    rkk_where = rkk(dd, "KruskalWallis")
  )
  for (reg in c("north", "south")) {
    dr <- subd(des, dat$region == reg, df)
    res[[paste0("tt1_by_", reg)]] <- tt1(dr)
    res[[paste0("tt2_by_", reg)]] <- tt2(dr)
    res[[paste0("rk_by_", reg)]] <- rk2(dr, "KruskalWallis")
    res[[paste0("rkk_by_", reg)]] <- rkk(dr, "KruskalWallis")
    res[[paste0("rkc_by_", reg)]] <- rk2(dr, sq)
    drw <- subd(des, dat$region == reg & dat$dom == 1, df)
    res[[paste0("tt2_where_by_", reg)]] <- tt2(drw)
  }
  out[[sub("_$", "", prefix)]] <- res
}
cat(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), file = args[2])
cat("R", as.character(packageVersion("survey")), "\n")

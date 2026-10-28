# R oracle for the Rust unit tests on src/test_fixtures.rs's sample (survey 4.5, MASS).
#
#   Rscript kernel_unit_r.R
#
# rake runs to convergence: R keeps only its last pass's projections, which match
# the final weights only once the weights have stopped moving.
#
# svyranktest's k-sample result names its fields backwards: `parameter` is the
# chi-square and `statistic` its df.
suppressMessages({library(survey); library(MASS)})
options(digits = 17)
d <- data.frame(
  st = c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
  psu = c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6),
  w = c(1.5, 2, 1, 3, 2.5, 1, 2, 1.5, 3, 1, 2, 2.5),
  y = c(3.1, 4.2, 2.8, 5.0, 4.4, 4.4, 6.1, 5.5, 2.2, 3.3, 4.8, 4.0),
  g2 = c(0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0),
  g3 = c("a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "b", "c"),
  r = c("x", "y", "x", "y", "y", "x", "x", "y", "x", "y", "y", "x"),
  cc = c("p", "p", "q", "q", "r", "p", "q", "r", "r", "p", "q", "q"),
  k = c(0, 7, 1, 12, 3, 0, 15, 2, 0, 9, 18, 1),
  x2 = c(1.0, 2.5, 0.5, 3.0, 2.0, 1.5, 4.0, 3.5, 0.8, 1.2, 2.8, 2.2)
)
d$g2f <- factor(d$g2)
des <- svydesign(ids = ~psu, strata = ~st, weights = ~w, data = d)
out <- function(name, v) cat(name, "=", paste(format(unname(v), digits = 17), collapse = ", "), "\n")

# ttest
t1 <- svyttest(y ~ 0, des)
out("t1_est", t1$estimate); out("t1_t", t1$statistic); out("t1_df", t1$parameter); out("t1_p", t1$p.value)
m1 <- svymean(~y, des); out("t1_se", SE(m1))
t2 <- svyttest(y ~ g2f, des)
out("t2_diff", t2$estimate); out("t2_t", t2$statistic); out("t2_df", t2$parameter); out("t2_p", t2$p.value)

# ranktest
for (tst in c("wilcoxon", "vanderWaerden", "median")) {
  r2 <- svyranktest(y ~ g2f, des, test = tst)
  out(paste0("rk2_", tst, "_delta"), r2$estimate); out(paste0("rk2_", tst, "_t"), r2$statistic)
  out(paste0("rk2_", tst, "_df"), r2$parameter); out(paste0("rk2_", tst, "_p"), r2$p.value)
}
rk <- svyranktest(y ~ g3, des, test = "wilcoxon")
out("rkk_chisq", rk$parameter); out("rkk_ndf", rk$statistic); out("rkk_ddf", rk$ddf); out("rkk_p", rk$p.value)

# distribution helpers
out("pt_2.3_7", 2 * pt(-2.3, 7)); out("pt_0.4_3.5", 2 * pt(-0.4, 3.5))
out("pf_3.1_2_9", pf(3.1, 2, 9, lower.tail = FALSE)); out("pf_0.7_1.6_8.2", pf(0.7, 1.6, 8.2, lower.tail = FALSE))
out("qnorm_.025_.3_.9", qnorm(c(0.025, 0.3, 0.9)))

# Rao-Scott: cells row-major, r outer
d$cell <- factor(paste(d$r, d$cc, sep = "_"), levels = c("x_p", "x_q", "x_r", "y_p", "y_q", "y_r"))
des <- update(des, cell = d$cell)
mc <- svymean(~cell, des)
out("rs_p", coef(mc)); out("rs_v", as.vector(vcov(mc)))
out("rs_degf", degf(des))
f <- svychisq(~r + cc, des, statistic = "F"); cs <- svychisq(~r + cc, des, statistic = "Chisq")
out("rs_F", f$statistic); out("rs_ndf", f$parameter[1]); out("rs_ddf", f$parameter[2]); out("rs_Fp", f$p.value)
out("rs_X2", cs$statistic); out("rs_X2p", cs$p.value)

# wols: weighted least squares and its design-based SE
fit <- lm(y ~ g2 + x2, data = d, weights = w)
out("wols_beta", coef(fit))
gl <- svyglm(y ~ g2 + x2, des)
out("wols_se", SE(gl))

# negbin
nb <- glm.nb(k ~ x2, data = d, weights = w)
out("nb_beta", coef(nb)); out("nb_theta", nb$theta)
mu <- fitted(nb)
out("nb_theta_ml", theta.ml(d$k, mu, weights = d$w, limit = 100, eps = .Machine$double.eps^0.25))
out("nb_loglik_th2", sum(d$w * dnbinom(d$k, size = 2, mu = mu, log = TRUE)))
out("nb_mu", mu)

# calibration sweeps: svytotal(~y) SE after each adjustment
ps <- postStratify(des, ~r, data.frame(r = c("x", "y"), Freq = c(15, 13)))
out("ps_w", weights(ps)); out("ps_se", SE(svytotal(~y, ps)))
rk2 <- rake(des, list(~r, ~cc), list(data.frame(r = c("x", "y"), Freq = c(15, 13)),
                                     data.frame(cc = c("p", "q", "r"), Freq = c(9, 11, 8))),
            control = list(maxit = 100, epsilon = 1e-12))
out("rake_w", weights(rk2)); out("rake_se", SE(svytotal(~y, rk2)))
gr <- calibrate(des, ~x2, c(`(Intercept)` = 25, x2 = 55))
out("greg_w", weights(gr)); out("greg_se", SE(svytotal(~y, gr)))
out("uncal_se", SE(svytotal(~y, des)))
nbt <- glm.nb(k ~ x2, data = d, weights = w, control = glm.control(epsilon = 1e-15, maxit = 500))
out("nbt_beta", coef(nbt)); out("nbt_theta", nbt$theta)
out("nbt_theta_ml_tight", theta.ml(d$k, fitted(nb), weights = d$w, limit = 100, eps = 1e-14))

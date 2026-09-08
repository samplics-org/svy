# scripts/gen_r_reference.R
#
# Regenerate the R reference values in
# tests/svy/regression/test_glm_against_rsurvey.py.
#
# Every model is run through survey's svyglm at epsilon = 1e-12, maxit = 100 —
# the same stopping rule the tests give svy — and printed at 17 significant
# digits, so the comparison measures the implementations rather than either
# side's default convergence tolerance.
#
# Run from packages/svy:  Rscript scripts/gen_r_reference.R
# Needs R with the survey package (4.5 when these values were recorded).

suppressMessages(library(survey))
options(digits = 17)

d <- read.csv("tests/test_data/apistrat.csv")
d$y_bin <- as.numeric(d$api00 > 743)
d$y_pos <- d$api00 / 100

designs <- list(
  weights_only   = svydesign(ids = ~1, weights = ~pw, data = d),
  psu_only       = svydesign(ids = ~dnum, weights = ~pw, data = d),
  stratified     = svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = d),
  psu_stratified = svydesign(ids = ~dnum, strata = ~stype, weights = ~pw, data = d, nest = TRUE)
)

vec <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse = ", "), "]")

emit <- function(tag, fam, form) {
  for (nm in names(designs)) {
    f <- svyglm(form, designs[[nm]], family = fam,
                control = glm.control(epsilon = 1e-12, maxit = 100))
    s <- summary(f)$coefficients
    ci <- confint(f, ddf = df.residual(f))
    ft <- regTermTest(f, ~ ell + meals + mobility)$Ftest
    cat(sprintf("## %s / %s\n", tag, nm))
    cat("beta ", vec(coef(f)), "\n")
    cat("se   ", vec(s[, 2]), "\n")
    cat("lci  ", vec(ci[, 1]), "\n")
    cat("uci  ", vec(ci[, 2]), "\n")
    cat("t    ", vec(s[, 3]), "\n")
    cat("p    ", vec(s[, 4]), "\n")
    cat(sprintf("F     %.17g\ndf    %d\niters %d\ndev   %.17g\n\n",
                as.numeric(ft), df.residual(f), f$iter, deviance(f)))
  }
}

emit("gaussian identity", gaussian(), api00 ~ ell + meals + mobility)
emit("binomial logit", quasibinomial("logit"), y_bin ~ ell + meals + mobility)
emit("binomial probit", quasibinomial("probit"), y_bin ~ ell + meals + mobility)
emit("binomial cauchit", quasibinomial("cauchit"), y_bin ~ ell + meals + mobility)
emit("binomial cloglog", quasibinomial("cloglog"), y_bin ~ ell + meals + mobility)
emit("poisson sqrt", quasipoisson("sqrt"), enroll ~ ell + meals + mobility)
emit("inversegaussian 1/mu^2", inverse.gaussian(link = "1/mu^2"), y_pos ~ ell + meals + mobility)

# ---------------------------------------------------------------------------
# Negative binomial.
#
# There is no negative binomial in survey itself. The design-based fit is
# survey::svymle over the joint (theta, beta) likelihood — the method Lumley
# gives in *Complex Surveys* (Appendix E, p254ff) and what sjstats::svyglm.nb
# implements. The four functions below are sjstats', reproduced so this script
# needs only survey and MASS.
# ---------------------------------------------------------------------------

nb_loglik <- function(y, theta, eta) {
  mu <- exp(eta)
  lgamma(theta + y) - lgamma(theta) - lgamma(y + 1) + theta * log(theta) +
    y * log(mu + (y == 0)) - (theta + y) * log(theta + mu)
}
nb_deta <- function(y, theta, eta) {
  mu <- exp(eta)
  (y / mu - (theta + y) / (theta + mu)) * mu
}
nb_dtheta <- function(y, theta, eta) {
  mu <- exp(eta)
  digamma(theta + y) - digamma(theta) + log(theta) + 1 - log(theta + mu) -
    (y + theta) / (mu + theta)
}
nb_score <- function(y, theta, eta) {
  cbind(nb_dtheta(y, theta, eta), nb_deta(y, theta, eta))
}

emit_nb <- function(form) {
  for (nm in names(designs)) {
    des <- designs[[nm]]
    dw <- weights(des)
    des <- update(des, scaled.weights = dw / mean(dw, na.rm = TRUE))
    m <- MASS::glm.nb(form, data = model.frame(des), weights = scaled.weights,
                      control = glm.control(epsilon = 1e-12, maxit = 100))
    f <- svymle(loglike = nb_loglik, grad = nb_score, design = des,
                formulas = list(theta = ~1, eta = form),
                start = c(m$theta, coef(m)), na.action = "na.omit")
    est <- coef(f)
    se <- sqrt(diag(vcov(f)))
    cat(sprintf("## negative binomial log / %s\n", nm))
    cat("theta   ", sprintf("%.17g", est[1]), "\n")
    cat("theta_se", sprintf("%.17g", se[1]), "\n")
    cat("beta ", vec(est[-1]), "\n")
    cat("se   ", vec(se[-1]), "\n\n")
  }
}

emit_nb(enroll ~ ell + meals + mobility)

# Fixed theta: an ordinary svyglm with MASS's family object, so the variance
# conditions on theta rather than carrying a row for it.
emit_nb_fixed <- function(form, theta) {
  for (nm in names(designs)) {
    f <- svyglm(form, designs[[nm]], family = MASS::negative.binomial(theta),
                control = glm.control(epsilon = 1e-12, maxit = 100))
    s <- summary(f)$coefficients
    cat(sprintf("## negative binomial fixed theta=%g / %s\n", theta, nm))
    cat("beta ", vec(coef(f)), "\n")
    cat("se   ", vec(s[, 2]), "\n")
    cat(sprintf("df    %d\n\n", df.residual(f)))
  }
}

emit_nb_fixed(enroll ~ ell + meals + mobility, 2.5)

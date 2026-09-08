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

#[allow(dead_code)]
pub(crate) struct GlmResult {
    pub params: Vec<f64>,
    pub cov_params: Vec<f64>,
    /// R svyglm's naive.cov, (X'WX)^-1 at the final beta, on the fit's mean-1 weight scale.
    pub naive_cov: Vec<f64>,
    pub scale: f64,
    pub df_resid: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: u32,
    pub n_obs: usize,
    /// Whether IRLS met the tolerance rather than exhausting `max_iter`.
    pub converged: bool,
    /// Negative binomial theta, with its SE when estimated rather than supplied.
    pub theta: Option<f64>,
    pub theta_se: Option<f64>,
    /// Set when the fit pushed rows to the edge of the response's range.
    pub separation: Option<Separation>,
}

/// Rows the fit drove to the boundary, and what is left to identify beta.
pub(crate) struct Separation {
    /// Contributing rows whose fitted value the fit cannot tell from y.
    pub boundary_rows: usize,
    /// Row-major `sum_i w_i x_i x_i'` over the other rows; a coefficient outside its row space is not identified.
    pub gram: Vec<f64>,
}

/// In tolerances: separated rows' deviance share stops near 1.6 (logit, log; ~3 cauchit), ordinary rows orders above.
pub(super) const SEPARATION_RESOLUTION: f64 = 10.0;

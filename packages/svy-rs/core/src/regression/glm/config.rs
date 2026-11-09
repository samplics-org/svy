use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Family {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    /// Negative binomial at a known theta, Var(mu) = mu + mu^2/theta; `negbin` estimates theta around it.
    NegativeBinomial(f64),
}

impl Family {
    pub(crate) fn from_str(s: &str, theta: Option<f64>) -> RResult<Self> {
        match s.to_lowercase().as_str() {
            "gaussian" => Ok(Family::Gaussian),
            "binomial" => Ok(Family::Binomial),
            "poisson" => Ok(Family::Poisson),
            "gamma" => Ok(Family::Gamma),
            "inversegaussian" | "inverse_gaussian" => Ok(Family::InverseGaussian),
            "negativebinomial" | "negative_binomial" => {
                let th = theta.ok_or(RegressionError::ThetaMissing)?;
                if !(th.is_finite() && th > 0.0) {
                    return Err(RegressionError::ThetaInvalid(th));
                }
                Ok(Family::NegativeBinomial(th))
            }
            _ => Err(RegressionError::UnsupportedFamily(s.to_string())),
        }
    }

    pub(crate) fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu * mu,
            Family::InverseGaussian => mu * mu * mu,
            Family::NegativeBinomial(theta) => mu + mu * mu / theta,
        }
    }

    /// R's `family$initialize`, on the prior-weight scale: on a flat surface IRLS stops where its start leads.
    pub(super) fn initial_mu(&self, y: f64, w: f64) -> f64 {
        let eps = 1e-10;
        match self {
            Family::Binomial => (w * y + 0.5) / (w + 1.0),
            Family::Poisson => y + 0.1,
            Family::Gamma | Family::InverseGaussian => y.max(eps),
            Family::Gaussian => y,
            // MASS::negative.binomial's initialize: y + (y == 0)/6.
            Family::NegativeBinomial(_) => {
                if y == 0.0 {
                    1.0 / 6.0
                } else {
                    y
                }
            }
        }
    }

    /// R's `family$dev.resids`, so sum_i w_i d(y_i, mu_i) is svyglm's deviance on the same weight scale.
    pub(super) fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            Family::Gaussian => (y - mu).powi(2),
            Family::Binomial => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                let t1 = if y > 0.0 { y * (y / mu_c).ln() } else { 0.0 };
                let t2 = if y < 1.0 {
                    (1.0 - y) * ((1.0 - y) / (1.0 - mu_c)).ln()
                } else {
                    0.0
                };
                2.0 * (t1 + t2)
            }
            Family::Poisson => {
                let mu_c = mu.max(1e-10);
                let t = if y > 0.0 { y * (y / mu_c).ln() } else { 0.0 };
                2.0 * (t - (y - mu_c))
            }
            Family::Gamma => {
                let mu_c = mu.max(1e-10);
                let y_c = y.max(1e-10);
                2.0 * (-(y_c / mu_c).ln() + (y - mu_c) / mu_c)
            }
            Family::InverseGaussian => (y - mu).powi(2) / (y.max(1e-10) * mu * mu),
            // MASS::negative.binomial's dev.resids.
            Family::NegativeBinomial(theta) => {
                let mu_c = mu.max(1e-10);
                2.0 * (y * (y.max(1.0) / mu_c).ln()
                    - (y + theta) * ((y + theta) / (mu_c + theta)).ln())
            }
        }
    }
}

/// -qnorm(.Machine$double.eps): the eta bound R's probit linkinv applies.
const PROBIT_ETA_MAX: f64 = 8.125_890_664_701_904;

/// -qcauchy(.Machine$double.eps): the eta bound R's cauchit linkinv applies.
const CAUCHIT_ETA_MAX: f64 = 1.433_540_284_805_664_8e15;

/// Standard Cauchy CDF; outside [-1, 1] it uses `atan(1/x)` as R's pcauchy, since `atan(x)/pi + 0.5` cancels.
fn cauchy_cdf(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x > 1.0 {
        1.0 - (1.0 / x).atan() / PI
    } else if x < -1.0 {
        (-1.0 / x).atan() / PI
    } else {
        0.5 + x.atan() / PI
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Link {
    Identity,
    Logit,
    Probit,
    Cauchit,
    Cloglog,
    Log,
    Sqrt,
    Inverse,
    InverseSquared,
}

// Phi(x) = erfc(-x/sqrt2)/2 on the tail side, so mu near 0 or 1 stays precise; ~5e-15 of R's pnorm.
// erfc: the all-positive confluent series below 1, a modified-Lentz continued fraction above.

const M_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const SQRT_PI: f64 = 1.772_453_850_905_516;

fn norm_pdf(x: f64) -> f64 {
    M_1_SQRT_2PI * (-0.5 * x * x).exp()
}

fn erfc(z: f64) -> f64 {
    if z < 0.0 {
        return 2.0 - erfc(-z);
    }

    if z < 1.0 {
        // erf(z) = (2z/sqrt(pi)) e^{-z^2} sum_n (2z^2)^n / (1*3*...*(2n+1))
        let z2 = z * z;
        let mut term = 2.0 * z / SQRT_PI * (-z2).exp();
        let mut sum = term;
        for n in 1..64 {
            term *= 2.0 * z2 / (2 * n + 1) as f64;
            sum += term;
            if term <= sum * 1e-18 {
                break;
            }
        }
        return 1.0 - sum;
    }

    // erfc(z) = e^{-z^2}/sqrt(pi) * 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + ...))))
    const TINY: f64 = 1e-300;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / z;
    let mut f = d;
    for n in 1..400 {
        let a = n as f64 / 2.0;
        d = z + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = z + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-17 {
            break;
        }
    }
    (-z * z).exp() / SQRT_PI * f
}

pub(crate) fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

impl Link {
    pub(crate) fn from_str(s: &str) -> RResult<Self> {
        match s.to_lowercase().as_str() {
            "identity" => Ok(Link::Identity),
            "logit" => Ok(Link::Logit),
            "probit" => Ok(Link::Probit),
            "cauchit" => Ok(Link::Cauchit),
            "cloglog" => Ok(Link::Cloglog),
            "log" => Ok(Link::Log),
            "sqrt" => Ok(Link::Sqrt),
            "inverse" => Ok(Link::Inverse),
            "inverse_squared" => Ok(Link::InverseSquared),
            _ => Err(RegressionError::UnsupportedLink(s.to_string())),
        }
    }

    pub(crate) fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Logit => (mu / (1.0 - mu)).ln(),
            Link::Probit => probit(mu),
            // As R's qcauchy: tan(pi*(mu - 1/2)) is catastrophic as mu nears 0 or 1.
            Link::Cauchit => -1.0 / (std::f64::consts::PI * mu).tan(),
            Link::Cloglog => (-((1.0 - mu).max(1e-10).ln())).max(1e-10).ln(),
            Link::Log => mu.max(1e-10).ln(),
            Link::Sqrt => mu.max(0.0).sqrt(),
            Link::Inverse => 1.0 / mu,
            Link::InverseSquared => 1.0 / (mu * mu),
        }
    }

    pub(crate) fn inverse(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => eta,
            Link::Logit => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }

            // R's make.link clamps (eta at +/-qnorm(eps), mu in [eps, 1-eps]) keep the binomial variance positive.
            Link::Probit => norm_cdf(eta.clamp(-PROBIT_ETA_MAX, PROBIT_ETA_MAX)),
            Link::Cauchit => {
                let e = eta.clamp(-CAUCHIT_ETA_MAX, CAUCHIT_ETA_MAX);
                cauchy_cdf(e)
            }
            Link::Cloglog => {
                let m = -(-(eta.min(700.0).exp())).exp_m1();
                m.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
            }
            Link::Log => eta.clamp(-30.0, 30.0).exp(),
            Link::Sqrt => eta * eta,
            Link::Inverse => 1.0 / eta,
            Link::InverseSquared => 1.0 / eta.sqrt(),
        }
    }

    /// d2μ/dη2, for the negative binomial's joint bread, so only for that family's links.
    pub(crate) fn mu_eta2(&self, mu: f64, eta: f64) -> f64 {
        match self {
            Link::Identity => 0.0,
            Link::Logit => mu * (1.0 - mu) * (1.0 - 2.0 * mu),
            Link::Probit => -eta * norm_pdf(eta),
            Link::Cauchit => {
                let d = 1.0 + eta * eta;
                -2.0 * eta / (std::f64::consts::PI * d * d)
            }
            Link::Cloglog => {
                let e = eta.min(700.0).exp();
                e * (-e).exp() * (1.0 - e)
            }
            Link::Log => mu,
            Link::Sqrt => 2.0,
            Link::Inverse => 2.0 * mu * mu * mu,
            Link::InverseSquared => 0.75 * mu.powi(5),
        }
    }

    /// dμ/dη. Probit and cloglog are the arms that need eta rather than mu.
    pub(crate) fn mu_eta(&self, mu: f64, eta: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Logit => mu * (1.0 - mu),
            Link::Probit => norm_pdf(eta).max(f64::EPSILON),
            Link::Cauchit => (1.0 / (std::f64::consts::PI * (1.0 + eta * eta))).max(f64::EPSILON),
            Link::Cloglog => {
                let e = eta.min(700.0).exp();
                (e * (-e).exp()).max(f64::EPSILON)
            }
            Link::Log => mu,
            Link::Sqrt => 2.0 * eta,
            Link::Inverse => -(mu * mu),
            Link::InverseSquared => -0.5 * mu.powi(3),
        }
    }
}

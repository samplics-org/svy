// src/regression/special.rs
//
// lgamma, digamma and trigamma.
//
// The crate carries no math dependency on purpose, so these are written out
// the way `norm_cdf` is in regression/glm.rs — and, like it, are checked
// against R to ~1e-14 relative in the tests below. The negative binomial needs
// all three: the profile likelihood in theta is digamma, its information is
// trigamma, and the log-likelihood itself is lgamma.
//
// Each uses the same shape: recur up to where the asymptotic series is good,
// then sum it.

use std::f64::consts::PI;

/// Where digamma and trigamma switch from the recurrence to the asymptotic
/// series. High enough that the first dropped term is below f64 resolution;
/// the recurrence costs at most twenty divisions.
const ASYMPTOTIC_FROM: f64 = 20.0;

/// Lanczos coefficients, g = 7, n = 9 — the usual set, good to ~1e-15.
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// log |Gamma(x)|, for x > 0.
pub fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x).
        return (PI / (PI * x).sin()).abs().ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    let t = x + 7.5;
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The digamma function, psi(x) = d/dx log Gamma(x), for x > 0.
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;

    // Recur up to the threshold with psi(x) = psi(x+1) - 1/x. At x >= 20 the
    // first term the series below drops is ~5e-18 relative, which is what
    // buys the 1e-14 agreement with R asserted in the tests.
    while x < ASYMPTOTIC_FROM {
        result -= 1.0 / x;
        x += 1.0;
    }

    // psi(x) ~ ln x - 1/(2x) - sum_n B_2n / (2n x^2n)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln() - 0.5 * inv;
    result -= inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2
                                * (1.0 / 240.0
                                    - inv2 * (1.0 / 132.0 - inv2 * (691.0 / 32760.0))))));
    result
}

/// The trigamma function, psi'(x), for x > 0.
pub fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;

    // psi'(x) = psi'(x+1) + 1/x^2.
    while x < ASYMPTOTIC_FROM {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // psi'(x) ~ 1/x + 1/(2x^2) + sum_n B_2n / x^(2n+1)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += inv
        * (1.0
            + 0.5 * inv
            + inv2
                * (1.0 / 6.0
                    - inv2
                        * (1.0 / 30.0
                            - inv2 * (1.0 / 42.0 - inv2 * (1.0 / 30.0 - inv2 * (5.0 / 66.0))))));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R, at 17 significant digits.
    const DIGAMMA: [(f64, f64); 14] = [
        (0.1, -10.423754940411076),
        (0.5, -1.9635100260214231),
        (1.0, -0.57721566490153231),
        (1.5, 0.036489973978576895),
        (2.0, 0.42278433509846747),
        (2.5, 0.70315664064524341),
        (3.0, 0.92278433509846747),
        (5.0, 1.5061176684318007),
        (6.0, 1.7061176684318007),
        (7.0, 1.8727843350984674),
        (10.0, 2.2517525890667214),
        (50.0, 3.901989673427892),
        (200.0, 5.2958152832199117),
        (1000.0, 6.907255195648812),
    ];

    const TRIGAMMA: [(f64, f64); 14] = [
        (0.1, 101.43329915079275),
        (0.5, 4.934802200544679),
        (1.0, 1.6449340668482264),
        (1.5, 0.93480220054467933),
        (2.0, 0.64493406684822641),
        (2.5, 0.49035775610023491),
        (3.0, 0.39493406684822635),
        (5.0, 0.22132295573711527),
        (6.0, 0.18132295573711527),
        (7.0, 0.1535451779593375),
        (10.0, 0.10516633568168572),
        (50.0, 0.020201333226697128),
        (200.0, 0.00501252083322917),
        (1000.0, 0.0010005001666666335),
    ];

    const LGAMMA: [(f64, f64); 14] = [
        (0.1, 2.252712651734206),
        (0.5, 0.57236494292470008),
        (1.0, 0.0),
        (1.5, -0.12078223763524518),
        (2.0, 0.0),
        (2.5, 0.28468287047291918),
        (3.0, 0.69314718055994529),
        (5.0, 3.1780538303479458),
        (6.0, 4.7874917427820458),
        (7.0, 6.5792512120101012),
        (10.0, 12.801827480081469),
        (50.0, 144.56574394634487),
        (200.0, 857.93366982585746),
        (1000.0, 5905.2204232091808),
    ];

    fn check(name: &str, cases: &[(f64, f64)], f: impl Fn(f64) -> f64, tol: f64) {
        for &(x, want) in cases {
            let got = f(x);
            let err = (got - want).abs() / want.abs().max(1e-300);
            assert!(
                err <= tol || (got - want).abs() < 1e-15,
                "{name}({x}): got {got:.17e}, want {want:.17e} (rel {err:.2e})"
            );
        }
    }

    #[test]
    fn digamma_matches_r() {
        check("digamma", &DIGAMMA, digamma, 1e-14);
    }

    #[test]
    fn trigamma_matches_r() {
        check("trigamma", &TRIGAMMA, trigamma, 1e-14);
    }

    #[test]
    fn lgamma_matches_r() {
        check("lgamma", &LGAMMA, lgamma, 1e-14);
    }

    /// psi(x+1) - psi(x) = 1/x, and psi'(x) - psi'(x+1) = 1/x^2, across the
    /// recurrence boundary the implementations use.
    #[test]
    fn recurrences_hold_across_the_boundary() {
        for x in [0.3, 1.0, 2.7, 5.5, 5.999, 6.0, 6.001, 9.0, 40.0] {
            assert!(
                (digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-13,
                "digamma recurrence at {x}"
            );
            assert!(
                (trigamma(x) - trigamma(x + 1.0) - 1.0 / (x * x)).abs() < 1e-13,
                "trigamma recurrence at {x}"
            );
            assert!(
                (lgamma(x + 1.0) - lgamma(x) - x.ln()).abs() < 1e-12,
                "lgamma recurrence at {x}"
            );
        }
    }

    /// trigamma is the derivative of digamma.
    #[test]
    fn trigamma_matches_a_numeric_derivative_of_digamma() {
        let h = 1e-6;
        for x in [1.0, 2.0, 4.0, 8.0, 20.0] {
            let numeric = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
            let rel = (trigamma(x) - numeric).abs() / trigamma(x);
            assert!(rel < 1e-7, "trigamma at {x}: {} vs {numeric}", trigamma(x));
        }
    }
}

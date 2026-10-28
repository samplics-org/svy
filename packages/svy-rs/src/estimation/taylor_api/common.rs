use super::*;

/// One by-group: label, estimate, SE, variance, n, deff and scores.
pub(super) type GroupRow<'a> = (&'a str, f64, f64, f64, u32, f64, Vec<f64>);

/// The SRS reference for the design effect: wor with N = sum of weights unless told otherwise.
/// `deff_pop_total`, the declared population size, keeps it right under rescaled weights.
pub(crate) fn parse_srs_ref(
    deff_ref: Option<&str>,
    deff_pop_total: Option<f64>,
) -> EResult<SrsRef> {
    match deff_ref.unwrap_or("wor") {
        "wr" => Ok(SrsRef::WithReplacement),
        "wor" => Ok(SrsRef::WithoutReplacement {
            pop_total: deff_pop_total,
        }),
        other => Err(EstimationError::InvalidChoice {
            what: "deff reference",
            got: other.to_string(),
            expected: "'wor' or 'wr'",
        }),
    }
}

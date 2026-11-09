use super::*;

/// The replicate weights, when the caller asked for a replication variance.
pub(super) fn rep_spec(
    df: &DataFrame,
    rep_weight_cols: Option<Vec<String>>,
    rep_coefs: Option<Vec<f64>>,
    rep_df: Option<f64>,
    variance_center: &str,
) -> PyResult<Option<RepSpec>> {
    match rep_weight_cols {
        None => Ok(None),
        Some(cols) => {
            let coefs = rep_coefs.ok_or_else(|| {
                to_py_err(&CategoricalError::MissingInput(
                    "rep_coefs is required with rep_weight_cols",
                ))
            })?;
            RepSpec::from_frame(df, &cols, coefs, variance_center, rep_df)
                .map(Some)
                .map_err(|e| to_py_err(&e))
        }
    }
}

#![warn(unreachable_pub)]
mod core;
mod errors;
mod sas_read;
mod spss_read;
mod spss_write;
mod stata_read;
mod stata_write;
mod xpt_read;
mod xpt_write;

use pyo3::prelude::*;

#[pymodule]
fn svyreadstat_rs(m: &Bound<PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "ReadStatValueError",
        py.get_type::<errors::ReadStatValueError>(),
    )?;
    m.add(
        "ReadStatRuntimeError",
        py.get_type::<errors::ReadStatRuntimeError>(),
    )?;

    m.add_function(wrap_pyfunction!(sas_read::df_parse_sas_file, m)?)?;

    m.add_function(wrap_pyfunction!(spss_read::df_parse_sav_file, m)?)?;
    m.add_function(wrap_pyfunction!(spss_read::df_parse_por_file, m)?)?;
    m.add_function(wrap_pyfunction!(spss_write::df_write_sav_file, m)?)?;

    m.add_function(wrap_pyfunction!(stata_read::df_parse_dta_file, m)?)?;
    m.add_function(wrap_pyfunction!(stata_write::df_write_dta_file, m)?)?;

    m.add_function(wrap_pyfunction!(xpt_read::df_parse_xpt_file, m)?)?;
    m.add_function(wrap_pyfunction!(xpt_write::df_write_xpt_file, m)?)?;

    Ok(())
}

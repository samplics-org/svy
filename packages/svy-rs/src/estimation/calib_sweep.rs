// src/estimation/calib_sweep.rs
//! Calibration-aware score centering, the analogue of R's `svyrecvar`
//! `postStrata` loop.
//!
//! A calibrating adjustment pins quantities in the population, which removes
//! sampling variability the uncentred estimator still charges for. Feeding the
//! stratified PSU-total formula the *cell-centred* scores accounts for it:
//!
//! ```text
//! e_i = w*_i (z_i - z_c(i))     z_c = sum_{i in c} w_i z_i / sum_{i in c} w_i
//! ```
//!
//! Cell means use the PREVIOUS weights throughout. For a plain
//! poststratification that choice is immaterial -- the adjustment factor is
//! constant within a cell and cancels -- but it changes the answer wherever the
//! factor varies inside a sweep's cell: raking after the first margin, and GREG,
//! whose factor varies row by row. R uses the previous weights in every branch.
//!
//! Verified against survey 4.5 on apiclus1 before being written here: cells
//! 23.875658304826, GREG 3.295880912157, raking 23.745841047537, each matching
//! R to twelve significant figures, and a poststratified margin total giving
//! exactly zero.

use crate::regression::wols::solve_kxk;

/// Marks a row outside the adjustment: factor 1, no centring.
pub const OUT_OF_SCOPE: u32 = u32::MAX;

/// How many iterations R runs its raking sweep. Not a convergence loop -- a
/// fixed count, matched here so the two agree.
const RAKE_ITERATIONS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepKind {
    /// One cell coding: poststratification and standardization.
    Cells,
    /// One coding per margin, swept in turn.
    Raking,
    /// Weighted-least-squares residual against the auxiliary matrix.
    Greg,
}

#[derive(Debug, Clone)]
pub struct CalibSweep {
    pub kind: SweepKind,
    /// One code vector per margin. `Cells` and `Greg` use at most one.
    pub cells: Vec<Vec<u32>>,
    pub n_cells: Vec<usize>,
    /// Auxiliary matrix, column-major, for `Greg`. Non-finite entries mark a
    /// row outside the calibration.
    pub aux: Vec<Vec<f64>>,
    pub prev_w: Vec<f64>,
    /// Current weights, already carrying R's `psw == 0 & oldw == 0 -> psw = 1`
    /// guard so zero-weight rows cannot divide by zero.
    pub new_w: Vec<f64>,
    /// False when only the composition was pinned (`shares`), leaving the
    /// grand total's variability intact.
    pub pins_total: bool,
}

impl CalibSweep {
    pub fn new(
        kind: SweepKind,
        cells: Vec<Vec<u32>>,
        n_cells: Vec<usize>,
        aux: Vec<Vec<f64>>,
        prev_w: Vec<f64>,
        mut new_w: Vec<f64>,
        pins_total: bool,
    ) -> Self {
        // Estimation-time `where=` zeroes the ACTIVE weight but leaves the
        // record's previous-weight column alone. Left as is, an excluded row
        // would add nothing to a cell's numerator (its score is zero) but its
        // full previous weight to the denominator, shrinking every cell mean.
        // R never sees this because `subset()` zeroes both vectors; zeroing
        // prev_w alongside new_w reproduces that, and then the 0/0 guard drops
        // the row from the sweep entirely.
        let mut prev_w = prev_w;
        for i in 0..new_w.len() {
            if new_w[i] == 0.0 {
                prev_w[i] = 0.0;
                new_w[i] = 1.0;
            }
        }
        Self {
            kind,
            cells,
            n_cells,
            aux,
            prev_w,
            new_w,
            pins_total,
        }
    }

    /// Centre `x` in place.
    pub fn apply(&self, x: &mut [f64]) {
        self.apply_in_domain(x, None)
    }

    /// Centre `x` in place, restricted to a domain.
    ///
    /// Grouped estimation domain-masks the scores rather than subsetting the
    /// frame, so out-of-group rows arrive with score 0 but a live weight. Left
    /// in, they would contribute nothing to each cell's numerator and their
    /// full weight to its denominator, pulling every cell mean toward zero.
    /// R sidesteps this because `subset()` zeroes both weight vectors; here the
    /// domain is passed in and those rows are skipped outright, which is the
    /// same thing.
    pub fn apply_in_domain(&self, x: &mut [f64], domain: Option<&[bool]>) {
        if x.len() != self.new_w.len() {
            return;
        }
        if let Some(d) = domain {
            if d.len() != x.len() {
                return;
            }
        }
        match self.kind {
            SweepKind::Cells => {
                if self.cells.is_empty() {
                    return;
                }
                if self.pins_total {
                    self.sweep_cells(x, 0, domain);
                } else {
                    self.sweep_shares(x, domain);
                }
            }
            SweepKind::Raking => {
                for _ in 0..RAKE_ITERATIONS {
                    for m in 0..self.cells.len() {
                        self.sweep_raking(x, m, domain);
                    }
                }
            }
            SweepKind::Greg => self.sweep_greg(x, domain),
        }
    }

    /// Ordinary poststratification: subtract the previous-weight cell mean.
    fn sweep_cells(&self, x: &mut [f64], margin: usize, domain: Option<&[bool]>) {
        let codes = &self.cells[margin];
        let k = self.n_cells[margin];
        let (mut num, mut den) = (vec![0.0; k], vec![0.0; k]);
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE || domain.is_some_and(|d| !d[i]) {
                continue;
            }
            let c = c as usize;
            num[c] += x[i] * self.prev_w[i] / self.new_w[i];
            den[c] += self.prev_w[i];
        }
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE || domain.is_some_and(|d| !d[i]) {
                continue;
            }
            let c = c as usize;
            if den[c] != 0.0 {
                x[i] -= (num[c] / den[c]) * self.new_w[i];
            }
        }
    }

    /// `shares` pin the composition but not the level.
    ///
    /// The constraints are `sum_{i in c} w*_i = s_c * sum_i w*_i`, so the pinned
    /// directions are `u_c = delta_c - s_c`, which sum to zero and therefore
    /// span k-1 dimensions rather than k. Projecting those out is exactly the
    /// GREG residual against `u_1..u_{k-1}`, so this reuses that solver rather
    /// than being a sweep of its own.
    ///
    /// `s_c` is not carried on the record because it does not need to be: the
    /// achieved composition IS the shares, by construction.
    fn sweep_shares(&self, x: &mut [f64], domain: Option<&[bool]>) {
        let codes = &self.cells[0];
        let k = self.n_cells[0];
        if k < 2 {
            return;
        }
        let mut cell_w = vec![0.0; k];
        let mut total = 0.0;
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE || domain.is_some_and(|d| !d[i]) {
                continue;
            }
            cell_w[c as usize] += self.new_w[i];
            total += self.new_w[i];
        }
        if total == 0.0 {
            return;
        }
        let shares: Vec<f64> = cell_w.iter().map(|w| w / total).collect();

        // One column per cell but the last: they are linearly dependent.
        let mut aux = vec![vec![0.0; x.len()]; k - 1];
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE {
                continue;
            }
            for (j, col) in aux.iter_mut().enumerate() {
                col[i] = f64::from(c as usize == j) - shares[j];
            }
        }
        self.wls_residual(x, &aux, domain);
    }

    /// R rakes with the UNWEIGHTED group mean of `x / w`, unlike the
    /// poststratification branch. Matched deliberately.
    fn sweep_raking(&self, x: &mut [f64], margin: usize, domain: Option<&[bool]>) {
        let codes = &self.cells[margin];
        let k = self.n_cells[margin];
        let (mut sum, mut cnt) = (vec![0.0; k], vec![0.0f64; k]);
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE || domain.is_some_and(|d| !d[i]) {
                continue;
            }
            let c = c as usize;
            sum[c] += x[i] / self.new_w[i];
            cnt[c] += 1.0;
        }
        for i in 0..x.len() {
            let c = codes[i];
            if c == OUT_OF_SCOPE || domain.is_some_and(|d| !d[i]) {
                continue;
            }
            let c = c as usize;
            if cnt[c] != 0.0 {
                x[i] -= (sum[c] / cnt[c]) * self.new_w[i];
            }
        }
    }

    /// GREG: `w*(z - X b)` with `b` the WLS fit of `z` on `X` weighted by the
    /// PREVIOUS weights. R stores `qr(X * sqrt(oldw))` and `w = g * sqrt(oldw)`,
    /// which unwinds to exactly this; using the calibrated weights instead is
    /// close but wrong (3.2631 against 3.2959 on apiclus1).
    fn sweep_greg(&self, x: &mut [f64], domain: Option<&[bool]>) {
        self.wls_residual(x, &self.aux, domain)
    }

    /// `w*(z - X b)` with `b` the WLS fit of `z` on `X` weighted by prev_w.
    fn wls_residual(&self, x: &mut [f64], aux: &[Vec<f64>], domain: Option<&[bool]>) {
        let k = aux.len();
        if k == 0 {
            return;
        }
        let n = x.len();
        let (mut a, mut b) = (vec![0.0; k * k], vec![0.0; k]);
        for i in 0..n {
            let w = self.prev_w[i];
            if w == 0.0
                || domain.is_some_and(|d| !d[i])
                || !aux.iter().all(|col| col[i].is_finite())
            {
                continue;
            }
            let z = x[i] / self.new_w[i];
            for p in 0..k {
                let xp = aux[p][i];
                b[p] += w * xp * z;
                for q in 0..k {
                    a[p * k + q] += w * xp * aux[q][i];
                }
            }
        }
        let Some(beta) = solve_kxk(&a, &b, k) else {
            return;
        };
        for i in 0..n {
            if !aux.iter().all(|col| col[i].is_finite()) {
                continue;
            }
            let mut fit = 0.0;
            for p in 0..k {
                fit += aux[p][i] * beta[p];
            }
            x[i] -= fit * self.new_w[i];
        }
    }
}

// ============================================================================
// Construction from a DataFrame
// ============================================================================

use polars::prelude::*;

/// Column names naming a weight-adjustment record, as passed from Python.
#[derive(Debug, Clone, Default)]
pub struct CalibSpec {
    pub kind: String,
    pub cells_cols: Vec<String>,
    pub aux_cols: Vec<String>,
    pub prev_wgt_col: String,
    pub new_wgt_col: String,
    pub pins_total: bool,
}

fn codes_from(df: &DataFrame, name: &str) -> PolarsResult<(Vec<u32>, usize)> {
    let col = df.column(name)?;
    // Cells are written as Int32; casting to Int64 first would allocate a
    // second full-length array before the collect below.
    let s = if matches!(col.dtype(), DataType::Int32) {
        col.clone()
    } else {
        col.cast(&DataType::Int32)?
    };
    let ca = s.i32()?;
    let mut out = Vec::with_capacity(ca.len());
    let mut max = 0i32;
    for v in ca.iter() {
        match v {
            // Null marks a row the adjustment never touched.
            None => out.push(OUT_OF_SCOPE),
            Some(c) if c < 0 => out.push(OUT_OF_SCOPE),
            Some(c) => {
                if c > max {
                    max = c;
                }
                out.push(c as u32)
            }
        }
    }
    Ok((out, (max + 1) as usize))
}

fn floats_from(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let col = df.column(name)?;
    // Weight columns are Float64 and complete in every path that reaches here,
    // so take the memcpy rather than an option-yielding iterator collect: this
    // runs over the full frame once per estimation call, per column.
    if matches!(col.dtype(), DataType::Float64) {
        let ca = col.f64()?;
        if ca.null_count() == 0 {
            if let Ok(sl) = ca.cont_slice() {
                return Ok(sl.to_vec());
            }
        }
    }
    let s = col.cast(&DataType::Float64)?;
    Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// Build a sweep, or `None` when the record is provenance-only, its columns are
/// gone, or the active weight no longer matches what the record describes.
///
/// Returning `None` is the documented fallback: variance reverts to treating
/// weights as fixed, which is what it does today, rather than centring against
/// a structure that may no longer hold.
pub fn build_calib_sweep(df: &DataFrame, spec: &CalibSpec) -> Option<CalibSweep> {
    let kind = match spec.kind.as_str() {
        "poststratification" | "standardization" => SweepKind::Cells,
        "raking" => SweepKind::Raking,
        "calibration" => SweepKind::Greg,
        _ => return None,
    };

    let prev_w = floats_from(df, &spec.prev_wgt_col).ok()?;
    let new_w = floats_from(df, &spec.new_wgt_col).ok()?;

    let (mut cells, mut n_cells) = (Vec::new(), Vec::new());
    for name in &spec.cells_cols {
        let (c, k) = codes_from(df, name).ok()?;
        cells.push(c);
        n_cells.push(k);
    }

    let mut aux = Vec::new();
    for name in &spec.aux_cols {
        aux.push(floats_from(df, name).ok()?);
    }

    match kind {
        SweepKind::Greg if aux.is_empty() => return None,
        SweepKind::Cells | SweepKind::Raking if cells.is_empty() => return None,
        _ => {}
    }

    Some(CalibSweep::new(
        kind,
        cells,
        n_cells,
        aux,
        prev_w,
        new_w,
        spec.pins_total,
    ))
}

/// Centre a score column, for callers that hold `Float64Chunked` rather than a
/// slice (the categorical estimators, which hand score columns to
/// `taylor_variance` / `taylor_variance_matrix`).
///
/// Sweeping at the call site means those variance functions stay exactly as
/// they are: the scores they receive are already centred.
pub fn sweep_scores(
    scores: &Float64Chunked,
    calib: Option<&CalibSweep>,
    domain: Option<&[bool]>,
) -> Float64Chunked {
    match calib {
        None => scores.clone(),
        Some(c) => {
            let mut v: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            c.apply_in_domain(&mut v, domain);
            Float64Chunked::from_slice(scores.name().clone(), &v)
        }
    }
}

# svy_rs/__init__.py
"""
svy-rs: Internal Rust extension for the svy package.
WARNING: This is an internal package. Do not depend on it directly.
"""

from svy_rs._internal import (
    # Taylor linearization
    taylor_mean,
    taylor_total,
    taylor_ratio,
    taylor_prop,
    taylor_median,
    # Replication
    replicate_mean,
    replicate_total,
    replicate_ratio,
    replicate_prop,
    replicate_median,
    # Regression
    fit_glm_rs,
    # Categorical
    ttest_rs,
    ranktest_rs,
    tabulate_rs,
    # Weighting
    rake,
    adjust_nr,
    normalize,
    poststratify,
    poststratify_factor,
    calibrate,
    calibrate_by_domain,
    calibrate_parallel,
    trim_weights,
    trim_weights_matrix,
    create_brr_wgts,
    create_jk_wgts,
    create_bootstrap_wgts,
    create_sdr_wgts,
    # Sampling
    select_srs_rs,
    select_pps_rs,
)

__version__ = "0.8.0"

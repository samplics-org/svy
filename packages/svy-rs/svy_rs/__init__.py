# svy_rs/__init__.py
"""
svy-rs: Internal Rust extension for the svy package.
WARNING: This is an internal package. Do not depend on it directly.
"""

from svy_rs._internal import (
    adjust_nr,
    calibrate,
    calibrate_by_domain,
    calibrate_parallel,
    create_bootstrap_wgts,
    create_brr_wgts,
    create_jk_wgts,
    create_sdr_wgts,
    # Regression
    fit_glm_rs,
    normalize,
    poststratify,
    poststratify_factor,
    # Weighting
    rake,
    ranktest_rs,
    # Replication
    replicate_mean,
    replicate_median,
    replicate_prop,
    replicate_ratio,
    replicate_total,
    select_pps_rs,
    # Sampling
    select_srs_rs,
    tabulate_rs,
    # Taylor linearization
    taylor_mean,
    taylor_mean_multi,
    taylor_median,
    taylor_median_multi,
    taylor_prop,
    taylor_prop_multi,
    taylor_ratio,
    taylor_ratio_multi,
    taylor_total,
    taylor_total_multi,
    trim_weights,
    trim_weights_matrix,
    # Categorical
    ttest_rs,
)

__version__ = "0.9.0"

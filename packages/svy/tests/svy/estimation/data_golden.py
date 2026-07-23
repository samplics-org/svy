# tests/svy/estimation/data_golden.py

# ==============================================================================
# BRR (Balanced Repeated Replication) - 8 Replicates, df=7
#
# Regenerated 2026-07-23 with R survey 4.5 after fixing the fixture
# generator's Hadamard matrix (svy_test_data_bbr.py): the original "H8"
# was H(4) stacked twice, whose all-ones first column kept the same PSU
# of stratum S1 in every replicate (unbalanced half-samples). The
# corrected fixture uses columns 1..4 of the Sylvester H(8). Point
# estimates are unchanged (data columns identical); SEs changed.
#
# R: svrepdesign(repweights=brr_1..8, weights=weight, type="BRR",
#    combined.weights=TRUE) on the conftest-prepared frame (drop_nulls,
#    low_income = income < 40000, hh_size = id %% 4 + 1); svymean /
#    svytotal / svyratio (+ svyby ~educ); CIs are t(7); proportions use
#    logit-transformed t(7) intervals.
# ==============================================================================
BRR = {
    "mean_overall": {"est": 54687.65, "se": 1038.5553, "lci": 52231.86, "uci": 57143.44},
    "mean_educ": {
        "High": {"est": 53927.07, "se": 1271.7015, "lci": 50919.98, "uci": 56934.17},
        "Low": {"est": 53918.5, "se": 1153.6354, "lci": 51190.58, "uci": 56646.41},
        "Med": {"est": 55315.36, "se": 1446.7566, "lci": 51894.32, "uci": 58736.4},
    },
    "total_overall": {"est": 23.14286, "se": 2.8326446, "lci": 16.444724, "uci": 29.841004},
    "total_educ": {
        "High": {"est": 2.032223, "se": 1.4504921, "lci": -1.397646, "uci": 5.462092},
        "Low": {"est": 4.136101, "se": 2.6170633, "lci": -2.05227, "uci": 10.324472},
        "Med": {"est": 16.97454, "se": 6.3829818, "lci": 1.881186, "uci": 32.067893},
    },
    "prop_overall": {
        0: {"est": 0.9092026, "se": 0.0124768, "lci": 0.8750706, "uci": 0.9347052},
        1: {"est": 0.0907974, "se": 0.0124768, "lci": 0.0652948, "uci": 0.1249294},
    },
    "prop_educ": {
        "High": {
            0: {"est": 0.9655841, "se": 0.0234257, "lci": 0.84122, "uci": 0.9933144},
            1: {"est": 0.0344159, "se": 0.0234257, "lci": 0.0066856, "uci": 0.15878},
        },
        "Low": {
            0: {"est": 0.9259446, "se": 0.0567244, "lci": 0.6387408, "uci": 0.9888168},
            1: {"est": 0.0740554, "se": 0.0567244, "lci": 0.0111832, "uci": 0.3612592},
        },
        "Med": {
            0: {"est": 0.8787396, "se": 0.0370274, "lci": 0.7611258, "uci": 0.9427965},
            1: {"est": 0.1212604, "se": 0.0370274, "lci": 0.0572035, "uci": 0.2388742},
        },
    },
    "ratio_overall": {"est": 21945.59, "se": 494.6305, "lci": 20775.98, "uci": 23115.21},
    "ratio_educ": {
        "High": {"est": 21821.85, "se": 1041.6098, "lci": 19358.83, "uci": 24284.86},
        "Low": {"est": 21370.63, "se": 1303.562, "lci": 18288.19, "uci": 24453.06},
        "Med": {"est": 22230.04, "se": 849.5388, "lci": 20221.2, "uci": 24238.88},
    },
}

# ==============================================================================
# BOOTSTRAP - 20 Replicates, df=19 (Inferred)
# ==============================================================================
BOOTSTRAP = {
    "mean_overall": {"est": 54687.65, "se": 925.9494, "lci": 52749.61, "uci": 56625.68},
    "mean_educ": {
        "High": {"est": 53927.07, "se": 1163.337, "lci": 51492.18, "uci": 56361.97},
        "Low": {"est": 53918.5, "se": 1026.865, "lci": 51769.24, "uci": 56067.75},
        "Med": {"est": 55315.36, "se": 1303.115, "lci": 52587.91, "uci": 58042.81},
    },
    "total_overall": {"est": 23.14286, "se": 2.598292, "lci": 17.70458, "uci": 28.58115},
    "total_educ": {
        "High": {"est": 2.032223, "se": 1.436104, "lci": -0.973577, "uci": 5.038023},
        "Low": {"est": 4.136101, "se": 2.789945, "lci": -1.70332, "uci": 9.975522},
        "Med": {"est": 16.97454, "se": 6.374537, "lci": 3.63248, "uci": 30.3166},
    },
    "prop_overall": {
        0: {"est": 0.9092026, "se": 0.0106685, "lci": 0.8842647, "uci": 0.9291973},
        1: {"est": 0.0907974, "se": 0.0106686, "lci": 0.0708027, "uci": 0.1157353},
    },
    "prop_educ": {
        "High": {
            0: {"est": 0.9655841, "se": 0.0305551, "lci": 0.8037279, "uci": 0.9948247},
            1: {"est": 0.0344159, "se": 0.0305551, "lci": 0.0051753, "uci": 0.1962721},
        },
        "Low": {
            0: {"est": 0.9259446, "se": 0.0692748, "lci": 0.6014473, "uci": 0.9904395},
            1: {"est": 0.0740554, "se": 0.0692748, "lci": 0.0095605, "uci": 0.3985526},
        },
        "Med": {
            0: {"est": 0.8787396, "se": 0.0356997, "lci": 0.7823341, "uci": 0.9359427},
            1: {"est": 0.1212604, "se": 0.0356997, "lci": 0.0640573, "uci": 0.2176659},
        },
    },
    "ratio_overall": {"est": 21945.59, "se": 449.7754, "lci": 21004.2, "uci": 22886.99},
    "ratio_educ": {
        "High": {"est": 21821.85, "se": 887.747, "lci": 19963.77, "uci": 23679.92},
        "Low": {"est": 21370.63, "se": 1102.425, "lci": 19063.23, "uci": 23678.03},
        "Med": {"est": 22230.04, "se": 807.1265, "lci": 20540.7, "uci": 23919.37},
    },
}

# ==============================================================================
# JACKKNIFE (JKn) - 8 Replicates, df=7
# ==============================================================================
JACKKNIFE = {
    "mean_overall": {"est": 54687.65, "se": 1334.084, "lci": 51533.04, "uci": 57842.26},
    "mean_educ": {
        "High": {"est": 53927.07, "se": 1668.563, "lci": 49981.55, "uci": 57872.6},
        "Low": {"est": 53918.5, "se": 1540.471, "lci": 50275.86, "uci": 57561.13},
        "Med": {"est": 55315.36, "se": 1877.722, "lci": 50875.25, "uci": 59755.46},
    },
    "total_overall": {"est": 23.14286, "se": 3.747237, "lci": 14.28206, "uci": 32.00367},
    "total_educ": {
        "High": {"est": 2.032223, "se": 1.918821, "lci": -2.505067, "uci": 6.569513},
        "Low": {"est": 4.136101, "se": 3.462049, "lci": -4.050344, "uci": 12.32255},
        "Med": {"est": 16.97454, "se": 8.443891, "lci": -2.992091, "uci": 36.94117},
    },
    "prop_overall": {
        0: {"est": 0.9092026, "se": 0.0159608, "lci": 0.8637465, "uci": 0.9405379},
        1: {"est": 0.0907974, "se": 0.0159608, "lci": 0.0594621, "uci": 0.1362536},
    },
    "prop_educ": {
        "High": {
            0: {"est": 0.9655841, "se": 0.0333861, "lci": 0.722838, "uci": 0.9966978},
            1: {"est": 0.0344159, "se": 0.0333862, "lci": 0.0033022, "uci": 0.2771624},
        },
        "Low": {
            0: {"est": 0.9259446, "se": 0.0797247, "lci": 0.4444193, "uci": 0.9949094},
            1: {"est": 0.0740554, "se": 0.0797247, "lci": 0.0050906, "uci": 0.5555805},
        },
        "Med": {
            0: {"est": 0.8787396, "se": 0.0498466, "lci": 0.7056577, "uci": 0.9563412},
            1: {"est": 0.1212604, "se": 0.0498466, "lci": 0.0436588, "uci": 0.2943424},
        },
    },
    "ratio_overall": {"est": 21945.59, "se": 648.1116, "lci": 20413.05, "uci": 23478.14},
    "ratio_educ": {
        "High": {"est": 21821.85, "se": 1391.082, "lci": 18532.46, "uci": 25111.23},
        "Low": {"est": 21370.63, "se": 1381.182, "lci": 18104.65, "uci": 24636.6},
        "Med": {"est": 22230.04, "se": 1118.7, "lci": 19584.73, "uci": 24875.34},
    },
}

# Source: scripts/verify_where_{brr,bootstrap,jackknife}.R
# Domain: educ %in% c("Med", "High"), sex != "None"
# Threshold: low_income = as.integer(income < 40000)
# Reference: srvyr::filter() for domain; svyciprop(method="logit") with R's
#   default df (degf(dsgn) on the filtered design — NOT n_reps-1).
#
# Each scenario reports degf(dsgn) for reference:
#   BRR (8 reps):       degf(dsgn) = 3
#   Bootstrap (20 reps): degf(dsgn) = 4
#   Jackknife (8 reps):  degf(dsgn) = 4
#
# Confidence intervals for non-prop estimators use df = n_reps - 1
# (matching Python's convention of n_reps - 1 returned from Rust).

# Regenerated 2026-07-23 alongside the BRR block above (corrected
# Sylvester H(8) fixture; see that block's provenance note). R survey
# 4.5: subset(rd, educ %in% c("Med","High")) on the sex != "None"
# frame; means/totals/ratios t(7); prop_overall via svyciprop (logit,
# degf(subset) = 3); prop_by_sex via svyby + svyciprop vartype="ci".
WHERE_BRR = {
    "mean_overall": {
        "est": 54944.21712125,
        "se": 1157.5829811,
        "lci": 52206.968331,
        "uci": 57681.465912,
    },
    "mean_by_sex": {
        "Female": {
            "est": 56999.8198660894,
            "se": 1821.4183586,
            "lci": 52692.849843,
            "uci": 61306.789889,
        },
        "Male": {
            "est": 52801.4188536413,
            "se": 976.1935616,
            "lci": 50493.087884,
            "uci": 55109.749824,
        },
    },
    "total_overall": {
        "est": 19.00676263288,
        "se": 5.1190965,
        "lci": 6.902023,
        "uci": 31.111502,
    },
    "ratio_overall": {
        "est": 22091.1380445766,
        "se": 637.0740095,
        "lci": 20584.697392,
        "uci": 23597.578697,
    },
    "prop_overall": {
        0: {
            "est": 0.9037317804,
            "se": 0.0249428064,
            "lci": 0.806613334,
            "uci": 0.9548099743,
        },
        1: {
            "est": 0.0962682196,
            "se": 0.0249428064,
            "lci": 0.0451900257,
            "uci": 0.193386666,
        },
    },
    "prop_by_sex": {
        "Female": {
            "est": 0.0481956286694167,
            "se": None,  # svyby + svyciprop doesn't return SE in vartype="ci"
            "lci": 0.01260535,
            "uci": 0.1672514,
        },
        "Male": {
            "est": 0.1463799739724606,
            "se": None,
            "lci": 0.07452247,
            "uci": 0.2674985,
        },
    },
}

WHERE_BOOTSTRAP = {
    "mean_overall": {
        "est": 54944.21712125,
        "se": 1033.62907895112,
        "lci": 52780.8065956658,
        "uci": 57107.627646827,
    },
    "mean_by_sex": {
        "Female": {
            "est": 56999.8198660894,
            "se": 1538.85374252550,
            "lci": 53778.9619667673,
            "uci": 60220.6777654115,
        },
        "Male": {
            "est": 52801.4188536413,
            "se": 1153.79255065333,
            "lci": 50386.5032913268,
            "uci": 55216.3344159558,
        },
    },
    "total_overall": {
        "est": 19.00676263288,
        "se": 5.08090559770398,
        "lci": 8.3723049987042,
        "uci": 29.6412202670487,
    },
    "ratio_overall": {
        "est": 22091.1380445766,
        "se": 592.298784280094,
        "lci": 20851.4424416815,
        "uci": 23330.8336474716,
    },
    "prop_overall": {
        0: {
            "est": 0.903731780369117,
            "se": 0.0203245796513125,
            "lci": 0.825846758245603,
            "uci": 0.94893859211738,
        },
        1: {
            "est": 0.0962682196308831,
            "se": 0.0203245796513125,
            "lci": 0.0510614078826204,
            "uci": 0.174153241754397,
        },
    },
    "prop_by_sex": {
        "Female": {
            "est": 0.0481956286694167,
            "se": None,
            "lci": 0.0144218887037830,
            "uci": 0.149096990927130,
        },
        "Male": {
            "est": 0.1463799739724606,
            "se": None,
            "lci": 0.0729794504337909,
            "uci": 0.271947746403437,
        },
    },
}

WHERE_JACKKNIFE = {
    "mean_overall": {
        "est": 54944.21712125,
        "se": 1504.93310823307,
        "lci": 51385.6157964935,
        "uci": 58502.8184459992,
    },
    "mean_by_sex": {
        "Female": {
            "est": 56999.8198660894,
            "se": 2359.07618111802,
            "lci": 51421.4911168628,
            "uci": 62578.1486153160,
        },
        "Male": {
            "est": 52801.4188536413,
            "se": 1249.39575070365,
            "lci": 49847.0673616905,
            "uci": 55755.7703455921,
        },
    },
    "total_overall": {
        "est": 19.00676263288,
        "se": 6.77192808810104,
        "lci": 2.99369724571038,
        "uci": 35.0198280200425,
    },
    "ratio_overall": {
        "est": 22091.1380445766,
        "se": 831.130828542514,
        "lci": 20125.8259311585,
        "uci": 24056.4501579946,
    },
    "prop_overall": {
        0: {
            "est": 0.903731780369117,
            "se": 0.0330679652280778,
            "lci": 0.761818704628179,
            "uci": 0.964977438333229,
        },
        1: {
            "est": 0.0962682196308831,
            "se": 0.0330679652280777,
            "lci": 0.0350225616667711,
            "uci": 0.238181295371821,
        },
    },
    "prop_by_sex": {
        "Female": {
            "est": 0.0481956286694167,
            "se": None,
            "lci": 0.00935382188309996,
            "uci": 0.213558091193017,
        },
        "Male": {
            "est": 0.1463799739724606,
            "se": None,
            "lci": 0.05743100290223044,
            "uci": 0.325516102712977,
        },
    },
}

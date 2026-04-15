# src/svy/core/enumerations.py
"""Provides the custom enums used throughout the modules."""

from enum import Enum, StrEnum, unique


@unique
class CaseStyle(StrEnum):
    SNAKE = "Snake"
    CAMEL = "Camel"
    PASCAL = "Pascal"
    KEBAB = "Kebab"


@unique
class DistFamily(StrEnum):
    GAUSSIAN = "Gaussian"
    BINOMIAL = "Binomial"
    POISSON = "Poisson"
    GAMMA = "Gamma"
    INVERSE_GAUSSIAN = "InverseGaussian"
    # Phase 2:
    # NEG_BINOMIAL = "Negative Binomial"
    # BETA = "Beta"


# @unique
# class EditMethod(StrEnum):
#     BOTTOM_CODING = "Bottom Code"
#     TOP_CODING = "Top Code"
#     BOTTOM_AND_TOP_CODING = "Top and Bottom Code"
#     RECODING = "Recode"
#     CATEGORIZING = "Categorize"
#     RENAMING = "Rename"
#     CLEANING_NAMES = "Cleaning Names"


@unique
class EstimationMethod(StrEnum):
    # Standard variance estimation (Linearization)
    TAYLOR = "Taylor"

    # Replicate variance estimation methods
    BRR = "BRR"
    BOOTSTRAP = "Bootstrap"
    JACKKNIFE = "Jackknife"
    SDR = "SDR"  # Successive Difference Replication (e.g. used by ACS)


@unique
class FitMethod(StrEnum):
    # OLS = "OLS"
    # WLS = "WLS"
    # FH = "FH"
    ML = "ML"
    REML = "REML"


@unique
class MeanVarMode(StrEnum):
    EQUAL_VAR = "Equal Variances"
    """Assume σ1 = σ2 (common variance)."""

    UNEQUAL_VAR = "Unequal Variances"
    """Allow σ1 ≠ σ2 (Welch-type); supports optimal allocation."""


@unique
class ModelType(StrEnum):
    LINEAR = "Linear"  # Gaussian + Identity
    LOGISTIC = "Logistic"  # Binomial + Logit
    POISSON = "Poisson"  # Poisson + Log
    GAMMA = "Gamma"  # Gamma + Log (or Inverse)


@unique
class MetadataSource(StrEnum):
    """Track where variable metadata originated from."""

    INFERRED = "inferred"  # Auto-detected from data types/values
    SCHEMA = "schema"  # From explicit Schema definition
    QUESTIONNAIRE = "questionnaire"  # From Questionnaire definition
    IMPORTED = "imported"  # From external file (SPSS, Stata, etc.)
    USER = "user"  # Manually set by user at runtime


@unique
class MissingKind(StrEnum):
    """
    Semantic classification of missing values.

    These kinds help distinguish between different reasons for missingness,
    which can inform analysis decisions (e.g., imputation strategy).

    User-generated (typically MNAR):
        DONT_KNOW - Respondent doesn't know the answer
        REFUSED - Respondent refused to answer
        NO_ANSWER - No answer provided (ambiguous cause)

    System/Design-generated (typically MAR/MCAR):
        SYSTEM - System-generated missing (data processing issue)
        SKIPPED - Question skipped due to routing/skip logic
        NOT_APPLICABLE - Question not applicable to respondent
        STRUCTURAL - Missing by study design (e.g., split questionnaire)
    """

    # User-generated missing (typically MNAR)
    DONT_KNOW = "dnk"  # -> is typically MNAR
    REFUSED = "refused"  # -> is typically MNAR
    NO_ANSWER = "no_answer"  # -> could be any of MNAR, MAR, or MCAR

    # System/design-generated missing (typically MAR/MCAR)
    STRUCTURAL = "struct"  # -> is typical MAR e.g. SKIPPED and N/A
    SYSTEM = "system"  # -> is typically MCAR


@unique
class MissingMechanism(StrEnum):
    MCAR = "Completely Missing At Random"
    MAR = "Missing At Random"
    MNAR = "Missing Not At Random"


@unique
class LetterCase(str, Enum):
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    ORIGINAL = "original"


@unique
class LinkFunction(StrEnum):
    IDENTITY = "identity"
    LOGIT = "logit"
    LOG = "log"
    INVERSE = "inverse"
    INVERSE_SQUARED = "inverse_squared"
    # PROBIT = "probit"  # Requires CDF implementation
    # CLOGLOG = "cloglog"


@unique
class MeasurementType(StrEnum):
    CONTINUOUS = "Numerical Continuous"
    DISCRETE = "Numerical Discrete"
    NOMINAL = "Categorical Nominal"
    ORDINAL = "Categorical Ordinal"
    STRING = "String"
    BOOLEAN = "Boolean"
    DATETIME = "Datetime"


@unique
class OnePropSizeMethod(StrEnum):
    WALD = "Wald"
    FLEISS = "Fleiss"
    WILSON = "Wilson"
    AGRESTI_COULL = "Agresti-Coull"
    CLOPPER_PEARSON = "Clopper-Pearson"


@unique
class PopParam(StrEnum):
    MEAN = "Mean"
    TOTAL = "Total"
    PROP = "Proportion"
    RATIO = "Ratio"
    MEDIAN = "Median"


@unique
class PPSMethod(StrEnum):
    BREWER = "PPS Brewer"
    # HV = "PPS Hanurav-Vijayan"
    MURPHY = "PPS Murphy"
    RS = "PPS Rao-Sampford"
    SYS = "PPS Systematic"
    WR = "PPS with replacement"


@unique
class PropVarMode(StrEnum):
    ALT_PROPS = "Alternative Proportions"
    """Variance under alternative: p1(1-p1), p2(1-p2)."""

    POOLED_PROP = "Pooled Proportion"
    """Variance under null with pooled p; conservative/classic."""

    MAX_VAR = "Maximum Variance"
    """Use 0.25 for each arm (p=0.5); very conservative."""


@unique
class QuantileMethod(StrEnum):
    LOWER = "Lower"
    HIGHER = "Higher"
    NEAREST = "Nearest"
    LINEAR = "Linear"
    MIDDLE = "Middle"


# @unique
# class RepMethod(StrEnum):
#     JACKKNIFE = "Jackknife"
#     BOOTSTRAP = "Bootstrap"
#     BRR = "BRR"


@unique
class RankScoreMethod(StrEnum):
    """Score function for design-based rank tests.

    KRUSKAL_WALLIS (WILCOXON in teh case of two groups) uses proportional ranks g(r) = r/N.
    VANDER_WAERDEN uses inverse-Normal scores g(r) = Φ⁻¹(r/N).
    MEDIAN uses indicator scores g(r) = I(r > N/2).
    """

    KRUSKAL_WALLIS = "Kruskal-Wallis"
    VANDER_WAERDEN = "vanderWaerden"
    MEDIAN = "Median"


@unique
class SelectMethod(StrEnum):
    SRS_WR = "SRS with replacement"
    SRS_WOR = "SRS without replacement"
    SRS_SYS = "Systematic"
    PPS_BREWER = "PPS Brewer"
    # PPS_HV = "PPS Hanurav-Vijayan"
    PPS_MURPHY = "PPS Murphy"
    PPS_RS = "PPS Rao-Sampford"
    PPS_SYS = "PPS Systematic"
    PPS_WR = "PPS with replacement"
    GRS = "General"


@unique
class SingletonHandling(Enum):
    ERROR = "error"  # Raise error if singletons exist
    CERTAINTY = "certainty"  # PSU→stratum, SSU/records→PSUs
    SKIP = "skip"  # Exclude from variance (data stays)
    COMBINE = "combine"  # Manual column value remapping
    COLLAPSE = "collapse"  # Merge into existing strata
    POOL = "pool"  # Combine singletons into pseudo-stratum
    SCALE = "scale"  # Post-hoc variance inflation (R's "average")
    CENTER = "center"  # Grand-mean centering (R's "adjust") - NotImplemented


# @unique
# class SizeOnePropMethod(StrEnum):
#     WALD = "Wald"
#     FLEISS = "Fleiss"


@unique
class TwoPropsSizeMethod(StrEnum):
    WALD = "Wald (closed-form)"
    NEWCOMBE = "Newcombe"
    MIETTINEN_NURMINEN = "Miettinen-Nurminen"
    FARRINGTON_MANNING = "Farrington-Manning"


@unique
class TableType(StrEnum):
    ONE_WAY = "One-Way"
    TWO_WAY = "Two-Way"


@unique
class TableUnits(StrEnum):
    PROPORTION = "Proportion"
    PERCENT = "Percent"
    COUNT = "Count"


@unique
class TTestType(StrEnum):
    ONE_SAMPLE = "One-Sample"
    TWO_SAMPLE = "Two-Sample"

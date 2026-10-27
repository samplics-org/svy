# svy/errors/_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from ._base import SvyError, SvyWarning


@dataclass(eq=False)
class ModelError(SvyError):
    """
    Raised for statistical modeling failures: convergence issues,
    mathematical domain violations, singular matrices, or state issues.
    """

    _DEFAULT_CODE: ClassVar[str] = "MODEL_ERROR"

    # ---- State Errors -------------------------------------------------------

    @classmethod
    def not_fitted(
        cls,
        *,
        where: Optional[str],
        method: str = "predict",
        hint: Optional[str] = "Call .fit() on the estimator first.",
    ) -> "ModelError":
        return cls.ValueError(
            title="Model not fitted",
            detail=f"Cannot call '{method}' because the model has not been fitted yet.",
            code="MODEL_NOT_FITTED",
            where=where,
            hint=hint,
        )

    # ---- Mathematical / Domain Errors ---------------------------------------

    @classmethod
    def domain_violation(
        cls,
        *,
        where: Optional[str],
        family: str,
        violation: str,
        found_values: Any = None,
        hint: Optional[str] = None,
    ) -> "ModelError":
        """
        E.g. Negative values for Gamma, or non-binary for Binomial.
        """
        return cls.ValueError(
            title="Distribution domain violation",
            detail=f"The target data violates the requirements for family '{family}': {violation}.",
            code="DOMAIN_VIOLATION",
            where=where,
            param="family",
            expected=f"valid range for {family}",
            got=found_values,
            hint=hint or "Check your target variable or filter invalid rows.",
        )

    # ---- Linear Algebra / Optimization Errors -------------------------------

    @classmethod
    def singular_matrix(
        cls,
        *,
        where: Optional[str],
        rank: int,
        dim: int,
        hint: Optional[str] = "Check for perfect multicollinearity among predictors.",
    ) -> "ModelError":
        return cls.ValueError(
            title="Singular design matrix",
            detail="The design matrix is singular (not invertible); parameters cannot be estimated uniquely.",
            code="SINGULAR_MATRIX",
            where=where,
            expected=f"full rank ({dim})",
            got=f"rank {rank}",
            hint=hint,
        )

    @classmethod
    def convergence_failed(
        cls,
        *,
        where: Optional[str],
        iterations: int,
        tol: float,
        last_diff: float | None = None,
        hint: Optional[str] = "Increase max_iter or check for data issues.",
    ) -> "ModelError":
        return cls.RuntimeError(
            title="Convergence failed",
            detail=f"Optimization did not converge after {iterations} iterations.",
            code="CONVERGENCE_FAILED",
            where=where,
            param="max_iter",
            extra={"tolerance": tol, "last_diff": last_diff},
            hint=hint,
        )

    @classmethod
    def insufficient_dof(
        cls,
        *,
        where: Optional[str],
        n_obs: int,
        n_params: int,
        hint: Optional[str] = "Reduce the number of predictors or increase sample size.",
    ) -> "ModelError":
        return cls.ValueError(
            title="Insufficient degrees of freedom",
            detail=f"Number of observations ({n_obs}) is too small for the number of parameters ({n_params}).",
            code="INSUFFICIENT_DOF",
            where=where,
            expected=f"n > {n_params}",
            got=n_obs,
            hint=hint,
        )


@dataclass(eq=False)
class ModelWarning(SvyWarning):
    pass

# svy/errors/_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from ._base import SvyError, SvyWarning
from ._codes import Code


@dataclass(eq=False)
class ModelError(SvyError):
    """
    Raised for statistical modeling failures: convergence issues,
    mathematical domain violations, singular matrices, or state issues.
    """

    _DEFAULT_CODE: ClassVar[str] = Code.MODEL_ERROR

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
            code=Code.MODEL_NOT_FITTED,
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
            code=Code.DOMAIN_VIOLATION,
            where=where,
            param="family",
            expected=f"valid range for {family}",
            got=found_values,
            hint=hint or "Check your target variable or filter invalid rows.",
        )


@dataclass(eq=False)
class ModelWarning(SvyWarning):
    pass

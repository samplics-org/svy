from svy.errors.base_errors import SvyError


class SinglePSUError(SvyError):
    """Only one PSU in the stratum"""

    pass


class ProbError(SvyError):
    """Not a valid probability"""

    pass


class MethodError(SvyError):
    """Method not applicable"""

    pass


class CertaintyError(SvyError):
    """Method not applicable"""

    pass


class DimensionError(SvyError):
    """Method not applicable"""

    pass

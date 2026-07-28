class ProjError(RuntimeError):
    """Raised when a coordinate operation cannot be completed."""


class CRSError(ProjError):
    """Raised when a CRS definition is unsupported or invalid."""

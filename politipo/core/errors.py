"""
Core exception hierarchy for Politipo type conversion system.
"""

class PolitipoError(Exception):
    """Base exception for all Politipo-related errors."""
    __slots__ = ()  # Prevent attribute assignment


class ConversionError(PolitipoError):
    """Base class for conversion failures."""
    __slots__ = ()


class NoConversionPathError(ConversionError):
    """Raised when no conversion path exists between type systems."""
    __slots__ = ()


class ConstraintViolationError(ConversionError):
    """Raised when a value fails constraint validation."""
    __slots__ = ()


class RegistrationError(PolitipoError):
    """Raised when plugin registration fails."""
    __slots__ = () 
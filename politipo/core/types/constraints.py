from dataclasses import dataclass
from typing import Any, Dict, Type
from ..errors import ConstraintViolationError


@dataclass(frozen=True)
class MinValue:
    """Numeric minimum constraint."""
    value: float

    def __post_init__(self):
        """Validate value type during initialization."""
        if not isinstance(self.value, (int, float)):
            raise TypeError("MinValue requires numeric value")

    def validate(self, val: Any) -> bool:
        """
        Validate that a value meets the minimum constraint.
        
        Args:
            val: Value to validate
            
        Returns:
            True if value meets constraint
            
        Raises:
            ConstraintViolationError: If value is invalid or too small
        """
        try:
            numeric_val = float(val)
            if numeric_val < self.value:
                raise ConstraintViolationError(
                    f"Value {val} is less than minimum {self.value}"
                )
            return True
        except (TypeError, ValueError) as e:
            raise ConstraintViolationError(
                f"Invalid numeric value: {val}"
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraint to dictionary."""
        return {"type": "min_value", "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinValue":
        """Create constraint from dictionary."""
        if "value" not in data:
            raise ValueError("MinValue requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class MaxLength:
    """String/collection maximum length constraint."""
    value: int

    def __post_init__(self):
        """Validate value during initialization."""
        if not isinstance(self.value, int):
            raise TypeError("MaxLength requires integer value")
        if self.value < 0:
            raise ValueError("MaxLength must be non-negative")

    def validate(self, val: Any) -> bool:
        """
        Validate that a value meets the length constraint.
        
        Args:
            val: Value to validate
            
        Returns:
            True if value meets constraint
            
        Raises:
            ConstraintViolationError: If value is invalid or too long
        """
        try:
            if len(val) > self.value:
                raise ConstraintViolationError(
                    f"Length {len(val)} exceeds maximum {self.value}"
                )
            return True
        except TypeError as e:
            raise ConstraintViolationError(
                f"Value {val} does not support length checking"
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraint to dictionary."""
        return {"type": "max_length", "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaxLength":
        """Create constraint from dictionary."""
        if "value" not in data:
            raise ValueError("MaxLength requires 'value' parameter")
        return cls(value=data["value"])


# Registry for constraint deserialization with type safety
_CONSTRAINT_REGISTRY: Dict[str, Type[Any]] = {
    "min_value": MinValue,
    "max_length": MaxLength
}


def constraint_from_dict(data: Dict[str, Any]) -> Any:
    """
    Factory method for deserializing constraints.
    
    Args:
        data: Dictionary representation of constraint
        
    Returns:
        Instantiated constraint
        
    Raises:
        ValueError: If constraint type is unknown or data is invalid
    """
    try:
        constraint_type = data["type"]
        if constraint_type not in _CONSTRAINT_REGISTRY:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        return _CONSTRAINT_REGISTRY[constraint_type].from_dict(data)
    except KeyError as e:
        raise ValueError("Constraint data must include 'type' field") from e 
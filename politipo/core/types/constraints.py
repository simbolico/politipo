from dataclasses import dataclass
from typing import Any, Dict, Type, Union, Pattern as RegexPattern, List
from decimal import Decimal, InvalidOperation
import re
from ..errors import ConstraintViolationError
from .abstract import Constraint


@dataclass(frozen=True)
class MinValue(Constraint):
    """Numeric minimum constraint (inclusive)."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"MinValue requires numeric value, got {type(self.value)}")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else val
            if numeric_val < self.value:
                raise ConstraintViolationError(
                    f"Value {val} is less than minimum {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for MinValue: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "min_value", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinValue":
        if "value" not in data:
            raise ValueError("MinValue requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class MaxValue(Constraint):
    """Numeric maximum constraint (inclusive)."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"MaxValue requires numeric value, got {type(self.value)}")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else val
            if numeric_val > self.value:
                raise ConstraintViolationError(
                    f"Value {val} is greater than maximum {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for MaxValue: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "max_value", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaxValue":
        if "value" not in data:
            raise ValueError("MaxValue requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class MinLength(Constraint):
    """String/collection minimum length constraint."""
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int):
            raise TypeError(f"MinLength requires integer value, got {type(self.value)}")
        if self.value < 0:
            raise ValueError("MinLength must be non-negative")

    def validate(self, val: Any) -> bool:
        try:
            if len(val) < self.value:
                raise ConstraintViolationError(
                    f"Length {len(val)} is less than minimum {self.value}"
                )
            return True
        except TypeError as e:
            raise ConstraintViolationError(
                f"Value '{val}' does not support length checking for MinLength"
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "min_length", "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinLength":
        if "value" not in data:
            raise ValueError("MinLength requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class MaxLength(Constraint):
    """String/collection maximum length constraint."""
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int):
            raise TypeError(f"MaxLength requires integer value, got {type(self.value)}")
        if self.value < 0:
            raise ValueError("MaxLength must be non-negative")

    def validate(self, val: Any) -> bool:
        try:
            if len(val) > self.value:
                raise ConstraintViolationError(
                    f"Length {len(val)} exceeds maximum {self.value}"
                )
            return True
        except TypeError as e:
            raise ConstraintViolationError(
                f"Value '{val}' does not support length checking for MaxLength"
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "max_length", "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaxLength":
        if "value" not in data:
            raise ValueError("MaxLength requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class GreaterThan(Constraint):
    """Numeric greater than constraint (exclusive)."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"GreaterThan requires numeric value, got {type(self.value)}")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else val
            if numeric_val <= self.value:
                raise ConstraintViolationError(
                    f"Value {val} is not greater than {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for GreaterThan: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "greater_than", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GreaterThan":
        if "value" not in data:
            raise ValueError("GreaterThan requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class LessThan(Constraint):
    """Numeric less than constraint (exclusive)."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"LessThan requires numeric value, got {type(self.value)}")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else val
            if numeric_val >= self.value:
                raise ConstraintViolationError(
                    f"Value {val} is not less than {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for LessThan: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "less_than", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessThan":
        if "value" not in data:
            raise ValueError("LessThan requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class GreaterThanOrEqual(Constraint):
    """Numeric greater than or equal constraint (inclusive)."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"GreaterThanOrEqual requires numeric value, got {type(self.value)}")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else val
            if numeric_val < self.value:
                raise ConstraintViolationError(
                    f"Value {val} is less than {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for GreaterThanOrEqual: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "greater_than_or_equal", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GreaterThanOrEqual":
        if "value" not in data:
            raise ValueError("GreaterThanOrEqual requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class LessThanOrEqual(Constraint):
    """Numeric less than or equal constraint (inclusive)."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"LessThanOrEqual requires numeric value, got {type(self.value)}")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else val
            if numeric_val > self.value:
                raise ConstraintViolationError(
                    f"Value {val} is greater than {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for LessThanOrEqual: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "less_than_or_equal", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessThanOrEqual":
        if "value" not in data:
            raise ValueError("LessThanOrEqual requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class Pattern(Constraint):
    """String regex pattern constraint."""
    regex: Union[str, RegexPattern]
    _compiled_regex: RegexPattern = None

    def __post_init__(self):
        if not isinstance(self.regex, (str, RegexPattern)):
            raise TypeError(f"Pattern requires string or compiled regex, got {type(self.regex)}")
        try:
            object.__setattr__(self, '_compiled_regex', re.compile(self.regex) if isinstance(self.regex, str) else self.regex)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {self.regex}") from e

    def validate(self, val: Any) -> bool:
        if not isinstance(val, str):
            raise ConstraintViolationError(f"Pattern constraint requires string value, got {type(val)}")
        if not self._compiled_regex.match(val):
            raise ConstraintViolationError(
                f"Value '{val}' does not match pattern '{self.regex}'"
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "pattern", "regex": self.regex if isinstance(self.regex, str) else self.regex.pattern}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        if "regex" not in data:
            raise ValueError("Pattern requires 'regex' parameter")
        return cls(regex=data["regex"])


@dataclass(frozen=True)
class MultipleOf(Constraint):
    """Numeric multiple-of constraint."""
    value: Union[int, float, Decimal]

    def __post_init__(self):
        if not isinstance(self.value, (int, float, Decimal)):
            raise TypeError(f"MultipleOf requires numeric value, got {type(self.value)}")
        if self.value == 0:
            raise ValueError("MultipleOf constraint cannot be zero")

    def validate(self, val: Any) -> bool:
        try:
            numeric_val = Decimal(str(val)) if not isinstance(val, (int, float, Decimal)) else Decimal(val)
            multiple = Decimal(str(self.value))
            if (numeric_val % multiple) != 0:
                raise ConstraintViolationError(
                    f"Value {val} is not a multiple of {self.value}"
                )
            return True
        except (TypeError, ValueError, InvalidOperation) as e:
            raise ConstraintViolationError(f"Invalid numeric value for MultipleOf: {val}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "multiple_of", "value": float(self.value) if isinstance(self.value, Decimal) else self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultipleOf":
        if "value" not in data:
            raise ValueError("MultipleOf requires 'value' parameter")
        return cls(value=data["value"])


@dataclass(frozen=True)
class OneOf(Constraint):
    """Constraint that ensures a value is one of a set of allowed values."""
    allowed_values: List[Any]

    def __post_init__(self):
        if not isinstance(self.allowed_values, (list, tuple, set)):
            raise TypeError(f"OneOf requires a sequence of allowed values, got {type(self.allowed_values)}")
        if not self.allowed_values:
            raise ValueError("OneOf requires at least one allowed value")

    def validate(self, val: Any) -> bool:
        if val not in self.allowed_values:
            raise ConstraintViolationError(
                f"Value {val} is not one of the allowed values: {self.allowed_values}"
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "one_of", "allowed_values": list(self.allowed_values)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OneOf":
        if "allowed_values" not in data:
            raise ValueError("OneOf requires 'allowed_values' parameter")
        return cls(allowed_values=data["allowed_values"])


# Registry for constraint deserialization
_CONSTRAINT_REGISTRY: Dict[str, Type[Constraint]] = {
    "min_value": MinValue,
    "max_value": MaxValue,
    "min_length": MinLength,
    "max_length": MaxLength,
    "greater_than": GreaterThan,
    "less_than": LessThan,
    "greater_than_or_equal": GreaterThanOrEqual,
    "less_than_or_equal": LessThanOrEqual,
    "pattern": Pattern,
    "multiple_of": MultipleOf,
    "one_of": OneOf,
}


def constraint_from_dict(data: Dict[str, Any]) -> Constraint:
    """Factory method for deserializing constraints.
    
    Args:
        data: Dictionary containing constraint data. Must include a 'type' field
             identifying the constraint type, and any required parameters for
             that specific constraint type (usually 'value').
             
    Returns:
        An instance of the appropriate Constraint subclass.
        
    Raises:
        ValueError: If the data dictionary is missing required fields or contains
                   an unknown constraint type.
        TypeError: If the provided values are of incorrect types for the constraint.
    """
    try:
        if not isinstance(data, dict):
            raise TypeError(f"Expected dictionary, got {type(data)}")
            
        constraint_type = data.get("type")
        if not constraint_type:
            raise ValueError("Constraint data must include 'type' field")
            
        if constraint_type not in _CONSTRAINT_REGISTRY:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
            
        constraint_class = _CONSTRAINT_REGISTRY[constraint_type]
        
        # Make a copy to avoid modifying the original dict
        constraint_data = data.copy()
        del constraint_data["type"]  # Remove type key as it's not needed by from_dict
        
        try:
            return constraint_class.from_dict(constraint_data)
        except (TypeError, ValueError) as e:
            # Wrap constraint-specific errors with more context
            raise ValueError(f"Invalid data for constraint type '{constraint_type}': {e}") from e
            
    except KeyError as e:
        raise ValueError(f"Missing required field in constraint data: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors and provide context
        raise ValueError(f"Failed to create constraint from data: {e}") from e 
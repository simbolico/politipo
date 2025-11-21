from typing import Any

from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError
from politipo.core.types import CanonicalType


class ModelToDictStrategy(ConversionStrategy):
    """Strategy for converting Pydantic/SQLModel instances to dictionaries."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """Check if this strategy can handle the conversion."""
        return (
            source.kind == "model"
            and source.name in ("pydantic", "sqlmodel")
            and target.kind == "dict"
        )

    def convert(self, value: Any, context: ConversionContext) -> dict[str, Any]:
        """Convert a Pydantic/SQLModel instance to a dictionary."""
        if not hasattr(value, "model_dump") and not hasattr(value, "dict"):
            raise ConversionError(f"Expected Pydantic/SQLModel instance, got {type(value)}")

        try:
            if hasattr(value, "model_dump"):  # Pydantic v2
                return value.model_dump()
            else:  # Pydantic v1
                return value.dict()
        except Exception as e:
            raise ConversionError(f"Failed to convert model to dict: {str(e)}")

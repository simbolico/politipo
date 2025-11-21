from typing import Any

from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType


class ListToPolarsStrategy(ConversionStrategy):
    """Strategy for converting lists to Polars DataFrames."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """Check if this strategy can handle the conversion."""
        return source.kind == "list" and target.kind == "dataframe" and target.name == "polars"

    def convert(self, value: Any, context: ConversionContext) -> Any:
        """Convert a list to a Polars DataFrame."""
        try:
            import polars as pl
        except ImportError:
            raise PolitipoError("polars is required for ListToPolarsStrategy")

        if not isinstance(value, list):
            raise ConversionError(f"Expected list, got {type(value)}")

        if not value:
            return pl.DataFrame()

        # Handle list of dictionaries
        if isinstance(value[0], dict):
            return pl.DataFrame(value)

        # Handle list of model instances
        if hasattr(value[0], "model_dump"):  # Pydantic v2
            return pl.DataFrame([item.model_dump() for item in value])
        elif hasattr(value[0], "dict"):  # Pydantic v1
            return pl.DataFrame([item.dict() for item in value])

        # Handle list of primitive values
        return pl.DataFrame(value)

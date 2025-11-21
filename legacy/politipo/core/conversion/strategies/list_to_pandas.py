from typing import Any

from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType

# We might need ModelToDictStrategy if list contains models
from .model_to_dict import ModelToDictStrategy


class ListToPandasStrategy(ConversionStrategy):
    """Strategy for converting lists to pandas DataFrames."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """Check if this strategy can handle the conversion."""
        return source.kind == "list" and target.kind == "dataframe" and target.name == "pandas"

    def convert(self, value: Any, context: ConversionContext) -> Any:
        """Convert a list to a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise PolitipoError("pandas is required for ListToPandasStrategy")

        if not isinstance(value, list):
            raise ConversionError(f"Expected list, got {type(value)}")

        if not value:
            # Handle empty list: create empty DataFrame? Or DataFrame with columns from canonical type?
            # Let's create an empty DF for now. Column definition could be complex.
            return pd.DataFrame()

        first_item = value[0]
        item_source_canonical = context.source.params.get(
            "item_type"
        )  # Get canonical type of list items

        try:
            if isinstance(first_item, dict):
                # Assume list of dicts
                return pd.DataFrame(value)
            # Check if items are Pydantic/SQLModel instances
            # We need to determine the item type more reliably
            elif item_source_canonical and item_source_canonical.kind == "composite":
                # Assume list of models, convert each to dict first
                model_to_dict_strategy = ModelToDictStrategy()
                dict_list = []
                # Create a dummy context for the inner conversion (Model -> Dict)
                # Target for inner conversion is dict
                dict_canonical = CanonicalType(kind="container", name="dict")  # Simple dict target
                inner_context = ConversionContext(
                    source=item_source_canonical,  # Source is the model type
                    target=dict_canonical,
                    source_type_system=context.source_type_system,  # Assuming source system handles models
                    target_type_system=None,  # Target system not strictly needed for dict
                    strict=context.strict,
                )
                for item in value:
                    dict_list.append(model_to_dict_strategy.convert(item, inner_context))
                return pd.DataFrame(dict_list)
            elif hasattr(first_item, "model_dump"):  # Pydantic v2
                return pd.DataFrame([item.model_dump() for item in value])
            elif hasattr(first_item, "dict"):  # Pydantic v1
                return pd.DataFrame([item.dict() for item in value])
            else:
                # Fallback: Try creating DataFrame directly, might fail for complex objects
                try:
                    return pd.DataFrame(value)
                except Exception as df_err:
                    raise ConversionError(
                        f"Cannot convert list of {type(first_item)} to DataFrame. Items must be dicts or known models."
                    ) from df_err

        except Exception as e:
            # Catch pandas errors or internal conversion errors
            raise ConversionError(f"Failed to convert list to DataFrame: {e}") from e

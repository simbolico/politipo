from functools import lru_cache
from typing import Any, Dict, List, Type

from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType
from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.utils.pydantic import get_pydantic_version
from politipo.core.conversion.utils import create_model_instance_from_dict


class PolarsToListStrategy(ConversionStrategy):
    """Strategy for converting Polars DataFrame to list of model instances."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """Returns True if source is a DataFrame and target is a list of models."""
        return (
            source.kind == "composite" and source.name == "DataFrame" and
            target.kind == "container" and target.name == "list" and
            target.params.get("item_type", {}).kind == "composite"
        )

    def convert(self, value: Any, context: ConversionContext) -> List[Any]:
        """Convert a Polars DataFrame to a list of model instances."""
        try:
            import polars as pl
        except ImportError:
            raise ConversionError("polars is not installed")

        if not isinstance(value, pl.DataFrame):
            raise ConversionError(f"Expected Polars DataFrame, got {type(value)}")

        # Get the canonical type for the list item
        item_canonical_type = context.target.params.get("item_type")
        if not item_canonical_type:
            raise ConversionError("Target list type missing 'item_type' in params")

        # Get the target item's type system (likely the same as the list's target system)
        item_system = context.target_type_system
        if not item_system:
            raise ConversionError("Target type system not found in context for PolarsToListStrategy")

        try:
            model_class = item_system.from_canonical(item_canonical_type)
            if not isinstance(model_class, type):
                raise ConversionError(f"Target system did not return a valid class for list item {item_canonical_type.name}")
        except NotImplementedError:
            raise ConversionError(f"Target system '{item_system.name}' cannot reconstruct model from canonical type.")
        except Exception as e:
            raise ConversionError(f"Failed to get target model class for list item: {e}")

        # Convert DataFrame to list of dicts
        try:
            records = value.to_dicts()
        except Exception as e:
            raise ConversionError(f"Failed to convert Polars DataFrame to dicts: {e}")

        # Convert each dict to a model instance
        try:
            return [
                create_model_instance_from_dict(model_class, record, context.strict)
                for record in records
            ]
        except Exception as e:
            raise ConversionError(f"Failed to create model instances of {model_class.__name__}: {e}")

    def _create_model_instance(self, model_class: Type, data: Dict[str, Any], strict: bool) -> Any:
        """Helper to create a model instance from dict data."""
        try:
            # Use model_validate for v2 strict, model_construct for v2 non-strict
            # Use parse_obj for v1 strict (if available), **kwargs for v1 non-strict
            pyd_version = get_pydantic_version()
            if pyd_version == 2:
                if strict:
                    # model_validate performs validation
                    return model_class.model_validate(data)
                else:
                    # model_construct skips validation (faster)
                    # Filter data to only include fields defined in the model
                    valid_field_names = getattr(model_class, 'model_fields', {}).keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_field_names}
                    return model_class.model_construct(**filtered_data)
            else:  # Pydantic v1
                if strict and hasattr(model_class, 'parse_obj'):
                    # parse_obj performs validation
                    return model_class.parse_obj(data)
                else:
                    # Direct **kwargs instantiation, potentially filter for safety
                    valid_field_names = getattr(model_class, '__fields__', {}).keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_field_names}
                    return model_class(**filtered_data)

        except Exception as e:
            # This block might become less necessary with model_construct/filtering above,
            # but keep as a final safety net if direct **kwargs fails in v1 non-strict.
            if strict:
                # Re-raise if strict mode failed validation/construction
                raise ConversionError(f"Failed to create {model_class.__name__} instance: {e}") from e
            else:
                # If even filtering + construction failed in non-strict, raise with context
                raise ConversionError(f"Non-strict creation failed for {model_class.__name__} after filtering: {e}") from e 
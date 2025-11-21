from typing import Any

from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError
from politipo.core.types import CanonicalType
from politipo.core.utils.pydantic import get_pydantic_version


class DictToModelStrategy(ConversionStrategy):
    """Strategy for converting dictionaries to model instances."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """Returns True if source is a dict and target is a composite type."""
        return source.kind == "container" and source.name == "dict" and target.kind == "composite"

    def convert(self, value: Any, context: ConversionContext) -> Any:
        """Convert a dictionary to a model instance."""
        if not isinstance(value, dict):
            raise ConversionError(f"Expected dict, got {type(value)}")

        # Get model class using the target type system and canonical type
        if not context.target_type_system:
            raise ConversionError("Target type system not found in context for DictToModelStrategy")

        try:
            model_class = context.target_type_system.from_canonical(context.target)
            if not isinstance(model_class, type):
                raise ConversionError(
                    f"Target system did not return a valid class for {context.target.name}"
                )
        except NotImplementedError:
            raise ConversionError(
                f"Target system '{context.target_type_system.name}' cannot reconstruct model from canonical type."
            )
        except Exception as e:
            raise ConversionError(f"Failed to get target model class from canonical type: {e}")

        # Convert dict to model instance
        try:
            return self._create_model_instance(model_class, value, context.strict)
        except Exception as e:
            raise ConversionError(f"Failed to create model instance of {model_class.__name__}: {e}")

    def _create_model_instance(self, model_class: type, data: dict[str, Any], strict: bool) -> Any:
        """Helper to create a model instance from dict data."""
        try:
            # Use model_validate for v2 strict, model_construct for v2 non-strict
            # Use parse_obj for v1 strict (if available), **kwargs for v1 non-strict
            pyd_version = get_pydantic_version()

            # Filter data to only include fields defined in the model
            if pyd_version == 2:
                valid_field_names = getattr(model_class, "model_fields", {}).keys()
            else:  # Pydantic v1
                valid_field_names = getattr(model_class, "__fields__", {}).keys()
            filtered_data = {k: v for k, v in data.items() if k in valid_field_names}

            if pyd_version == 2:
                if strict:
                    # model_validate performs validation
                    return model_class.model_validate(filtered_data)
                else:
                    # model_construct skips validation (faster)
                    return model_class.model_construct(**filtered_data)
            else:  # Pydantic v1
                if strict and hasattr(model_class, "parse_obj"):
                    # parse_obj performs validation
                    return model_class.parse_obj(filtered_data)
                else:
                    # Direct **kwargs instantiation
                    return model_class(**filtered_data)

        except Exception as e:
            raise ConversionError(f"Failed to create {model_class.__name__} instance: {e}") from e

"""Utilities for model instantiation."""
from typing import Any, Dict, Type

from politipo.core.errors import ConversionError
from politipo.core.utils.pydantic import get_pydantic_version


def create_model_instance_from_dict(
    model_class: Type,
    data: Dict[str, Any],
    strict: bool
) -> Any:
    """
    Creates a Pydantic/SQLModel instance from dictionary data.

    Handles Pydantic v1/v2 differences and strict/non-strict modes.

    Args:
        model_class: The target model class (e.g., Pydantic BaseModel).
        data: The dictionary containing data for the model.
        strict: If True, performs full validation. If False, attempts
                faster creation (e.g., model_construct) and filters
                extraneous data.

    Returns:
        An instance of the model_class.

    Raises:
        ConversionError: If instantiation fails, especially in strict mode
                        or if non-strict fallback also fails.
    """
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
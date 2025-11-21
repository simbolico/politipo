from typing import Any

from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType

# Attempt to import SQLAlchemy components
try:
    from sqlalchemy.orm import DeclarativeMeta

    _SQLA_AVAILABLE = True
except ImportError:
    _SQLA_AVAILABLE = False
    DeclarativeMeta = None


class DictToSQLAlchemyModelStrategy(ConversionStrategy):
    """Strategy for converting dictionaries to SQLAlchemy model instances."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """
        Returns True if the source is a dict and the target is a SQLAlchemy model.
        Relies on metadata set by SQLAlchemyModelTypeSystem.
        """
        return (
            source.kind == "container"
            and source.name == "dict"
            and target.kind == "composite"
            and target.meta is not None
            and target.meta.get("origin_system") == "sqlalchemy_model"
        )

    def convert(self, value: Any, context: ConversionContext) -> Any:
        """
        Convert a dictionary to an SQLAlchemy model instance.

        Args:
            value: The dictionary to convert.
            context: The conversion context containing target type information.

        Returns:
            An instance of the target SQLAlchemy model.

        Raises:
            ConversionError: If the input is not a dict, the target type system
                             is missing, the target model class cannot be resolved,
                             or instantiation fails.
            PolitipoError: If SQLAlchemy is not installed.
        """
        if not _SQLA_AVAILABLE:
            raise PolitipoError("SQLAlchemy is required for DictToSQLAlchemyModelStrategy")

        if not isinstance(value, dict):
            raise ConversionError(f"Expected dict, got {type(value)}")

        # Get the target SQLAlchemy model class from the context
        if not context.target_type_system:
            raise ConversionError(
                "Target type system (SQLAlchemyModelTypeSystem) not found in context"
            )

        try:
            # Use the target system to reconstruct the model class from the canonical type
            model_class: type = context.target_type_system.from_canonical(context.target)
            if not isinstance(model_class, type) or not hasattr(model_class, "__table__"):
                # Check if it's a DeclarativeMeta type
                if not isinstance(getattr(model_class, "__class__", None), DeclarativeMeta):
                    raise ConversionError(
                        f"Target system did not return a valid SQLAlchemy model class for {context.target.name}"
                    )

        except NotImplementedError:
            raise ConversionError(
                f"Target system '{context.target_type_system.name}' cannot reconstruct model from canonical type."
            )
        except Exception as e:
            raise ConversionError(
                f"Failed to get target SQLAlchemy model class from canonical type: {e}"
            ) from e

        # Instantiate the model
        try:
            # Filter dict keys based on model columns if not in strict mode
            data_to_instantiate = value
            if not context.strict:
                valid_column_names = {c.name for c in model_class.__table__.columns}
                data_to_instantiate = {k: v for k, v in value.items() if k in valid_column_names}

            # Instantiate using **kwargs
            # Note: This assumes the dictionary keys match model attribute names.
            # Complex relationships or non-column attributes might require custom handling
            # or should be set after initial instantiation if needed.
            instance = model_class(**data_to_instantiate)
            return instance
        except TypeError as e:
            # Catch errors like unexpected keyword arguments
            if context.strict:
                raise ConversionError(
                    f"Failed to instantiate {model_class.__name__} with data {value}. Strict mode error: {e}"
                ) from e
            else:
                # Try again without filtering if initial filtering failed, though unlikely
                try:
                    return model_class(**value)
                except Exception as final_e:
                    raise ConversionError(
                        f"Failed to instantiate {model_class.__name__} even in non-strict mode: {final_e}"
                    ) from final_e
        except Exception as e:
            # Catch other potential instantiation errors
            raise ConversionError(
                f"Failed to create SQLAlchemy model instance of {model_class.__name__}: {e}"
            ) from e

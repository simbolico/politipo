# politipo/core/conversion/strategies/sqlalchemy_model_to_dict.py

from typing import Any, Dict

from politipo.core.types import CanonicalType
from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError, PolitipoError

# Attempt to import SQLAlchemy components, required for type checking and logic
try:
    from sqlalchemy.orm.attributes import instance_state
    _SQLA_AVAILABLE = True
except ImportError:
    _SQLA_AVAILABLE = False
    # Define dummy for type hinting if needed, though runtime checks handle absence
    instance_state = None


class SQLAlchemyModelToDictStrategy(ConversionStrategy):
    """Strategy for converting SQLAlchemy model instances to dictionaries."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """
        Returns True if the source is a SQLAlchemy model and the target is a dict.
        Relies on metadata set by SQLAlchemyModelTypeSystem.
        """
        return (
            source.kind == "composite" and
            source.meta is not None and
            source.meta.get("origin_system") == "sqlalchemy_model" and
            target.kind == "container" and
            target.name == "dict"
        )

    def convert(self, value: Any, context: ConversionContext) -> Dict[str, Any]:
        """
        Convert an SQLAlchemy model instance to a dictionary.

        Args:
            value: The SQLAlchemy model instance to convert.
            context: The conversion context.

        Returns:
            A dictionary representing the model's column data.

        Raises:
            ConversionError: If the input is not a valid SQLAlchemy model instance
                             or if conversion fails.
            PolitipoError: If SQLAlchemy is not installed.
        """
        if not _SQLA_AVAILABLE:
            raise PolitipoError("SQLAlchemy is required for SQLAlchemyModelToDictStrategy")

        # Basic check: Does it look like an SQLAlchemy model instance?
        # A more robust check might involve checking instance_state
        if not hasattr(value, '__table__') or not hasattr(value, '__class__'):
             # Try checking instance state as a more reliable indicator
             try:
                 state = instance_state(value)
                 if not state:
                     raise AttributeError # Not an instrumented instance
             except (AttributeError, TypeError):
                 raise ConversionError(f"Expected an SQLAlchemy model instance, got {type(value)}")

        try:
            # Extract column names from the model's table definition
            column_names = [c.name for c in value.__table__.columns]

            # Build the dictionary using getattr
            data_dict = {}
            for col_name in column_names:
                # Use getattr to retrieve the value; handle potential unloaded attributes
                # Note: This might trigger lazy loading if attributes are not loaded.
                # For complex scenarios, pre-loading attributes might be needed.
                try:
                    data_dict[col_name] = getattr(value, col_name)
                except AttributeError:
                    # This shouldn't typically happen for mapped columns, but handle defensively
                    data_dict[col_name] = None # Or raise an error? Default to None for safety.

            return data_dict
        except AttributeError as e:
             # Catch errors like value.__table__ not existing if the initial check failed
            raise ConversionError(f"Failed to access SQLAlchemy model properties: {e}") from e
        except Exception as e:
            # Catch other potential errors during attribute access or processing
            raise ConversionError(f"Failed to convert SQLAlchemy model instance to dict: {e}") from e 
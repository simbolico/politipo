# politipo/core/conversion/strategies/sqlalchemy_model_to_model.py

from typing import Any

from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.conversion.utils.instantiation import (
    create_model_instance_from_dict,  # Helper for Pydantic/SQLModel
)
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType

# Attempt to import SQLAlchemy components
try:
    from sqlalchemy.orm.attributes import instance_state

    _SQLA_AVAILABLE = True
except ImportError:
    _SQLA_AVAILABLE = False
    instance_state = None

# Attempt to import Pydantic components
try:
    from pydantic import BaseModel

    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = None


class SQLAlchemyModelToModelStrategy(ConversionStrategy):
    """
    Strategy for converting SQLAlchemy model instances to Pydantic/SQLModel instances.
    Internally converts the SQLAlchemy model to a dictionary first.
    """

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """
        Returns True if source is SQLAlchemy model and target is Pydantic/SQLModel.
        """
        return (
            source.kind == "composite"
            and source.meta is not None
            and source.meta.get("origin_system") == "sqlalchemy_model"
            and target.kind == "composite"
            and target.meta is not None
            and target.meta.get("origin_system") in ("pydantic", "sqlmodel")
        )

    def convert(self, value: Any, context: ConversionContext) -> Any:
        """
        Convert an SQLAlchemy model instance to a Pydantic/SQLModel instance.

        Args:
            value: The SQLAlchemy model instance.
            context: The conversion context.

        Returns:
            An instance of the target Pydantic or SQLModel model.

        Raises:
            ConversionError: If conversion fails at any step.
            PolitipoError: If SQLAlchemy or Pydantic is not installed.
        """
        if not _SQLA_AVAILABLE:
            raise PolitipoError("SQLAlchemy is required for SQLAlchemyModelToModelStrategy")
        if not _PYDANTIC_AVAILABLE:
            # Check specifically if target is SQLModel which also needs Pydantic
            is_sqlmodel_target = (
                context.target.meta and context.target.meta.get("origin_system") == "sqlmodel"
            )
            if is_sqlmodel_target:
                raise PolitipoError(
                    "Pydantic (dependency of SQLModel) is required for SQLAlchemyModelToModelStrategy"
                )
            # Only raise if the target is explicitly Pydantic
            elif context.target.meta and context.target.meta.get("origin_system") == "pydantic":
                raise PolitipoError("Pydantic is required for SQLAlchemyModelToModelStrategy")
            # If target system unknown or different, this strategy shouldn't have been selected anyway

        # 1. Convert SQLAlchemy Model to Dictionary
        try:
            # Basic check for SQLAlchemy instance
            if not hasattr(value, "__table__"):
                try:
                    state = instance_state(value)
                    if not state:
                        raise AttributeError
                except (AttributeError, TypeError):
                    raise ConversionError(
                        f"Expected an SQLAlchemy model instance, got {type(value)}"
                    )

            column_names = [c.name for c in value.__table__.columns]
            data_dict = {col: getattr(value, col) for col in column_names}
        except Exception as e:
            raise ConversionError(
                f"Failed to convert SQLAlchemy model {type(value).__name__} to dict: {e}"
            ) from e

        # 2. Get Target Pydantic/SQLModel Class
        if not context.target_type_system:
            raise ConversionError(
                "Target type system (PydanticTypeSystem or SQLModelTypeSystem) not found in context"
            )
        try:
            # Use the target system (Pydantic or SQLModel) to get the class
            model_class: type = context.target_type_system.from_canonical(context.target)
            # Basic validation that we got a class, ideally check if it's a BaseModel subclass
            if not isinstance(model_class, type) or (
                _PYDANTIC_AVAILABLE and not issubclass(model_class, BaseModel)
            ):
                # Allow if Pydantic isn't installed but SQLModel might be the target (edge case)
                if not (
                    context.target.meta and context.target.meta.get("origin_system") == "sqlmodel"
                ):
                    raise ConversionError(
                        f"Target system did not return a valid Pydantic/SQLModel class for {context.target.name}"
                    )
        except Exception as e:
            raise ConversionError(f"Failed to get target Pydantic/SQLModel class: {e}") from e

        # 3. Instantiate Pydantic/SQLModel from Dictionary
        try:
            # Use the utility function which handles Pydantic v1/v2 and strict mode
            instance = create_model_instance_from_dict(model_class, data_dict, context.strict)
            return instance
        except Exception as e:
            # The helper raises ConversionError, but catch just in case
            raise ConversionError(
                f"Failed to create Pydantic/SQLModel instance {model_class.__name__} from dict: {e}"
            ) from e

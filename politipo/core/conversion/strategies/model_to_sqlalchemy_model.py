# politipo/core/conversion/strategies/model_to_sqlalchemy_model.py

from typing import Any, Dict, Type

from politipo.core.types import CanonicalType
from politipo.core.conversion.context import ConversionContext
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.utils.pydantic import get_pydantic_version # To check for model_dump/dict

# Attempt to import SQLAlchemy components
try:
    from sqlalchemy.orm import DeclarativeMeta
    _SQLA_AVAILABLE = True
except ImportError:
    _SQLA_AVAILABLE = False
    DeclarativeMeta = None

# Attempt to import Pydantic components
try:
    from pydantic import BaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = None


class ModelToSQLAlchemyModelStrategy(ConversionStrategy):
    """
    Strategy for converting Pydantic/SQLModel instances to SQLAlchemy model instances.
    Internally converts the source model to a dictionary first.
    """

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """
        Returns True if source is Pydantic/SQLModel and target is SQLAlchemy model.
        """
        return (
            source.kind == "composite" and
            source.meta is not None and
            source.meta.get("origin_system") in ("pydantic", "sqlmodel") and
            target.kind == "composite" and
            target.meta is not None and
            target.meta.get("origin_system") == "sqlalchemy_model"
        )

    def convert(self, value: Any, context: ConversionContext) -> Any:
        """
        Convert a Pydantic/SQLModel instance to an SQLAlchemy model instance.

        Args:
            value: The Pydantic or SQLModel instance.
            context: The conversion context.

        Returns:
            An instance of the target SQLAlchemy model.

        Raises:
            ConversionError: If conversion fails at any step.
            PolitipoError: If SQLAlchemy or Pydantic is not installed.
        """
        if not _SQLA_AVAILABLE:
            raise PolitipoError("SQLAlchemy is required for ModelToSQLAlchemyModelStrategy")
        if not _PYDANTIC_AVAILABLE:
             # Check specifically if it's a SQLModel type which also needs Pydantic
             is_sqlmodel_source = context.source.meta and context.source.meta.get("origin_system") == "sqlmodel"
             if is_sqlmodel_source:
                 raise PolitipoError("Pydantic (dependency of SQLModel) is required for ModelToSQLAlchemyModelStrategy")
             # Only raise if the source is explicitly Pydantic
             elif context.source.meta and context.source.meta.get("origin_system") == "pydantic":
                 raise PolitipoError("Pydantic is required for ModelToSQLAlchemyModelStrategy")
             # If source system unknown or different, let it proceed, maybe it has .dict() anyway


        # 1. Convert Source Model (Pydantic/SQLModel) to Dictionary
        try:
            if hasattr(value, "model_dump"):  # Pydantic v2 preferred
                data_dict = value.model_dump()
            elif hasattr(value, "dict"):      # Pydantic v1 fallback
                data_dict = value.dict()
            else:
                raise ConversionError(f"Source object {type(value)} does not have 'model_dump' or 'dict' method.")
        except Exception as e:
            raise ConversionError(f"Failed to convert source model {type(value).__name__} to dict: {e}") from e

        # 2. Get Target SQLAlchemy Model Class
        if not context.target_type_system:
            raise ConversionError("Target type system (SQLAlchemyModelTypeSystem) not found in context")
        try:
            model_class: Type = context.target_type_system.from_canonical(context.target)
            if not isinstance(model_class, type) or not hasattr(model_class, '__table__'):
                 # Check if it's a DeclarativeMeta type
                 if not isinstance(getattr(model_class, '__class__', None), DeclarativeMeta):
                      raise ConversionError(f"Target system did not return a valid SQLAlchemy model class for {context.target.name}")
        except Exception as e:
            raise ConversionError(f"Failed to get target SQLAlchemy model class: {e}") from e

        # 3. Instantiate SQLAlchemy Model from Dictionary
        try:
            # Filter dict keys based on model columns if not in strict mode
            data_to_instantiate = data_dict
            if not context.strict:
                valid_column_names = {c.name for c in model_class.__table__.columns}
                data_to_instantiate = {k: v for k, v in data_dict.items() if k in valid_column_names}

            instance = model_class(**data_to_instantiate)
            return instance
        except TypeError as e:
             if context.strict:
                 raise ConversionError(f"Failed to instantiate {model_class.__name__} from dict {data_dict}. Strict mode error: {e}") from e
             else:
                 # Try again without filtering if initial filtering failed
                 try:
                     return model_class(**data_dict)
                 except Exception as final_e:
                     raise ConversionError(f"Failed to instantiate {model_class.__name__} even in non-strict mode: {final_e}") from final_e
        except Exception as e:
            raise ConversionError(f"Failed to create SQLAlchemy model instance {model_class.__name__} from dict: {e}") from e 
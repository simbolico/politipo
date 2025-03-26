from typing import Any, Dict, Type
from functools import lru_cache

try:
    from sqlmodel import SQLModel
    from pydantic.fields import FieldInfo
    # Attempt to import other relevant SQLModel parts if needed for detection/mapping
except ImportError:
    SQLModel = None
    FieldInfo = None
    # Allow type checking

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError
# Import Pydantic system for delegation - assumes it's registered
# This creates a dependency, might need a registry lookup instead
from politipo.plugins.pydantic import PydanticTypeSystem


class SQLModelTypeSystem(TypeSystem):
    """Type system implementation for SQLModel classes."""

    name = "sqlmodel"

    def __init__(self):
        if SQLModel is None:
            raise PolitipoError("SQLModel is not installed.")
        # Instantiate Pydantic system - replace with DI/registry later
        try:
            self._pydantic_system = PydanticTypeSystem()
        except PolitipoError:
            raise PolitipoError("Pydantic must be installed to use SQLModelTypeSystem.")

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic SQLModel type."""
        return CanonicalType(
            kind="composite",
            name="SQLModel",
            params={"fields": {}},
            meta=TypeMeta(data={
                "origin_system": self.name,
                "is_sqlmodel": True
            })
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: Type[SQLModel]) -> CanonicalType:
        """Converts a SQLModel class to CanonicalType by combining Pydantic structure and SQL metadata."""
        if not self.detect(type_obj):
            raise ConversionError(f"Object {type_obj} is not a SQLModel class.")

        # 1. Get Pydantic Canonical Representation
        try:
            pydantic_canonical = self._pydantic_system.to_canonical(type_obj)
            if pydantic_canonical.kind != "composite":
                # Should be composite if it's a SQLModel class
                raise ConversionError("Pydantic plugin did not return a composite type for SQLModel.")
        except Exception as e:
            raise ConversionError(f"Failed to get Pydantic canonical representation for {type_obj.__name__}: {e}")

        # 2. Extract SQL Specific Metadata
        sql_meta_data = {
            "origin_system": self.name,  # Override origin
            "is_sqlmodel": True,
        }
        # Table name
        sql_meta_data['sql_tablename'] = getattr(type_obj, '__tablename__', type_obj.__name__.lower())  # Default convention

        # Primary Keys (Check Pydantic fields for primary_key=True)
        pk_names = []
        try:
            model_fields = getattr(type_obj, 'model_fields', getattr(type_obj, '__fields__', {}))
            for name, field in model_fields.items():
                field_info = getattr(field, 'field_info', field)  # Adapt for v1/v2 FieldInfo access
                if getattr(field_info, 'primary_key', False):
                    pk_names.append(name)
        except Exception:
            pass  # Ignore errors inspecting fields for PK
        sql_meta_data['sql_primary_key'] = pk_names

        # Extract Indexes from __table_args__
        sql_meta_data['sql_indexes'] = []
        try:
            table_args = getattr(type_obj, '__table_args__', None)
            if isinstance(table_args, tuple):
                for arg in table_args:
                    if hasattr(arg, '__visit_name__') and arg.__visit_name__ == 'index':
                        sql_meta_data['sql_indexes'].append(str(arg))
        except Exception:
            pass  # Ignore errors extracting indexes

        # Extract Relationships
        sql_meta_data['sql_relationships'] = {}
        try:
            for name, attr in type_obj.__dict__.items():
                if hasattr(attr, 'prop') and hasattr(attr.prop, 'target'):
                    sql_meta_data['sql_relationships'][name] = {
                        'target': attr.prop.target.name,
                        'uselist': attr.prop.uselist,
                        'direction': str(attr.prop.direction),
                        'foreign_keys': [str(fk) for fk in attr.prop.foreign_keys] if attr.prop.foreign_keys else []
                    }
        except Exception:
            pass  # Ignore errors extracting relationships

        # 3. Merge Metadata
        final_meta_data = pydantic_canonical.meta.data.copy() if pydantic_canonical.meta else {}
        final_meta_data.update(sql_meta_data)  # SQLModel specifics override/add

        # 4. Return combined CanonicalType
        return CanonicalType(
            kind=pydantic_canonical.kind,
            name=pydantic_canonical.name,
            params=pydantic_canonical.params,  # Reuse fields from Pydantic version
            constraints=pydantic_canonical.constraints,  # Reuse constraints from Pydantic version
            meta=TypeMeta(data=final_meta_data)
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> Type[SQLModel]:
        """Dynamic SQLModel creation is not supported yet."""
        raise NotImplementedError("Dynamic SQLModel reconstruction from CanonicalType is complex and not yet supported.")

    def detect(self, obj: Any) -> bool:
        """Returns True if obj is a SQLModel class."""
        return (
            SQLModel is not None and
            isinstance(obj, type) and
            issubclass(obj, SQLModel)
        )

    # @property
    # def pydantic_system(self) -> PydanticTypeSystem:
    #     # Example of lazy loading/lookup (requires engine access or injection)
    #     if self._pydantic_system is None:
    #         # Replace with actual lookup mechanism
    #         from politipo.plugins.pydantic import PydanticTypeSystem
    #         self._pydantic_system = PydanticTypeSystem()
    #     return self._pydantic_system

    # @lru_cache(maxsize=128)
    # def to_canonical(self, type_obj: Type[SQLModel]) -> CanonicalType:
    #     """Converts a SQLModel class to CanonicalType.
    #        Delegates heavily to Pydantic's logic but could add SQL specific meta.
    #     """
    #     if not self.detect(type_obj):
    #          raise ConversionError(f"Object {type_obj} is not a SQLModel class.")

    #     fields = {}
    #     try:
    #         # Use Pydantic v2+ field inspection if available
    #         model_fields = getattr(type_obj, 'model_fields', None)
    #         if model_fields:
    #              for name, field in model_fields.items():
    #                  field_canon = CanonicalType(kind="primitive", name="any", meta=TypeMeta(data={"details": "sqlmodel_field_placeholder"}))
    #                  fields[name] = {"type": field_canon, "constraints": {}, "required": field.is_required()}
    #         else: # Fallback for Pydantic v1 style
    #              for name, field in getattr(type_obj, '__fields__', {}).items():
    #                  field_canon = CanonicalType(kind="primitive", name="any", meta=TypeMeta(data={"details": "sqlmodel_field_placeholder"}))
    #                  fields[name] = {"type": field_canon, "constraints": {}, "required": field.required}

    #     except AttributeError:
    #          # Failed to get fields, return basic structure
    #          pass

    #     # Extract potential table name, primary key info etc. into meta
    #     meta_data = {
    #         "origin_system": self.name,
    #         "is_sqlmodel": True
    #     }
    #     if hasattr(type_obj, '__tablename__'):
    #         meta_data['sql_tablename'] = type_obj.__tablename__
    #     # Add more SQL specific metadata extraction here

    #     return CanonicalType(
    #         kind="composite",
    #         name=type_obj.__name__,
    #         params={"fields": fields},
    #         meta=TypeMeta(data=meta_data)
    #     )

    #     # Ideal delegation:
    #     # canon = self.pydantic_system.to_canonical(type_obj)
    #     # # Add SQLModel specific metadata
    #     # sql_meta = {"is_sqlmodel": True, ...}
    #     # merged_meta = {**(canon.meta or {}), **sql_meta}
    #     # return CanonicalType(kind=canon.kind, name=canon.name, params=canon.params, constraints=canon.constraints, meta=merged_meta) 
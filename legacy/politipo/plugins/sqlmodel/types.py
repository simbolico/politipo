from functools import lru_cache
from typing import Any

try:
    from pydantic.fields import FieldInfo
    from sqlalchemy import Column, Index, MetaData, Table
    from sqlalchemy.orm import relationship
    from sqlmodel import Field, SQLModel

    # Attempt to import other relevant SQLModel parts if needed for detection/mapping
except ImportError:
    SQLModel = None
    Field = None
    Column = None
    Index = None
    Table = None
    MetaData = None
    relationship = None
    FieldInfo = None
    # Allow type checking

from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType, TypeMeta, TypeSystem

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
            meta=TypeMeta(data={"origin_system": self.name, "is_sqlmodel": True}),
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: type[SQLModel]) -> CanonicalType:
        """
        Converts a SQLModel class to CanonicalType by combining Pydantic structure and SQL metadata.

        This function extracts metadata from these SQLModel attributes:
            - `__tablename__`: The name of the database table.
            - Model fields (using Pydantic's field inspection):
                - Name: The field name.
                - Type: Mapped to a canonical type using the Pydantic type system.
                - `primary_key`:  If True, the field is a primary key.
                - `foreign_key`:  A foreign key relationship.
            - `__table_args__`: Extracts index definitions from the table arguments.
            - Relationships (using `sqlalchemy.orm.relationship`):
                - Target: The target SQLModel class.
                - `uselist`:  Whether the relationship is one-to-many or one-to-one.
                - `direction`: The direction of the relationship.
                - `foreign_keys`: Foreign key columns involved in the relationship.

        It supports mapping these types to canonical types (delegating to Pydantic):
            - `int`, `str`, `float`, `bool`, `date`, `datetime`, `Decimal`, `list`, `dict`
            - Other types are mapped to 'any'.

        Args:
            type_obj: The SQLModel class to convert.

        Returns:
            A CanonicalType representation of the SQLModel.

        Raises:
            ConversionError: If the object is not a SQLModel class, or if extraction fails.
        """
        if not self.detect(type_obj):
            raise ConversionError(f"Object {type_obj} is not a SQLModel class.")

        # 1. Get Pydantic Canonical Representation
        try:
            pydantic_canonical = self._pydantic_system.to_canonical(type_obj)
            if pydantic_canonical.kind != "composite":
                # Should be composite if it's a SQLModel class
                raise ConversionError(
                    "Pydantic plugin did not return a composite type for SQLModel."
                )
        except Exception as e:
            raise ConversionError(
                f"Failed to get Pydantic canonical representation for {type_obj.__name__}: {e}"
            )

        # 2. Extract SQL Specific Metadata
        sql_meta_data = {
            "origin_system": self.name,  # Override origin
            "is_sqlmodel": True,
        }
        # Table name
        sql_meta_data["sql_tablename"] = getattr(
            type_obj, "__tablename__", type_obj.__name__.lower()
        )

        # Primary Keys (Check Pydantic fields for primary_key=True)
        pk_names = []
        try:
            model_fields = getattr(type_obj, "model_fields", getattr(type_obj, "__fields__", {}))
            for name, field in model_fields.items():
                field_info = getattr(field, "field_info", field)  # Adapt for v1/v2 FieldInfo access
                if getattr(field_info, "primary_key", False):
                    pk_names.append(name)
        except Exception:
            pass  # Ignore errors inspecting fields for PK
        sql_meta_data["sql_primary_key"] = pk_names

        # Extract Indexes from __table_args__
        sql_meta_data["sql_indexes"] = []
        try:
            table_args = getattr(type_obj, "__table_args__", None)
            if isinstance(table_args, tuple):
                for arg in table_args:
                    if hasattr(arg, "__visit_name__") and arg.__visit_name__ == "index":
                        sql_meta_data["sql_indexes"].append(
                            {
                                "name": arg.name,
                                "columns": [col.name for col in arg.columns],
                                "unique": arg.unique,
                            }
                        )
        except Exception:
            pass  # Ignore errors extracting indexes

        # Extract Relationships
        sql_meta_data["sql_relationships"] = {}
        try:
            # Iterate through attributes and handle potential lazy loading
            for name, attr in type_obj.__dict__.items():
                # Trigger attribute access and handle AttributeError if it is lazy-loaded.
                try:
                    # This line will attempt to access the attribute and may trigger lazy loading
                    attr_value = getattr(type_obj, name)
                    if hasattr(attr_value, "prop") and hasattr(attr_value.prop, "target"):
                        sql_meta_data["sql_relationships"][name] = {
                            "target": attr_value.prop.target.name,
                            "uselist": attr_value.prop.uselist,
                            "direction": str(attr_value.prop.direction),
                            "foreign_keys": (
                                [str(fk) for fk in attr_value.prop.foreign_keys]
                                if attr_value.prop.foreign_keys
                                else []
                            ),
                        }
                except AttributeError:
                    # Handle lazy-loaded attributes
                    print(
                        f"Warning: Could not access attribute '{name}' on SQLModel '{type_obj.__name__}'. Assuming it is lazy-loaded."
                    )
                    continue
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
            meta=TypeMeta(data=final_meta_data),
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> type[SQLModel]:
        """
        Reconstruct a SQLModel class from CanonicalType.

        Args:
            canonical: The canonical type representation

        Returns:
            A dynamically created SQLModel class

        Raises:
            ConversionError: If reconstruction fails
        """
        if canonical.kind != "composite":
            raise ConversionError(
                f"Cannot create SQLModel from non-composite type: {canonical.kind}"
            )

        meta = canonical.meta.data if canonical.meta else {}
        if not meta.get("is_sqlmodel", False):
            raise ConversionError("Canonical type does not represent a SQLModel")

        try:
            # 1. Get base Pydantic model from canonical
            base_model_class = self._pydantic_system.from_canonical(canonical)
        except Exception as e:
            raise ConversionError(f"Failed to create base Pydantic model: {e}") from e

        # 2. Extract SQL metadata
        tablename = meta.get("sql_tablename", canonical.name.lower())
        primary_keys = meta.get("sql_primary_key", [])
        indexes = meta.get("sql_indexes", [])
        relationships = meta.get("sql_relationships", {})

        # 3. Prepare model attributes
        model_attrs = {
            "__tablename__": tablename,
            "_sa_table_args": self._build_table_args(indexes),
        }

        # 4. Process fields to add SQLModel-specific attributes
        fields = canonical.params.get("fields", {})
        for name, field_info in fields.items():
            field_meta = field_info.get("meta", {})
            field_constraints = field_info.get("constraints", {})

            # Get the field from the base model
            if hasattr(base_model_class, name):
                base_field = getattr(base_model_class, name)
                field_type = base_field.annotation if hasattr(base_field, "annotation") else Any

                # Create SQLModel Field with SQL-specific attributes
                field_kwargs = {}

                # Handle primary key
                if name in primary_keys:
                    field_kwargs["primary_key"] = True

                # Handle foreign keys if present in constraints
                if "foreign_key" in field_constraints:
                    field_kwargs["foreign_key"] = field_constraints["foreign_key"]

                # Handle nullable/optional
                field_kwargs["nullable"] = not field_info.get("required", True)

                # Handle default value
                if "default" in field_info:
                    field_kwargs["default"] = field_info["default"]

                # Create the SQLModel Field
                model_attrs[name] = Field(default=field_info.get("default", ...), **field_kwargs)

        # 5. Add relationships
        for rel_name, rel_info in relationships.items():
            rel_kwargs = {
                "back_populates": rel_info.get("back_populates"),
                "uselist": rel_info.get("uselist", True),
            }
            # Filter out None values
            rel_kwargs = {k: v for k, v in rel_kwargs.items() if v is not None}

            model_attrs[rel_name] = relationship(rel_info["target"], **rel_kwargs)

        # 6. Create the SQLModel class
        try:
            model_class = type(canonical.name, (SQLModel,), model_attrs)

            # Mark as table model
            model_class.__table__ = True

            return model_class
        except Exception as e:
            raise ConversionError(f"Failed to create SQLModel class: {e}") from e

    def detect(self, obj: Any) -> bool:
        """Returns True if obj is a SQLModel class."""
        return SQLModel is not None and isinstance(obj, type) and issubclass(obj, SQLModel)

    def _build_table_args(self, indexes: list[dict]) -> tuple:
        """Build SQLAlchemy table arguments from metadata."""
        table_args = []

        # Create Index objects
        for idx in indexes:
            try:
                table_args.append(
                    Index(idx.get("name"), *idx.get("columns", []), unique=idx.get("unique", False))
                )
            except Exception:
                continue  # Skip invalid indexes

        return tuple(table_args)

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

import inspect
from typing import Any, Type
from functools import lru_cache

try:
    from sqlalchemy.orm import DeclarativeBase, RelationshipProperty
    from sqlalchemy.inspection import inspect as sqlalchemy_inspect
except ImportError:
    DeclarativeBase = None
    RelationshipProperty = None
    sqlalchemy_inspect = None

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError
from .types import SQLAlchemyTypeSystem


class SQLAlchemyModelTypeSystem(TypeSystem):
    """Type system implementation for SQLAlchemy declarative models (table structures)."""

    name = "sqlalchemy_model"

    def __init__(self):
        if DeclarativeBase is None:
            raise PolitipoError("SQLAlchemy is not installed or version mismatch.")
        # Instantiate the column type system - improve with registry/DI later
        self._col_type_system = SQLAlchemyTypeSystem()

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic SQLAlchemy Model type."""
        return CanonicalType(
            kind="composite",
            name="SQLAlchemyModel",
            params={"fields": {}},
            meta=TypeMeta(data={"origin_system": self.name})
        )

    @lru_cache(maxsize=64)
    def to_canonical(self, type_obj: Type) -> CanonicalType:
        """Converts a SQLAlchemy declarative model class to CanonicalType."""
        if not self.detect(type_obj):
            raise ConversionError(f"Object {type_obj} is not a recognized SQLAlchemy declarative model class.")

        try:
            mapper = sqlalchemy_inspect(type_obj)
            table = mapper.local_table
        except Exception as e:
            raise ConversionError(f"Could not inspect SQLAlchemy model {type_obj.__name__}: {e}")

        fields = {}
        pk_col_names = set(pk_col.name for pk_col in table.primary_key.columns)

        for col in table.columns:
            # Get column type canonical representation
            try:
                # Pass the col.type instance to get details like length/precision
                canonical_col_type = self._col_type_system.to_canonical(col.type)
            except Exception as e:
                # Fallback if column type conversion fails
                canonical_col_type = CanonicalType(
                    kind="primitive",
                    name="any",
                    meta=TypeMeta(data={"error": str(e)})
                )

            # Add primary key info to column type meta if applicable
            is_pk = col.name in pk_col_names
            if is_pk:
                col_meta_data = {
                    "sql_primary_key": True,
                    **(canonical_col_type.meta.data if canonical_col_type.meta else {})
                }
                canonical_col_type = CanonicalType(
                    kind=canonical_col_type.kind,
                    name=canonical_col_type.name,
                    params=canonical_col_type.params,
                    constraints=canonical_col_type.constraints,
                    meta=TypeMeta(data=col_meta_data)
                )

            # Handle default - this can be complex (scalar, SQL func, Python func)
            default_value_repr = ...  # Ellipsis for no default
            if col.default is not None:
                # Simple scalar default
                if hasattr(col.default, 'arg'):
                    default_value_repr = col.default.arg
                # TODO: Handle SQL functions, Python callables etc. - complex
                # else: default_value_repr = f"SQLDefault({col.default})"

            # Use standard field structure
            fields[col.name] = {
                "type": canonical_col_type,
                "required": not col.nullable,  # required = not nullable
                "default": default_value_repr,
                "description": col.comment,
            }

        # Extract table metadata
        meta_data = {
            "origin_system": self.name,
            "sql_tablename": table.name,
            "sql_primary_key": list(pk_col_names),
            "sql_indexes": [str(idx) for idx in table.indexes],  # Basic representation
            "sql_constraints": [str(c) for c in table.constraints if c not in table.primary_key],  # Basic
            "sql_relationships": {},  # Placeholder for relationship info
            "description": inspect.getdoc(type_obj),
        }

        # Extract basic relationship info (complex details deferred)
        try:
            for name, prop in mapper.relationships.items():
                meta_data["sql_relationships"][name] = {
                    "target": prop.mapper.class_.__name__,  # Target class name
                    "direction": str(prop.direction),
                    "uselist": prop.uselist,
                    # Add more details like foreign_keys, lazy strategy if needed
                }
        except Exception:
            meta_data["sql_relationships"] = {"error": "Failed to extract relationships"}

        return CanonicalType(
            kind="composite",
            name=type_obj.__name__,
            params={"fields": fields},  # Use standardized 'fields' key
            meta=TypeMeta(data=meta_data)
        )

    @lru_cache(maxsize=64)
    def from_canonical(self, canonical: CanonicalType) -> Type:
        """Dynamic SQLAlchemy model creation is not supported yet."""
        raise NotImplementedError("Dynamic SQLAlchemy model reconstruction from CanonicalType is complex and not yet supported.")

    def detect(self, obj: Any) -> bool:
        """Returns True if obj is a SQLAlchemy declarative model class."""
        # Need robust check, isinstance(obj, type) and having __table__ is a good start
        # Also check inheritance from DeclarativeBase if possible/reliable
        return (
            DeclarativeBase is not None and
            isinstance(obj, type) and
            hasattr(obj, '__table__') and  # Mapped class indicator
            hasattr(obj, '__mapper__') and  # Mapped class indicator
            obj is not DeclarativeBase  # Don't detect the base itself
        ) 
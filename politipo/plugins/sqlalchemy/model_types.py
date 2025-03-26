from typing import Any, Dict, Type, Optional, List, Union, Tuple
from functools import lru_cache
import inspect
from decimal import Decimal

try:
    from sqlalchemy import Column, Table, MetaData, Index, UniqueConstraint, CheckConstraint, ForeignKey
    from sqlalchemy.orm import DeclarativeMeta, Mapped, mapped_column, relationship
    from sqlalchemy.orm.decl_api import registry
    from sqlalchemy.sql.schema import SchemaItem
    from sqlalchemy.sql import text
    from sqlalchemy import func
except ImportError:
    Column = Table = MetaData = DeclarativeMeta = Mapped = mapped_column = registry = SchemaItem = None
    Index = UniqueConstraint = CheckConstraint = ForeignKey = relationship = text = func = None

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError
from politipo.plugins.sqlalchemy.types import SQLAlchemyTypeSystem

class SQLAlchemyModelTypeSystem(TypeSystem):
    """Type system implementation for SQLAlchemy declarative models."""

    name = "sqlalchemy_model"

    def __init__(self):
        if DeclarativeMeta is None:
            raise PolitipoError("SQLAlchemy is not installed. Cannot initialize SQLAlchemyModelTypeSystem.")
        # Use SQLAlchemyTypeSystem for handling column types
        self._type_system = SQLAlchemyTypeSystem()

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic model type as the default target."""
        return CanonicalType(
            kind="composite",
            name="Model",
            params={"fields": {}},
            meta=TypeMeta(data={"origin_system": self.name})
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: Type) -> CanonicalType:
        """Convert a SQLAlchemy declarative model to CanonicalType."""
        if not self.detect(type_obj):
            raise ConversionError(f"Type {type_obj} is not a SQLAlchemy declarative model")

        fields_canonical = {}
        table = getattr(type_obj, '__table__', None)
        if not table:
            raise ConversionError(f"Model {type_obj} has no __table__ attribute")

        # Process columns
        for name, column in table.columns.items():
            # Get column type
            column_type = column.type
            type_canonical = self._type_system.to_canonical(column_type)

            # Extract constraints
            constraints = {}
            if column.primary_key:
                constraints['primary_key'] = True
            if not column.nullable:
                constraints['nullable'] = False
            if column.unique:
                constraints['unique'] = True
            if column.foreign_keys:
                fk_list = []
                for fk in column.foreign_keys:
                    fk_list.append({
                        'table': fk.column.table.name,
                        'column': fk.column.name
                    })
                constraints['foreign_keys'] = fk_list

            # Get default value and server default
            default_value = ...  # Ellipsis represents no default
            if column.default is not None:
                if column.default.is_scalar:
                    default_value = column.default.arg
                elif column.default.is_callable:
                    default_value = {
                        'type': 'callable',
                        'name': column.default.name or str(column.default.arg),
                        'args': getattr(column.default, 'args', []),
                        'kwargs': getattr(column.default, 'kwargs', {})
                    }
            if column.server_default is not None:
                constraints['server_default'] = {
                    'sql': str(column.server_default.arg),
                    'for_update': bool(column.server_onupdate)
                }

            fields_canonical[name] = {
                "type": type_canonical,
                "constraints": constraints,
                "required": not column.nullable,
                "default": default_value,
                "description": getattr(column, 'doc', None),
            }

        # Extract model metadata
        model_meta = {
            "origin_system": self.name,
            "table_name": table.name,
            "schema": table.schema,
        }

        # Add model docstring if available
        model_description = inspect.getdoc(type_obj)
        if model_description:
            model_meta["description"] = model_description

        # Extract table constraints and indexes
        table_args = []
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                table_args.append({
                    'type': 'unique_constraint',
                    'name': constraint.name,
                    'columns': [col.name for col in constraint.columns]
                })
            elif isinstance(constraint, CheckConstraint):
                table_args.append({
                    'type': 'check_constraint',
                    'name': constraint.name,
                    'sqltext': str(constraint.sqltext)
                })

        for index in table.indexes:
            table_args.append({
                'type': 'index',
                'name': index.name,
                'columns': [col.name for col in index.columns],
                'unique': index.unique,
                'kwargs': {
                    'postgresql_using': getattr(index, 'postgresql_using', None),
                    'postgresql_where': str(getattr(index, 'postgresql_where', None)) if getattr(index, 'postgresql_where', None) else None
                }
            })

        if table_args:
            model_meta['table_args'] = table_args

        # Add relationships with more detail
        relationships = {}
        for name, rel in inspect.getmembers(type_obj):
            if hasattr(rel, 'property') and hasattr(rel.property, 'direction'):
                rel_info = {
                    'type': str(rel.property.direction.name),
                    'target': rel.property.mapper.class_.__name__,
                    'uselist': rel.property.uselist,
                    'back_populates': rel.property.back_populates,
                    'cascade': rel.property.cascade,
                    'lazy': rel.property.lazy,
                }
                # Extract foreign keys
                if rel.property.foreign_keys:
                    rel_info['foreign_keys'] = [
                        {'table': fk.column.table.name, 'column': fk.column.name}
                        for fk in rel.property.foreign_keys
                    ]
                relationships[name] = rel_info
        if relationships:
            model_meta["relationships"] = relationships

        return CanonicalType(
            kind="composite",
            name=type_obj.__name__,
            params={"fields": fields_canonical},
            meta=TypeMeta(data=model_meta)
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> Type:
        """
        Reconstructs a SQLAlchemy declarative model from CanonicalType.

        Args:
            canonical: The canonical type representation to convert.

        Returns:
            A SQLAlchemy declarative model class.

        Raises:
            ConversionError: If reconstruction fails or required data is missing.
        """
        if canonical.kind != "composite":
            raise ConversionError(f"Cannot create SQLAlchemy model from non-composite type: {canonical.kind}")

        fields_info = canonical.params.get("fields")
        if not fields_info:
            raise ConversionError(f"Cannot reconstruct SQLAlchemy model '{canonical.name}' without field definitions")

        # Create registry for declarative models
        mapper_registry = registry()
        Base = mapper_registry.generate_base()

        # Prepare columns for model
        columns = {}
        for name, field_info in fields_info.items():
            try:
                # Get SQLAlchemy type for the field
                field_type_canonical = field_info["type"]
                sa_type = self._type_system.from_canonical(field_type_canonical)

                # Prepare column kwargs
                column_kwargs = {}

                # Handle constraints
                constraints = field_info.get("constraints", {})
                if constraints.get("primary_key"):
                    column_kwargs["primary_key"] = True
                if not field_info.get("required", True):
                    column_kwargs["nullable"] = True
                if constraints.get("unique"):
                    column_kwargs["unique"] = True

                # Handle foreign keys
                if "foreign_keys" in constraints:
                    fk_list = []
                    for fk in constraints["foreign_keys"]:
                        fk_list.append(ForeignKey(f"{fk['table']}.{fk['column']}"))
                    if fk_list:
                        column_kwargs["foreign_keys"] = fk_list

                # Handle default value
                default_value = field_info.get("default", ...)
                if default_value is not ...:
                    if isinstance(default_value, dict) and default_value.get('type') == 'callable':
                        # Handle callable defaults
                        if hasattr(func, default_value['name']):
                            default_fn = getattr(func, default_value['name'])
                            column_kwargs["default"] = default_fn(
                                *default_value.get('args', []),
                                **default_value.get('kwargs', {})
                            )
                    else:
                        column_kwargs["default"] = default_value

                # Handle server default
                if "server_default" in constraints:
                    server_default = constraints["server_default"]
                    column_kwargs["server_default"] = text(server_default["sql"])
                    if server_default.get("for_update"):
                        column_kwargs["server_onupdate"] = text(server_default["sql"])

                # Add description/comment if available
                if field_info.get("description"):
                    column_kwargs["comment"] = field_info["description"]

                # Create column
                columns[name] = Column(sa_type, **column_kwargs)

            except Exception as e:
                raise ConversionError(f"Error creating column '{name}' for model '{canonical.name}': {e}") from e

        # Get table name and schema from metadata
        meta = canonical.meta.data if canonical.meta else {}
        table_name = meta.get("table_name", canonical.name.lower())
        schema = meta.get("schema")

        # Build table arguments
        table_args = []
        if schema:
            table_args.append({"schema": schema})

        # Add constraints and indexes from metadata
        for arg in meta.get("table_args", []):
            arg_type = arg.get("type")
            if arg_type == "unique_constraint":
                table_args.append(
                    UniqueConstraint(
                        *arg["columns"],
                        name=arg.get("name")
                    )
                )
            elif arg_type == "check_constraint":
                table_args.append(
                    CheckConstraint(
                        arg["sqltext"],
                        name=arg.get("name")
                    )
                )
            elif arg_type == "index":
                index_kwargs = {k: v for k, v in arg.get("kwargs", {}).items() if v is not None}
                table_args.append(
                    Index(
                        arg.get("name"),
                        *[columns[col] for col in arg["columns"]],
                        unique=arg.get("unique", False),
                        **index_kwargs
                    )
                )

        # Create model attributes
        model_attrs = {
            "__tablename__": table_name,
            "__table_args__": tuple(table_args) if table_args else None,
            **columns
        }

        # Add docstring if available
        if meta.get("description"):
            model_attrs["__doc__"] = meta["description"]

        # Add relationships
        for rel_name, rel_info in meta.get("relationships", {}).items():
            rel_kwargs = {
                "uselist": rel_info.get("uselist", True),
                "back_populates": rel_info.get("back_populates"),
                "cascade": rel_info.get("cascade"),
                "lazy": rel_info.get("lazy", "select"),
            }
            # Filter out None values
            rel_kwargs = {k: v for k, v in rel_kwargs.items() if v is not None}

            # Add foreign keys if specified
            if "foreign_keys" in rel_info:
                rel_kwargs["foreign_keys"] = [
                    columns[fk["column"]] for fk in rel_info["foreign_keys"]
                    if fk["column"] in columns
                ]

            model_attrs[rel_name] = relationship(rel_info["target"], **rel_kwargs)

        try:
            model = type(canonical.name, (Base,), model_attrs)
            return model
        except Exception as e:
            raise ConversionError(f"Failed to create SQLAlchemy model '{canonical.name}': {e}") from e

    def detect(self, obj: Any) -> bool:
        """Returns True if the object is a SQLAlchemy declarative model class."""
        if DeclarativeMeta is None:
            return False

        return (
            isinstance(obj, type) and
            isinstance(getattr(obj, '__class__', None), DeclarativeMeta) and
            hasattr(obj, '__table__')
        ) 
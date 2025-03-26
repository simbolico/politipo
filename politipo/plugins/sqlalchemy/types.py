from typing import Any, Dict, Type
from functools import lru_cache
from decimal import Decimal

try:
    import sqlalchemy.types as sa_types
except ImportError:
    # Allow type checking even if sqlalchemy isn't installed
    sa_types = None

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError


class SQLAlchemyTypeSystem(TypeSystem):
    """Type system implementation for SQLAlchemy types."""

    name = "sqlalchemy"

    def __init__(self):
        if sa_types is None:
            raise PolitipoError("SQLAlchemy is not installed. Cannot initialize SQLAlchemyTypeSystem.")

        # Mapping from SA type classes to canonical primitive names
        self._to_canonical_map: Dict[Type[sa_types.TypeEngine], str] = {
            sa_types.Integer: "int",
            sa_types.SmallInteger: "int",
            sa_types.BigInteger: "int",
            sa_types.String: "str",
            sa_types.Text: "str",
            sa_types.Unicode: "str",
            sa_types.UnicodeText: "str",
            sa_types.Enum: "str", # Represent Enum as string canonically
            sa_types.Float: "float",
            sa_types.Numeric: "decimal",
            sa_types.Boolean: "bool",
            sa_types.Date: "date",
            sa_types.DateTime: "datetime",
            sa_types.Time: "time",
            sa_types.Interval: "timedelta",
            sa_types.LargeBinary: "bytes",
            sa_types.PickleType: "any", # Pickle is too specific
            sa_types.JSON: "any", # JSON could be dict/list, map to 'any' for simplicity
        }

        # Mapping from canonical primitive names to SA type classes
        self._from_canonical_map: Dict[str, Type[sa_types.TypeEngine]] = {
            "int": sa_types.Integer,
            "str": sa_types.String, # Default to String
            "float": sa_types.Float,
            "decimal": sa_types.Numeric,
            "bool": sa_types.Boolean,
            "date": sa_types.Date,
            "datetime": sa_types.DateTime,
            "time": sa_types.Time,
            "timedelta": sa_types.Interval,
            "bytes": sa_types.LargeBinary,
            "any": sa_types.PickleType, # Default 'any' to PickleType? Or raise error?
        }

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic SQL type as the default target."""
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={"origin_system": self.name})
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: Type) -> CanonicalType:
        """Convert a SQLAlchemy type to CanonicalType."""
        # Handle SQLAlchemy types
        if isinstance(type_obj, type) and hasattr(type_obj, '__visit_name__'):
            # Extract type info
            type_name = getattr(type_obj, '__visit_name__', 'unknown')
            type_params = {}
            
            # Map common SQL types to canonical primitives
            sql_to_canonical = {
                'integer': 'int',
                'numeric': 'decimal',
                'string': 'str',
                'boolean': 'bool',
                'datetime': 'datetime',
                'date': 'date',
                'time': 'time',
                'interval': 'timedelta',
                'binary': 'bytes'
            }
            
            canonical_name = sql_to_canonical.get(type_name, 'any')
            
            # Extract type parameters if available
            if hasattr(type_obj, 'length'):
                type_params['length'] = type_obj.length
            if hasattr(type_obj, 'precision'):
                type_params['precision'] = type_obj.precision
            if hasattr(type_obj, 'scale'):
                type_params['scale'] = type_obj.scale
                
            return CanonicalType(
                kind="primitive",
                name=canonical_name,
                params=type_params,
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "sql_type": type_name
                })
            )

        # Fallback for unknown types
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={
                "origin_system": self.name,
                "original_type": str(type_obj)
            })
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> Type[sa_types.TypeEngine]:
        """Reconstructs an SQLAlchemy type from CanonicalType."""
        if canonical.kind != "primitive":
            # SQLAlchemy types map primarily to primitives
            # Could potentially handle complex types via JSON or PickleType
            # Raise error or default to a generic type like PickleType or String
            # raise ConversionError(f"Cannot convert non-primitive CanonicalType '{canonical.kind}:{canonical.name}' to SQLAlchemy type")
            return sa_types.PickleType # Or sa_types.String?

        if canonical.name in self._from_canonical_map:
            base_sa_type = self._from_canonical_map[canonical.name]
            kwargs = {}

            # Apply length constraint back to String
            if base_sa_type is sa_types.String:
                max_length_constraint = canonical.constraints.get('MaxLength')
                if max_length_constraint and isinstance(max_length_constraint.value, int):
                     kwargs['length'] = max_length_constraint.value

            # Apply precision/scale back to Numeric
            if base_sa_type is sa_types.Numeric:
                if 'precision' in canonical.params:
                    kwargs['precision'] = canonical.params['precision']
                if 'scale' in canonical.params:
                    kwargs['scale'] = canonical.params['scale']

             # Handle Enum reconstruction
            if base_sa_type is sa_types.Enum and 'enum_values' in canonical.params:
                # SA Enum requires the values at instantiation
                enum_values = canonical.params['enum_values']
                # Enum name is typically required, use canonical name or default
                enum_name = f"{canonical.name}_enum"
                return sa_types.Enum(*enum_values, name=enum_name) # Pass values directly

            # Handle DateTime timezone
            if base_sa_type is sa_types.DateTime and canonical.params.get('timezone'):
                kwargs['timezone'] = True

            # Instantiate the type with kwargs if any
            if kwargs:
                try:
                    return base_sa_type(**kwargs)
                except TypeError:
                    # If kwargs are not applicable (e.g., Integer with length), return base type
                    return base_sa_type()
            else:
                return base_sa_type()
        else:
            # Fallback for unmapped canonical types
            # Consider raising an error or defaulting to a specific type
            # raise ConversionError(f"No SQLAlchemy type mapping found for canonical name '{canonical.name}'")
            return sa_types.PickleType # Or String?

    def detect(self, obj: Any) -> bool:
        """Returns True if the object is an SQLAlchemy type."""
        if sa_types is None: return False # Cannot detect if not installed

        # Check if it's an instance of a TypeEngine or a subclass of TypeEngine
        return isinstance(obj, sa_types.TypeEngine) or (
               isinstance(obj, type) and issubclass(obj, sa_types.TypeEngine)
        ) 
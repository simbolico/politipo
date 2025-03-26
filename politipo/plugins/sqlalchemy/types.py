from typing import Any, Dict, Type, Optional, Union
from functools import lru_cache
from decimal import Decimal
import inspect

try:
    import sqlalchemy.types as sa_types
    from sqlalchemy.sql.type_api import TypeEngine
except ImportError:
    # Allow type checking even if sqlalchemy isn't installed
    sa_types = None
    TypeEngine = None

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
            sa_types.Enum: "str",  # Represent Enum as string canonically
            sa_types.Float: "float",
            sa_types.Numeric: "decimal",
            sa_types.Boolean: "bool",
            sa_types.Date: "date",
            sa_types.DateTime: "datetime",
            sa_types.Time: "time",
            sa_types.Interval: "timedelta",
            sa_types.LargeBinary: "bytes",
            sa_types.PickleType: "any",  # Pickle is too specific
            sa_types.JSON: "dict",  # Map JSON to dict instead of any
        }

        # Mapping from canonical primitive names to SA type classes
        self._from_canonical_map: Dict[str, Type[sa_types.TypeEngine]] = {
            "int": sa_types.Integer,
            "str": sa_types.String,  # Default to String
            "float": sa_types.Float,
            "decimal": sa_types.Numeric,
            "bool": sa_types.Boolean,
            "date": sa_types.Date,
            "datetime": sa_types.DateTime,
            "time": sa_types.Time,
            "timedelta": sa_types.Interval,
            "bytes": sa_types.LargeBinary,
            "dict": sa_types.JSON,  # Map dict to JSON
            "any": sa_types.PickleType,
        }

    def _extract_type_info(self, type_obj: Union[Type[TypeEngine], TypeEngine]) -> Dict[str, Any]:
        """
        Extract detailed type information from a SQLAlchemy type.
        
        Args:
            type_obj: SQLAlchemy type to analyze
            
        Returns:
            Dictionary containing type metadata
        """
        info = {
            "type_name": getattr(type_obj, '__visit_name__', type_obj.__class__.__name__),
            "module": type_obj.__class__.__module__,
            "is_custom": not type_obj.__class__.__module__.startswith('sqlalchemy'),
        }

        # Extract common attributes
        for attr in ['length', 'precision', 'scale', 'timezone', 'collation']:
            if hasattr(type_obj, attr):
                value = getattr(type_obj, attr)
                if value is not None:
                    info[attr] = value

        # Handle enum types
        if isinstance(type_obj, sa_types.Enum):
            info['enum_values'] = type_obj.enums
            info['enum_name'] = type_obj.name

        # Extract python type if available
        if hasattr(type_obj, 'python_type'):
            try:
                python_type = type_obj.python_type
                info['python_type'] = python_type.__name__
                info['python_module'] = python_type.__module__
            except (AttributeError, TypeError):
                pass

        # Extract custom type parameters
        if info['is_custom']:
            params = {}
            for key, value in inspect.getmembers(type_obj):
                if not key.startswith('_') and not callable(value):
                    try:
                        # Only include serializable values
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            params[key] = value
                    except Exception:
                        continue
            if params:
                info['custom_params'] = params

        return info

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic SQL type as the default target."""
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={"origin_system": self.name})
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: Type) -> CanonicalType:
        """
        Convert a SQLAlchemy type to CanonicalType.
        
        Args:
            type_obj: SQLAlchemy type to convert
            
        Returns:
            CanonicalType representation
        """
        # Handle SQLAlchemy types
        if isinstance(type_obj, (TypeEngine, type)) and hasattr(type_obj, '__visit_name__'):
            # Extract detailed type info
            type_info = self._extract_type_info(type_obj)
            
            # Try to map to a canonical primitive
            type_class = type_obj if isinstance(type_obj, type) else type_obj.__class__
            canonical_name = self._to_canonical_map.get(type_class)

            if canonical_name:
                # Known type - extract parameters
                params = {}
                if 'length' in type_info:
                    params['length'] = type_info['length']
                if 'precision' in type_info:
                    params['precision'] = type_info['precision']
                if 'scale' in type_info:
                    params['scale'] = type_info['scale']
                if 'enum_values' in type_info:
                    params['enum_values'] = type_info['enum_values']
                if 'timezone' in type_info:
                    params['timezone'] = type_info['timezone']

                return CanonicalType(
                    kind="primitive",
                    name=canonical_name,
                    params=params,
                    meta=TypeMeta(data={
                        "origin_system": self.name,
                        "sql_type": type_info
                    })
                )
            else:
                # Unknown type - try to infer from python_type
                inferred_type = "any"
                if 'python_type' in type_info:
                    python_type_map = {
                        'str': 'str',
                        'int': 'int',
                        'float': 'float',
                        'bool': 'bool',
                        'datetime.date': 'date',
                        'datetime.datetime': 'datetime',
                        'datetime.time': 'time',
                        'datetime.timedelta': 'timedelta',
                        'decimal.Decimal': 'decimal',
                        'bytes': 'bytes',
                        'dict': 'dict',
                        'list': 'list'
                    }
                    key = f"{type_info['python_module']}.{type_info['python_type']}"
                    inferred_type = python_type_map.get(key, 'any')

                return CanonicalType(
                    kind="primitive",
                    name=inferred_type,
                    params=type_info.get('custom_params', {}),
                    meta=TypeMeta(data={
                        "origin_system": self.name,
                        "sql_type": type_info,
                        "is_inferred": True
                    })
                )

        # Non-SQLAlchemy type fallback
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={
                "origin_system": self.name,
                "original_type": str(type_obj),
                "original_module": getattr(type_obj, '__module__', None),
                "original_name": getattr(type_obj, '__name__', str(type_obj)),
                "is_fallback": True
            })
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> Type[sa_types.TypeEngine]:
        """
        Reconstructs an SQLAlchemy type from CanonicalType.
        
        Args:
            canonical: The canonical type to convert
            
        Returns:
            SQLAlchemy type
        """
        if canonical.kind != "primitive":
            # For non-primitive types, try to use the original SQL type if available
            meta = canonical.meta.data if canonical.meta else {}
            sql_type_info = meta.get('sql_type', {})
            
            if sql_type_info and not sql_type_info.get('is_custom', False):
                # Try to reconstruct the original SQL type
                type_name = sql_type_info.get('type_name')
                if type_name:
                    try:
                        type_class = getattr(sa_types, type_name.title(), None)
                        if type_class:
                            kwargs = {k: v for k, v in sql_type_info.items() 
                                    if k in ['length', 'precision', 'scale', 'timezone']}
                            return type_class(**kwargs)
                    except Exception:
                        pass
            
            # Fallback to JSON for composite types
            return sa_types.JSON

        # Handle primitive types
        if canonical.name in self._from_canonical_map:
            base_sa_type = self._from_canonical_map[canonical.name]
            kwargs = {}

            # Get original SQL type info if available
            meta = canonical.meta.data if canonical.meta else {}
            sql_type_info = meta.get('sql_type', {})

            # Apply type-specific parameters
            if base_sa_type is sa_types.String:
                # Try original length first, then fall back to MaxLength constraint
                length = sql_type_info.get('length')
                if length is None and 'MaxLength' in canonical.constraints:
                    length = canonical.constraints['MaxLength'].value
                if length is not None:
                    kwargs['length'] = length

            elif base_sa_type is sa_types.Numeric:
                # Use original precision/scale if available
                if 'precision' in sql_type_info:
                    kwargs['precision'] = sql_type_info['precision']
                if 'scale' in sql_type_info:
                    kwargs['scale'] = sql_type_info['scale']

            elif base_sa_type is sa_types.Enum and 'enum_values' in canonical.params:
                enum_values = canonical.params['enum_values']
                enum_name = sql_type_info.get('enum_name', f"{canonical.name}_enum")
                return sa_types.Enum(*enum_values, name=enum_name)

            elif base_sa_type is sa_types.DateTime:
                # Preserve timezone setting
                if 'timezone' in sql_type_info:
                    kwargs['timezone'] = sql_type_info['timezone']

            # Try to instantiate with kwargs
            try:
                return base_sa_type(**kwargs)
            except TypeError:
                return base_sa_type()
        else:
            # Check if we have original SQL type info
            meta = canonical.meta.data if canonical.meta else {}
            sql_type_info = meta.get('sql_type', {})
            
            if sql_type_info and not sql_type_info.get('is_custom', False):
                # Try to reconstruct the original SQL type
                type_name = sql_type_info.get('type_name')
                if type_name:
                    try:
                        type_class = getattr(sa_types, type_name.title(), None)
                        if type_class:
                            kwargs = {k: v for k, v in sql_type_info.items() 
                                    if k in ['length', 'precision', 'scale', 'timezone']}
                            return type_class(**kwargs)
                    except Exception:
                        pass

            # Fallback based on canonical name
            fallback_map = {
                'dict': sa_types.JSON,
                'list': sa_types.JSON,
                'object': sa_types.JSON,
                'any': sa_types.PickleType
            }
            return fallback_map.get(canonical.name, sa_types.String)()

    def detect(self, obj: Any) -> bool:
        """Returns True if the object is an SQLAlchemy type."""
        if sa_types is None:
            return False

        return (
            isinstance(obj, TypeEngine) or
            (isinstance(obj, type) and issubclass(obj, TypeEngine))
        ) 
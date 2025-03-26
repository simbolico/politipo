from typing import Any, Dict, Type, Union, get_origin, get_args, Optional, List, Tuple, Set, FrozenSet
import datetime
import decimal
from functools import lru_cache

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError


class PythonTypeSystem(TypeSystem):
    """Type system implementation for standard Python types."""

    name = "python"

    def get_default_canonical(self) -> CanonicalType:
        """Returns the most generic Python type (Any)."""
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={"origin_system": self.name})
        )

    @lru_cache(maxsize=256)
    def to_canonical(self, type_obj: Type) -> CanonicalType:
        """Converts a Python type to CanonicalType."""
        origin = get_origin(type_obj)
        args = get_args(type_obj)

        # Handle Optional[T] (Union[T, None])
        if origin is Union and type(None) in args:
            inner_type = next((arg for arg in args if arg is not type(None)), Any)
            # Recursively get canonical for inner type
            # Nullability is often contextual, but could add meta if needed
            return self.to_canonical(inner_type) # .with_meta(...)

        # Handle container types
        if origin in (list, List, tuple, Tuple, set, Set, frozenset, FrozenSet):
            container_name = origin.__name__.lower().replace('[]','') # list, tuple, set, frozenset
            if args:
                item_type = self.to_canonical(args[0])
            else: # Handles plain list, tuple, etc.
                item_type = CanonicalType(kind="primitive", name="any")
            return CanonicalType(
                kind="container",
                name=container_name,
                params={"item_type": item_type},
                meta=TypeMeta(data={"origin_system": self.name})
            )

        # Handle dict
        if origin in (dict, Dict):
            if args and len(args) == 2:
                key_type = self.to_canonical(args[0])
                value_type = self.to_canonical(args[1])
            else: # Handles plain dict
                key_type = CanonicalType(kind="primitive", name="any")
                value_type = CanonicalType(kind="primitive", name="any")
            return CanonicalType(
                kind="container",
                name="dict",
                params={"key_type": key_type, "value_type": value_type},
                meta=TypeMeta(data={"origin_system": self.name})
            )

        # Handle primitive types
        primitive_map: Dict[Type, str] = {
            int: "int",
            str: "str",
            float: "float",
            bool: "bool",
            bytes: "bytes",
            datetime.datetime: "datetime",
            datetime.date: "date",
            datetime.time: "time",
            datetime.timedelta: "timedelta",
            decimal.Decimal: "decimal",
            type(None): "null", # Explicit mapping for NoneType
            Any: "any"
        }
        if type_obj in primitive_map:
            return CanonicalType(
                kind="primitive",
                name=primitive_map[type_obj],
                meta=TypeMeta(data={"origin_system": self.name})
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


    @lru_cache(maxsize=256)
    def from_canonical(self, canonical: CanonicalType) -> Type:
        """Reconstructs a Python type from CanonicalType."""
        if canonical.kind == "primitive":
            primitive_map: Dict[str, Type] = {
                "int": int,
                "str": str,
                "float": float,
                "bool": bool,
                "bytes": bytes,
                "datetime": datetime.datetime,
                "date": datetime.date,
                "time": datetime.time,
                "timedelta": datetime.timedelta,
                "decimal": decimal.Decimal,
                "null": type(None),
                "any": Any
            }
            if canonical.name in primitive_map:
                return primitive_map[canonical.name]
            else:
                # Attempt to handle formats stored in params
                if canonical.name == "str" and canonical.params.get("format") == "email":
                    # No standard Python email type, return str
                    return str
                # Add more format handling if needed (UUID etc.)
                return Any # Fallback

        elif canonical.kind == "container":
            container_map: Dict[str, Type] = {
                "list": List,
                "tuple": Tuple,
                "set": Set,
                "frozenset": FrozenSet,
                "dict": Dict,
            }
            if canonical.name in container_map:
                container = container_map[canonical.name]
                if canonical.name == "dict":
                    key_type = self.from_canonical(canonical.params.get("key_type", CanonicalType(kind="primitive", name="any")))
                    value_type = self.from_canonical(canonical.params.get("value_type", CanonicalType(kind="primitive", name="any")))
                    # Ensure Any is used correctly if types are missing
                    key_type = key_type if key_type is not None else Any
                    value_type = value_type if value_type is not None else Any
                    return container[key_type, value_type]
                else:
                    item_type = self.from_canonical(canonical.params.get("item_type", CanonicalType(kind="primitive", name="any")))
                    # Ensure Any is used correctly if item_type is missing
                    item_type = item_type if item_type is not None else Any
                    return container[item_type]

        # Fallback for composite (if needed later) or unknown kinds
        return Any

    def detect(self, obj: Any) -> bool:
        """Returns True if the object is a standard Python type or typing construct."""
        # Check if obj is a type itself
        if isinstance(obj, type):
             # Basic types
            if obj in (int, str, float, bool, bytes, list, dict, tuple, set, frozenset,
                       datetime.datetime, datetime.date, datetime.time, datetime.timedelta,
                       decimal.Decimal, type(None)):
                return True
        # Check typing module generics (List, Dict, Optional, Union, Tuple, Set, etc.)
        if hasattr(obj, '__origin__') and obj.__origin__ is not None:
             # Check if origin is from typing or built-in generics
             typing_origins = (List, Dict, Tuple, Set, FrozenSet, Union, Optional)
             builtin_origins = (list, dict, tuple, set, frozenset) # For Python 3.9+ generics
             if obj.__origin__ in typing_origins or obj.__origin__ in builtin_origins:
                 return True
        # Allow Any
        if obj is Any:
            return True

        return False 
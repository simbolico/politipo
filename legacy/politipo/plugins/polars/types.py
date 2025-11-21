from functools import lru_cache
from typing import Any

try:
    import polars as pl
except ImportError:
    pl = None  # Allow type checking

from politipo.core.errors import PolitipoError
from politipo.core.types import CanonicalType, TypeMeta, TypeSystem


class PolarsTypeSystem(TypeSystem):
    """Type system implementation for Polars data types."""

    name = "polars"

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic DataFrame type as the default Polars target."""
        return CanonicalType(
            kind="composite",
            name="DataFrame",
            params={
                "columns": {},  # Empty columns dict for generic DataFrame
                "schema": {},  # Empty schema for generic DataFrame
            },
            meta=TypeMeta(
                data={"origin_system": self.name, "polars_schema": {}}  # Empty schema metadata
            ),
        )

    def __init__(self):
        if pl is None:
            raise PolitipoError("Polars is not installed. Cannot initialize PolarsTypeSystem.")

        # Mapping from Polars type classes to canonical primitive names
        self._to_canonical_map: dict[type[pl.DataType], str] = {
            # Integer types
            pl.Int8: "int",
            pl.Int16: "int",
            pl.Int32: "int",
            pl.Int64: "int",
            pl.UInt8: "int",
            pl.UInt16: "int",
            pl.UInt32: "int",
            pl.UInt64: "int",  # Map unsigned to int too
            # Float types
            pl.Float32: "float",
            pl.Float64: "float",
            # Other numerics
            pl.Decimal: "decimal",
            # String/Binary
            pl.Utf8: "str",
            pl.Binary: "bytes",
            # Boolean
            pl.Boolean: "bool",
            # Temporal types
            pl.Date: "date",
            pl.Time: "time",
            pl.Datetime: "datetime",
            pl.Duration: "timedelta",
            # Complex types - map to 'any' or specific container/composite?
            pl.List: "list",  # Special handling needed for inner type
            pl.Struct: "struct",  # Map to composite? Needs field info.
            pl.Object: "any",  # Generic object
            pl.Categorical: "str",  # Represent categorical as underlying string type
            pl.Null: "null",
        }

        # Mapping from canonical primitive names to default Polars types
        self._from_canonical_map: dict[str, type[pl.DataType]] = {
            "int": pl.Int64,
            "float": pl.Float64,
            "decimal": pl.Decimal,
            "str": pl.Utf8,
            "bytes": pl.Binary,
            "bool": pl.Boolean,
            "date": pl.Date,
            "time": pl.Time,
            "datetime": pl.Datetime,  # Default to microsecond precision, no timezone
            "timedelta": pl.Duration,  # Default to microsecond unit
            "null": pl.Null,
            "any": pl.Object,
            # Mapping containers back is complex, handled separately
        }

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: type[pl.DataType]) -> CanonicalType:
        """Converts a Polars type to CanonicalType."""
        # Handle instances and classes
        pl_type_class = type_obj if isinstance(type_obj, type) else type(type_obj)

        # Handle List specifically
        if pl_type_class is pl.List:
            inner_type = getattr(type_obj, "inner", None)
            item_canon = (
                self.to_canonical(inner_type)
                if inner_type
                else CanonicalType(kind="primitive", name="any")
            )
            return CanonicalType(
                kind="container",
                name="list",
                params={"item_type": item_canon},
                meta=TypeMeta(data={"origin_system": self.name}),
            )

        # Handle Struct specifically
        if pl_type_class is pl.Struct:
            fields = {}
            if hasattr(type_obj, "fields"):  # Check Polars version for field access
                for field_obj in type_obj.fields:
                    field_name = field_obj.name
                    field_type = field_obj.dtype
                    fields[field_name] = {
                        "type": self.to_canonical(field_type)
                    }  # Required? Constraints?
            return CanonicalType(
                kind="composite",
                name="Struct",
                params={"fields": fields},
                meta=TypeMeta(data={"origin_system": self.name}),
            )

        # Handle primitive/other types
        for pl_class, canonical_name in self._to_canonical_map.items():
            # Use exact match first, then issubclass (though Polars types are often final)
            if pl_type_class is pl_class or issubclass(pl_type_class, pl_class):
                params = {}
                # Extract precision/scale for Decimal
                if pl_type_class is pl.Decimal:
                    # Polars Decimal has precision/scale on instances, might not be on type itself easily
                    # Store info if available on the instance passed
                    if isinstance(type_obj, pl.Decimal):
                        if type_obj.precision:
                            params["precision"] = type_obj.precision
                        if type_obj.scale is not None:
                            params["scale"] = type_obj.scale  # scale can be 0

                # Extract time unit / timezone for Datetime/Duration
                if pl_type_class is pl.Datetime and isinstance(type_obj, pl.Datetime):
                    if type_obj.time_unit:
                        params["time_unit"] = type_obj.time_unit
                    if type_obj.time_zone:
                        params["time_zone"] = type_obj.time_zone
                if pl_type_class is pl.Duration and isinstance(type_obj, pl.Duration):
                    if type_obj.time_unit:
                        params["time_unit"] = type_obj.time_unit

                return CanonicalType(
                    kind="primitive",
                    name=canonical_name,
                    params=params,
                    meta=TypeMeta(data={"origin_system": self.name}),
                )

        # Fallback for unknown Polars types
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={"origin_system": self.name, "original_type": str(type_obj)}),
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> type[pl.DataType]:
        """Reconstructs a Polars type from CanonicalType."""
        if canonical.kind == "primitive":
            if canonical.name in self._from_canonical_map:
                base_pl_type = self._from_canonical_map[canonical.name]
                kwargs = {}  # Polars types usually don't take kwargs at class level like SA

                # Instantiate with parameters if applicable (e.g., Datetime, Duration)
                # Note: Polars usually requires parameters on instance creation, not type definition
                if base_pl_type is pl.Datetime:
                    time_unit = canonical.params.get("time_unit", "us")  # default 'us'
                    time_zone = canonical.params.get("time_zone")
                    return pl.Datetime(time_unit=time_unit, time_zone=time_zone)
                if base_pl_type is pl.Duration:
                    time_unit = canonical.params.get("time_unit", "us")  # default 'us'
                    return pl.Duration(time_unit=time_unit)
                if base_pl_type is pl.Decimal:
                    # Need precision/scale for Decimal instance, maybe return type only?
                    # Polars Decimal needs precision/scale; cannot instantiate without it.
                    # Default or raise error if not provided.
                    precision = canonical.params.get("precision")  # optional
                    scale = canonical.params.get("scale", 0)  # default 0
                    return pl.Decimal(precision=precision, scale=scale)

                return base_pl_type  # Return the type class directly
            else:
                return pl.Object  # Fallback

        elif canonical.kind == "container" and canonical.name == "list":
            item_type = self.from_canonical(
                canonical.params.get("item_type", CanonicalType(kind="primitive", name="any"))
            )
            return pl.List(inner=item_type)

        elif canonical.kind == "composite" and canonical.name == "Struct":
            schema_dict = {}
            for name, field_info in canonical.params.get("fields", {}).items():
                schema_dict[name] = self.from_canonical(field_info["type"])
            return pl.Struct(schema_dict)

        # Fallback for other kinds
        return pl.Object

    def detect(self, obj: Any) -> bool:
        """Returns True if the object is a Polars data type."""
        if pl is None:
            return False
        # Check if it's an instance or a class type derived from DataType
        return isinstance(obj, pl.DataType) or (
            isinstance(obj, type) and issubclass(obj, pl.DataType)
        )

from typing import Any, Dict, Type, Optional, Union
from functools import lru_cache
from decimal import Decimal
import datetime
import re

try:
    import pandera as pa
    from pandera.engines import numpy_engine, pandas_engine # For type mapping
    from pandera.typing import Series, DataFrame
except ImportError:
    pa = None # Allow type checking
    Series = None
    DataFrame = None

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError
# Import constraint classes
from politipo.core.types.constraints import (
    MinValue, MaxValue, MinLength, MaxLength,
    GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual,
    Pattern, MultipleOf, constraint_from_dict, Constraint
)


class PanderaTypeSystem(TypeSystem):
    """Type system implementation for Pandera schemas."""

    name = "pandera"

    def __init__(self):
        if pa is None:
            raise PolitipoError("Pandera is not installed.")

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic DataFrame schema type."""
        return CanonicalType(
            kind="composite",
            name="DataFrameSchema",
            params={"fields": {}},
            meta=TypeMeta(data={
                "origin_system": self.name,
                "schema_type": "dataframe"
            })
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: Union[pa.DataFrameSchema, pa.SeriesSchema]) -> CanonicalType:
        """Converts a Pandera schema to CanonicalType."""
        if not self.detect(type_obj):
            raise ConversionError(f"Object {type_obj} is not a Pandera schema.")

        # Handle Series schema
        if isinstance(type_obj, pa.SeriesSchema):
            return self._series_to_canonical(type_obj)

        # Handle DataFrame schema
        if isinstance(type_obj, pa.DataFrameSchema):
            return self._dataframe_to_canonical(type_obj)

        raise ConversionError(f"Unsupported Pandera schema type: {type(type_obj)}")

    def _series_to_canonical(self, schema: pa.SeriesSchema) -> CanonicalType:
        """Convert a SeriesSchema to CanonicalType."""
        # Extract constraints
        constraints = {}
        if schema.nullable is not None:
            constraints["nullable"] = schema.nullable
        if schema.unique is not None:
            constraints["unique"] = schema.unique
        if schema.coerce is not None:
            constraints["coerce"] = schema.coerce
        if schema.regex is not None:
            constraints["regex"] = schema.regex
        if schema.checks:
            constraints["checks"] = [str(check) for check in schema.checks]

        # Extract dtype info
        dtype_name = str(schema.dtype) if schema.dtype else "object"

        # Build metadata
        meta_data = {
            "origin_system": self.name,
            "schema_type": "series",
            "pandas_dtype": dtype_name
        }

        # Add any custom metadata from schema
        if hasattr(schema, "metadata") and schema.metadata:
            meta_data["custom_metadata"] = schema.metadata

        return CanonicalType(
            kind="primitive",
            name=dtype_name,
            constraints=constraints,
            meta=TypeMeta(data=meta_data)
        )

    def _dataframe_to_canonical(self, schema: pa.DataFrameSchema) -> CanonicalType:
        """Convert a DataFrameSchema to CanonicalType."""
        fields: Dict[str, Dict] = {}

        # Process each column
        for name, column in schema.columns.items():
            # Convert column schema to canonical type
            column_canon = self._series_to_canonical(column.schema)

            # Add field entry with type and constraints
            fields[name] = {
                "type": column_canon,
                "required": not column.nullable if column.nullable is not None else True,
                "constraints": {}
            }

            # Add column-specific constraints
            if column.unique is not None:
                fields[name]["constraints"]["unique"] = column.unique
            if column.coerce is not None:
                fields[name]["constraints"]["coerce"] = column.coerce
            if column.regex is not None:
                fields[name]["constraints"]["regex"] = column.regex
            if column.checks:
                fields[name]["constraints"]["checks"] = [str(check) for check in column.checks]

        # Build metadata
        meta_data = {
            "origin_system": self.name,
            "schema_type": "dataframe",
        }

        # Add index information if present
        if schema.index:
            meta_data["index"] = {
                "names": schema.index.names if isinstance(schema.index, list) else [schema.index.name],
                "dtypes": [str(idx.dtype) for idx in (schema.index if isinstance(schema.index, list) else [schema.index])]
            }

        # Add DataFrame-level constraints
        if schema.unique_column_names is not None:
            meta_data["unique_column_names"] = schema.unique_column_names
        if schema.coerce is not None:
            meta_data["coerce"] = schema.coerce
        if schema.strict is not None:
            meta_data["strict"] = schema.strict
        if schema.ordered is not None:
            meta_data["ordered"] = schema.ordered

        # Add any custom metadata from schema
        if hasattr(schema, "metadata") and schema.metadata:
            meta_data["custom_metadata"] = schema.metadata

        return CanonicalType(
            kind="composite",
            name="DataFrameSchema",
            params={"fields": fields},
            meta=TypeMeta(data=meta_data)
        )

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> Union[pa.DataFrameSchema, pa.SeriesSchema]:
        """Reconstruct a Pandera schema from CanonicalType."""
        if pa is None:
            raise PolitipoError("Pandera is not installed.")

        meta = canonical.meta.data if canonical.meta else {}
        schema_type = meta.get("schema_type")

        if schema_type == "series":
            return self._canonical_to_series(canonical)
        elif schema_type == "dataframe":
            return self._canonical_to_dataframe(canonical)
        else:
            raise ConversionError(f"Unsupported schema type in canonical: {schema_type}")

    def _canonical_to_series(self, canonical: CanonicalType) -> pa.SeriesSchema:
        """Convert CanonicalType to SeriesSchema."""
        if canonical.kind != "primitive":
            raise ConversionError("Series canonical type must be primitive")

        meta = canonical.meta.data if canonical.meta else {}
        constraints = canonical.constraints or {}

        # Create series schema
        return pa.SeriesSchema(
            dtype=meta.get("pandas_dtype", "object"),
            nullable=constraints.get("nullable"),
            unique=constraints.get("unique"),
            coerce=constraints.get("coerce"),
            regex=constraints.get("regex"),
            checks=None,  # Complex to reconstruct checks
            metadata=meta.get("custom_metadata")
        )

    def _canonical_to_dataframe(self, canonical: CanonicalType) -> pa.DataFrameSchema:
        """Convert CanonicalType to DataFrameSchema."""
        if canonical.kind != "composite":
            raise ConversionError("DataFrame canonical type must be composite")

        meta = canonical.meta.data if canonical.meta else {}
        fields = canonical.params.get("fields", {})

        # Convert fields to column schemas
        columns = {}
        for name, field_info in fields.items():
            field_type = field_info["type"]
            field_constraints = field_info.get("constraints", {})

            # Convert field to series schema
            series_schema = self._canonical_to_series(field_type)

            # Create column with constraints
            columns[name] = pa.Column(
                schema=series_schema,
                nullable=not field_info.get("required", True),
                unique=field_constraints.get("unique"),
                coerce=field_constraints.get("coerce"),
                regex=field_constraints.get("regex"),
                checks=None  # Complex to reconstruct checks
            )

        # Create DataFrame schema
        return pa.DataFrameSchema(
            columns=columns,
            index=self._build_index_from_meta(meta.get("index")),
            unique_column_names=meta.get("unique_column_names"),
            coerce=meta.get("coerce"),
            strict=meta.get("strict"),
            ordered=meta.get("ordered"),
            metadata=meta.get("custom_metadata")
        )

    def _build_index_from_meta(self, index_meta: Optional[Dict]) -> Optional[Union[pa.Index, list]]:
        """Helper to reconstruct index schema from metadata."""
        if not index_meta:
            return None

        names = index_meta.get("names", [])
        dtypes = index_meta.get("dtypes", [])

        if len(names) == 1:
            return pa.Index(dtype=dtypes[0], name=names[0])
        
        return [pa.Index(dtype=dtype, name=name) for name, dtype in zip(names, dtypes)]

    def detect(self, obj: Any) -> bool:
        """Returns True if obj is a Pandera schema."""
        return (
            pa is not None and
            isinstance(obj, (pa.DataFrameSchema, pa.SeriesSchema))
        )

    def _pandera_check_to_constraint(self, check: pa.Check) -> Optional[Constraint]:
        """Maps a Pandera Check object to a core Constraint object."""
        stats = check.statistics
        check_name = check.name or check._check_fn.__name__ if hasattr(check, '_check_fn') else None

        if check_name == 'greater_than_or_equal_to' and 'min_value' in stats:
            return MinValue(value=stats['min_value'])
        if check_name == 'less_than_or_equal_to' and 'max_value' in stats:
            return MaxValue(value=stats['max_value'])
        if check_name == 'greater_than' and 'min_value' in stats:
            return GreaterThan(value=stats['min_value'])
        if check_name == 'less_than' and 'max_value' in stats:
            return LessThan(value=stats['max_value'])
        if check_name == 'startswith' and 'string' in stats: # Example check
             # No direct core constraint, maybe map to Pattern?
             return Pattern(regex=f"^{re.escape(stats['string'])}")
        if check_name == 'str_length' and ('min_value' in stats or 'max_value' in stats) :
             # Pandera combines min/max length, need to create separate constraints
             if 'min_value' in stats and stats['min_value'] is not None:
                 # Only return one constraint per check for now? This check needs splitting logic
                 return MinLength(value=stats['min_value'])
             if 'max_value' in stats and stats['max_value'] is not None:
                 return MaxLength(value=stats['max_value'])
             # TODO: Handle combined case better if needed
        if check_name == 'str_matches' and 'pattern' in stats:
            return Pattern(regex=stats['pattern'])
        # Add more mappings for other standard Pandera checks
        # eq, ne, isin, notin, unique, etc. may not have direct Constraint mapping yet

        return None

    def _constraint_to_pandera_check(self, constraint: Constraint) -> Optional[pa.Check]:
         """Maps a core Constraint object back to a Pandera Check."""
         if isinstance(constraint, MinValue):
             return pa.Check.ge(constraint.value)
         if isinstance(constraint, MaxValue):
             return pa.Check.le(constraint.value)
         if isinstance(constraint, MinLength):
             return pa.Check.str_length(min_value=constraint.value)
         if isinstance(constraint, MaxLength):
             return pa.Check.str_length(max_value=constraint.value)
         if isinstance(constraint, GreaterThan):
             return pa.Check.gt(constraint.value)
         if isinstance(constraint, LessThan):
             return pa.Check.lt(constraint.value)
         if isinstance(constraint, GreaterThanOrEqual):
             return pa.Check.ge(constraint.value) # Same as MinValue
         if isinstance(constraint, LessThanOrEqual):
             return pa.Check.le(constraint.value) # Same as MaxValue
         if isinstance(constraint, Pattern):
             return pa.Check.str_matches(constraint.regex)
         if isinstance(constraint, MultipleOf):
             # Pandera doesn't have a built-in multiple_of check? Custom check needed.
             # return pa.Check(lambda x: (Decimal(str(x)) % Decimal(str(constraint.value))) == 0, name="multiple_of")
             return None # Cannot map back directly without custom check support

         return None 
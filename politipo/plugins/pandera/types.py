from typing import Any, Dict, Type, Optional, Union, List, Callable
from functools import lru_cache
from decimal import Decimal
import datetime
import re
import operator

try:
    import pandera as pa
    from pandera.engines import numpy_engine, pandas_engine
    from pandera.typing import Series, DataFrame
    import numpy as np
    import pandas as pd
except ImportError:
    pa = None
    Series = None
    DataFrame = None
    np = None
    pd = None

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types.constraints import (
    MinValue, MaxValue, MinLength, MaxLength,
    GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual,
    Pattern, MultipleOf, OneOf, constraint_from_dict, Constraint
)
from politipo.core.utils import make_hashable, safe_json_dumps


class PanderaTypeSystem(TypeSystem):
    """Type system implementation for Pandera schemas."""

    name = "pandera"

    # Mapping of constraint types to Pandera check methods
    CHECK_MAPPING = {
        GreaterThan: ("greater_than", "min_value"),
        LessThan: ("less_than", "max_value"),
        GreaterThanOrEqual: ("greater_than_or_equal_to", "min_value"),
        LessThanOrEqual: ("less_than_or_equal_to", "max_value"),
        MinValue: ("in_range", "min_value"),
        MaxValue: ("in_range", "max_value"),
        MinLength: ("str_length", "min_value"),
        MaxLength: ("str_length", "max_value"),
        Pattern: ("str_matches", "pattern"),
        OneOf: ("isin", "allowed_values"),
        MultipleOf: (None, None)  # Custom check needed
    }

    # Mapping of Pandera check names to constraint types
    REVERSE_CHECK_MAPPING = {
        "greater_than": GreaterThan,
        "less_than": LessThan,
        "greater_than_or_equal_to": GreaterThanOrEqual,
        "less_than_or_equal_to": LessThanOrEqual,
        "in_range": MinValue,  # Note: This is approximate, depends on which bound is set
        "str_length": MinLength,  # Note: This is approximate, depends on which bound is set
        "str_matches": Pattern,
        "isin": OneOf,
    }

    def __init__(self):
        if pa is None:
            raise PolitipoError("Pandera is not installed.")
        self._cache = {}  # Initialize cache

    def _get_cache_key(self, type_obj: Union[pa.DataFrameSchema, pa.SeriesSchema, Type[DataFrame], Type[Series]]) -> str:
        """Create a hashable key for caching schema conversions."""
        if isinstance(type_obj, (pa.DataFrameSchema, pa.SeriesSchema)):
            # For schema objects, use class name and schema details
            schema_details = []
            if hasattr(type_obj, 'dtype'):
                schema_details.append(str(type_obj.dtype))
            if hasattr(type_obj, 'columns'):
                schema_details.extend(sorted(type_obj.columns.keys()))
            return f"{type_obj.__class__.__name__}:{':'.join(schema_details)}"
        # For types, use string representation
        return str(type_obj)

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

    def to_canonical(self, type_obj: Union[pa.DataFrameSchema, pa.SeriesSchema, Type[DataFrame], Type[Series]]) -> CanonicalType:
        """
        Converts a Pandera schema or type to CanonicalType.
        
        Args:
            type_obj: A Pandera schema object (DataFrameSchema/SeriesSchema) or type annotation (DataFrame/Series)
            
        Returns:
            CanonicalType representation of the schema
            
        Raises:
            ConversionError: If the object is not a supported Pandera type
        """
        # Try to get from cache first
        cache_key = self._get_cache_key(type_obj)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.detect(type_obj):
            raise ConversionError(f"Object {type_obj} is not a Pandera schema or type.")

        # Handle Series schema and type
        if isinstance(type_obj, pa.SeriesSchema) or (isinstance(type_obj, type) and issubclass(type_obj, Series)):
            result = self._series_to_canonical(type_obj)
        # Handle DataFrame schema and type
        elif isinstance(type_obj, pa.DataFrameSchema) or (isinstance(type_obj, type) and issubclass(type_obj, DataFrame)):
            result = self._dataframe_to_canonical(type_obj)
        else:
            raise ConversionError(f"Unsupported Pandera type: {type(type_obj)}")

        # Cache the result
        self._cache[cache_key] = result
        return result

    def _series_to_canonical(self, schema: Union[pa.SeriesSchema, Type[Series]]) -> CanonicalType:
        """Convert a SeriesSchema or Series type to CanonicalType."""
        # Extract schema from Series type if needed
        if isinstance(schema, type) and issubclass(schema, Series):
            schema = getattr(schema, "__schema__", pa.SeriesSchema())

        constraints = {}
        
        # Extract basic constraints
        if schema.nullable is not None:
            constraints["nullable"] = schema.nullable
        if schema.unique is not None:
            constraints["unique"] = schema.unique
        if hasattr(schema, 'coerce') and schema.coerce is not None:
            constraints["coerce"] = schema.coerce
            
        # Check for regex attribute (pattern constraint)
        if hasattr(schema, 'regex') and schema.regex is not None:
            constraints["pattern"] = schema.regex

        # Process validation checks
        if schema.checks:
            for check in schema.checks:
                constraint = self._pandera_check_to_constraint(check)
                if constraint:
                    constraints.update(constraint.to_dict())

        # Extract dtype info
        dtype_name = str(schema.dtype) if schema.dtype else "object"

        # Build metadata
        meta_data = {
            "origin_system": self.name,
            "schema_type": "series",
            "pandas_dtype": dtype_name
        }

        # Add any custom metadata
        if hasattr(schema, "metadata") and schema.metadata:
            meta_data["custom_metadata"] = schema.metadata

        return CanonicalType(
            kind="primitive",
            name=self._map_dtype_to_canonical(dtype_name),
            constraints=constraints,
            meta=TypeMeta(data=meta_data)
        )

    def _dataframe_to_canonical(self, schema: Union[pa.DataFrameSchema, Type[DataFrame]]) -> CanonicalType:
        """Convert a DataFrameSchema or DataFrame type to CanonicalType."""
        # Extract schema from DataFrame type if needed
        if isinstance(schema, type) and issubclass(schema, DataFrame):
            schema = getattr(schema, "__schema__", pa.DataFrameSchema())

        fields: Dict[str, Dict] = {}

        # Process each column
        for name, column in schema.columns.items():
            try:
                # Convert each Column, creating a SeriesSchema or using existing schema
                series_kwargs = {}

                # Only pass attributes that exist, defaulting to None
                series_kwargs['dtype'] = getattr(column, 'dtype', None)
                series_kwargs['checks'] = getattr(column, 'checks', None)
                series_kwargs['nullable'] = getattr(column, 'nullable', None)
                series_kwargs['coerce'] = getattr(column, 'coerce', None)
                series_kwargs['name'] = name

                series_schema = pa.SeriesSchema(**{k: v for k, v in series_kwargs.items() if v is not None})
                column_canon = self._series_to_canonical(series_schema)

            except Exception as e:
                print(f"Warning: Failed to create SeriesSchema for column '{name}': {e}")
                column_canon = CanonicalType(
                    kind="primitive",
                    name=self._map_dtype_to_canonical(str(getattr(column, 'dtype', "object"))),  # Handle missing dtype
                    meta=TypeMeta(data={"pandas_dtype": str(getattr(column, 'dtype', "object"))})
                )

            # --- Consolidated Constraint Handling ---
            field_constraints = {
                "nullable": getattr(column, "nullable", None),
                "unique": getattr(column, "unique", None),
                "coerce": getattr(column, "coerce", None),
                "pattern": getattr(column, "regex", None),  # Directly get regex if available
                **self._extract_checks_as_constraints(getattr(column, "checks", None))
            }
            field_constraints = {k: v for k, v in field_constraints.items() if v is not None}  # Remove None values
            # --- End Consolidated Constraint Handling ---

            fields[name] = {
                "type": column_canon,
                "required": not field_constraints.get("nullable", True),  # Default to required if no nullable
                "constraints": field_constraints,
                "description": getattr(column, "description", None)
            }

        # Build metadata
        meta_data = {
            "origin_system": self.name,
            "schema_type": "dataframe",
        }

        # Add index information
        if schema.index:
            meta_data["index"] = {
                "names": schema.index.names if isinstance(schema.index, list) else [schema.index.name],
                "dtypes": [str(idx.dtype) for idx in (schema.index if isinstance(schema.index, list) else [schema.index])]
            }

        # Add DataFrame-level constraints and metadata with safe defaults
        for key in ["unique_column_names", "coerce", "strict", "ordered", "metadata"]:
            value = getattr(schema, key, None)
            if value is not None:
                meta_data[key] = value

        return CanonicalType(
            kind="composite",
            name="DataFrameSchema",
            params={"fields": fields},
            meta=TypeMeta(data=meta_data)
        )

    def _extract_checks_as_constraints(self, checks: Optional[List[pa.Check]]) -> Dict[str, Any]:
        """Helper to extract constraints from a list of Pandera checks."""
        constraints = {}
        if checks:
            for check in checks:
                constraint = self._pandera_check_to_constraint(check)
                if constraint:
                    constraints.update(constraint.to_dict())
        return constraints

    def _get_canonical_cache_key(self, canonical: CanonicalType) -> str:
        """Create a hashable key for a CanonicalType object."""
        # Use our make_hashable utility to create a stable hash representation
        try:
            # For dictionaries and complex objects, this creates a hashable representation
            hashable_repr = make_hashable({
                "kind": canonical.kind,
                "name": canonical.name,
                "meta": canonical.meta.data if canonical.meta else {},
                # Include a summary of params rather than full details
                "params_summary": {
                    "num_fields": len(canonical.params.get("fields", {})) if "fields" in canonical.params else 0,
                    "field_names": sorted(canonical.params.get("fields", {}).keys()) if "fields" in canonical.params else []
                }
            })
            # Convert the hashable representation to a string
            return f"canonical:{hash(hashable_repr)}"
        except Exception:
            # Fallback: use a simplified string representation
            return f"canonical_fallback:{canonical.kind}:{canonical.name}"

    def from_canonical(self, canonical: CanonicalType) -> Union[pa.DataFrameSchema, pa.SeriesSchema]:
        """
        Reconstruct a Pandera schema from CanonicalType.
        
        Args:
            canonical: The canonical type representation
            
        Returns:
            A Pandera schema object (DataFrameSchema or SeriesSchema)
            
        Raises:
            ConversionError: If reconstruction fails
        """
        if pa is None:
            raise PolitipoError("Pandera is not installed.")
            
        # Create a cache key based on canonical type properties
        cache_key = self._get_canonical_cache_key(canonical)
            
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        meta = canonical.meta.data if canonical.meta else {}
        schema_type = meta.get("schema_type")

        # Default to dataframe schema for composite types if schema_type is missing
        if schema_type is None:
            if canonical.kind == "composite":
                schema_type = "dataframe"
            elif canonical.kind == "primitive":
                schema_type = "series"
            else:
                # For other kinds, make a best guess based on structure
                if "fields" in canonical.params:
                    schema_type = "dataframe"
                else:
                    schema_type = "series"

        if schema_type == "series":
            result = self._canonical_to_series(canonical)
        elif schema_type == "dataframe":
            result = self._canonical_to_dataframe(canonical)
        else:
            raise ConversionError(f"Unsupported schema type in canonical: {schema_type}")
            
        # Cache the result
        self._cache[cache_key] = result
        return result

    def _canonical_to_series(self, canonical: CanonicalType) -> pa.SeriesSchema:
        """Convert CanonicalType to SeriesSchema."""
        if canonical.kind != "primitive":
            raise ConversionError("Series canonical type must be primitive")

        meta = canonical.meta.data if canonical.meta else {}
        constraints = canonical.constraints or {}
        
        # Extract parameters using a compatibility approach
        kwargs = {
            "dtype": meta.get("pandas_dtype", "object"),
            "nullable": constraints.get("nullable"),
            "unique": constraints.get("unique"),
            "coerce": constraints.get("coerce"),
        }
        
        # Remove None values to avoid unexpected keyword argument errors
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Convert constraints to checks
        checks = []
        for constraint_name, constraint_data in constraints.items():
            if constraint_name in ("nullable", "unique", "coerce"):
                continue
                
            # Handle pattern constraint specifically
            if constraint_name == "pattern" and constraint_data:
                try:
                    # Add pattern as a str_matches check
                    pattern = constraint_data if isinstance(constraint_data, str) else constraint_data.get("pattern", "")
                    if pattern:
                        checks.append(pa.Check.str_matches(pattern))
                except Exception as e:
                    # Log warning or handle error as needed
                    print(f"Warning: Failed to add pattern check: {e}")
                continue
                
            # Handle min_length constraint
            if constraint_name == "MinLength" and constraint_data:
                try:
                    # Extract value correctly based on whether it's a dict or a Constraint object
                    min_value = None
                    if hasattr(constraint_data, 'value'):
                        min_value = constraint_data.value
                    elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                        min_value = constraint_data['value']
                    elif isinstance(constraint_data, (int, float)):
                        min_value = constraint_data
                        
                    if min_value is not None:
                        checks.append(pa.Check.str_length(min_value=min_value))
                except Exception as e:
                    print(f"Warning: Failed to add min_length check: {e}")
                continue
                
            # Handle max_length constraint
            if constraint_name == "MaxLength" and constraint_data:
                try:
                    # Extract value correctly
                    max_value = None
                    if hasattr(constraint_data, 'value'):
                        max_value = constraint_data.value
                    elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                        max_value = constraint_data['value']
                    elif isinstance(constraint_data, (int, float)):
                        max_value = constraint_data
                        
                    if max_value is not None:
                        checks.append(pa.Check.str_length(max_value=max_value))
                except Exception as e:
                    print(f"Warning: Failed to add max_length check: {e}")
                continue
                
            # Handle min_value (ge) constraint
            if constraint_name == "MinValue" and constraint_data:
                try:
                    # Extract value correctly
                    min_value = None
                    if hasattr(constraint_data, 'value'):
                        min_value = constraint_data.value
                    elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                        min_value = constraint_data['value']
                    elif isinstance(constraint_data, (int, float)):
                        min_value = constraint_data
                        
                    if min_value is not None:
                        # Use greater_than_or_equal_to instead of in_range
                        checks.append(pa.Check.greater_than_or_equal_to(min_value))
                except Exception as e:
                    print(f"Warning: Failed to add min_value check: {e}")
                continue
                
            # Handle max_value (le) constraint
            if constraint_name == "MaxValue" and constraint_data:
                try:
                    # Extract value correctly
                    max_value = None
                    if hasattr(constraint_data, 'value'):
                        max_value = constraint_data.value
                    elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                        max_value = constraint_data['value']
                    elif isinstance(constraint_data, (int, float)):
                        max_value = constraint_data
                        
                    if max_value is not None:
                        # Use less_than_or_equal_to instead of in_range
                        checks.append(pa.Check.less_than_or_equal_to(max_value))
                except Exception as e:
                    print(f"Warning: Failed to add max_value check: {e}")
                continue
                
            try:
                # General approach for other constraints
                constraint_dict = {"type": constraint_name}
                if isinstance(constraint_data, dict):
                    constraint_dict.update(constraint_data)
                else:
                    constraint_dict["value"] = constraint_data
                
                constraint = constraint_from_dict(constraint_dict)
                if constraint:
                    check = self._constraint_to_pandera_check(constraint)
                    if check:
                        checks.append(check)
            except Exception as e:
                # Log warning or handle error as needed
                print(f"Warning: Failed to convert constraint {constraint_name}: {e}")
                continue

        # Add checks to kwargs only if they exist
        if checks:
            kwargs["checks"] = checks
        
        # Add metadata if it exists
        if meta.get("custom_metadata"):
            kwargs["metadata"] = meta.get("custom_metadata")
        
        try:
            # Create series schema with validated kwargs
            return pa.SeriesSchema(**kwargs)
        except Exception as e:
            # If that fails, try a minimal approach
            print(f"Warning: Error creating full SeriesSchema: {e}")
            return pa.SeriesSchema(dtype=kwargs.get("dtype", "object"))
            
    def _canonical_to_dataframe(self, canonical: CanonicalType) -> pa.DataFrameSchema:
        """Convert CanonicalType to DataFrameSchema."""
        if canonical.kind != "composite":
            raise ConversionError("DataFrame canonical type must be composite")

        meta = canonical.meta.data if canonical.meta else {}
        fields = canonical.params.get("fields", {})

        # Convert fields to column schemas
        columns = {}
        for name, field_info in fields.items():
            try:
                field_type = field_info["type"]
                field_constraints = field_info.get("constraints", {})

                # Try to get the base dtype from the field type
                dtype = "object"  # Default fallback
                if field_type.kind == "primitive":
                    # Map primitive types to proper dtypes
                    type_name = field_type.name
                    if type_name == "int":
                        dtype = "int64"
                    elif type_name == "float":
                        dtype = "float64"
                    elif type_name == "bool":
                        dtype = "bool"
                    elif type_name == "str":
                        dtype = "string"
                    elif type_name == "datetime":
                        dtype = "datetime64[ns]"

                # Extract column parameters using a compatibility approach
                col_kwargs = {
                    "dtype": dtype,
                    "nullable": not field_info.get("required", True),
                    "unique": field_constraints.get("unique"),
                    "coerce": field_constraints.get("coerce"),
                }
                
                # Remove None values to avoid unexpected keyword argument errors
                col_kwargs = {k: v for k, v in col_kwargs.items() if v is not None}
                
                # Add description if available
                if field_info.get("description"):
                    col_kwargs["description"] = field_info.get("description")

                # Convert constraints to checks
                checks = []
                for constraint_name, constraint_data in field_constraints.items():
                    if constraint_name in ("nullable", "unique", "coerce"):
                        continue
                        
                    # Handle specific constraints based on data type
                    if constraint_name in ("MinLength", "min_length") and dtype == "string":
                        try:
                            # Extract value correctly based on whether it's a dict, Constraint object, or primitive value
                            min_value = None
                            if hasattr(constraint_data, 'value'):
                                min_value = constraint_data.value
                            elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                                min_value = constraint_data['value']
                            elif isinstance(constraint_data, (int, float)):
                                min_value = constraint_data
                                
                            if min_value is not None:
                                checks.append(pa.Check.str_length(min_value=min_value))
                        except Exception as e:
                            print(f"Warning: Failed to add min_length check for column '{name}': {e}")
                        continue
                        
                    if constraint_name in ("MaxLength", "max_length") and dtype == "string":
                        try:
                            # Extract value correctly based on whether it's a dict, Constraint object, or primitive value
                            max_value = None
                            if hasattr(constraint_data, 'value'):
                                max_value = constraint_data.value
                            elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                                max_value = constraint_data['value']
                            elif isinstance(constraint_data, (int, float)):
                                max_value = constraint_data
                                
                            if max_value is not None:
                                checks.append(pa.Check.str_length(max_value=max_value))
                        except Exception as e:
                            print(f"Warning: Failed to add max_length check for column '{name}': {e}")
                        continue
                    
                    # Handle min_value (ge) constraint
                    if constraint_name in ("MinValue", "ge", "GreaterThanOrEqual") and dtype in ("int64", "float64"):
                        try:
                            # Extract value correctly based on whether it's a dict, Constraint object, or primitive value
                            min_value = None
                            if hasattr(constraint_data, 'value'):
                                min_value = constraint_data.value
                            elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                                min_value = constraint_data['value']
                            elif isinstance(constraint_data, (int, float)):
                                min_value = constraint_data
                                
                            if min_value is not None:
                                # Use greater_than_or_equal_to instead of in_range
                                checks.append(pa.Check.greater_than_or_equal_to(min_value))
                        except Exception as e:
                            print(f"Warning: Failed to add min_value check for column '{name}': {e}")
                        continue
                        
                    # Handle max_value (le) constraint
                    if constraint_name in ("MaxValue", "le", "LessThanOrEqual") and dtype in ("int64", "float64"):
                        try:
                            # Extract value correctly based on whether it's a dict, Constraint object, or primitive value
                            max_value = None
                            if hasattr(constraint_data, 'value'):
                                max_value = constraint_data.value
                            elif isinstance(constraint_data, dict) and 'value' in constraint_data:
                                max_value = constraint_data['value']
                            elif isinstance(constraint_data, (int, float)):
                                max_value = constraint_data
                                
                            if max_value is not None:
                                # Use less_than_or_equal_to instead of in_range
                                checks.append(pa.Check.less_than_or_equal_to(max_value))
                        except Exception as e:
                            print(f"Warning: Failed to add max_value check for column '{name}': {e}")
                        continue
                        
                    # Handle pattern constraint specifically
                    if constraint_name in ("Pattern", "pattern") and dtype == "string":
                        try:
                            # Add pattern as a str_matches check
                            pattern = constraint_data if isinstance(constraint_data, str) else constraint_data.get("pattern", "")
                            if pattern:
                                checks.append(pa.Check.str_matches(pattern))
                        except Exception as e:
                            # Log warning or handle error as needed
                            print(f"Warning: Failed to add pattern check for column '{name}': {e}")
                        continue
                        
                    # Try generic constraint conversion as a fallback
                    try:
                        # Properly construct the constraint dictionary
                        constraint_dict = {"type": constraint_name}
                        if isinstance(constraint_data, dict):
                            constraint_dict.update(constraint_data)
                        else:
                            constraint_dict["value"] = constraint_data
                        
                        constraint = constraint_from_dict(constraint_dict)
                        if constraint:
                            check = self._constraint_to_pandera_check(constraint)
                            if check:
                                checks.append(check)
                    except Exception as e:
                        # Log warning or handle error as needed
                        print(f"Warning: Failed to convert generic constraint {constraint_name} for column '{name}': {e}")
                        continue

                # Add checks to kwargs only if they exist
                if checks:
                    col_kwargs["checks"] = checks

                try:
                    # Create column with constraints
                    columns[name] = pa.Column(**col_kwargs)
                except Exception as e:
                    # Fallback to simpler column if that fails
                    print(f"Warning: Error creating full Column for '{name}': {e}")
                    columns[name] = pa.Column(dtype=dtype)

            except Exception as e:
                print(f"Warning: Error creating column '{name}': {e}")
                # Add a basic fallback column
                columns[name] = pa.Column(dtype="object")

        # Extract schema parameters with compatibility
        df_kwargs = {
            "columns": columns,
            "coerce": True  # Add coerce=True to ensure type conversion
        }
        
        # Add basic metadata if available
        for key in ["unique_column_names", "strict", "ordered"]:
            if key in meta and meta[key] is not None:
                df_kwargs[key] = meta[key]
                
        # Add custom metadata if available
        if meta.get("custom_metadata"):
            df_kwargs["metadata"] = meta.get("custom_metadata")

        # Build index schema if specified
        try:
            if "index" in meta:
                df_kwargs["index"] = self._build_index_from_meta(meta["index"])
        except Exception as e:
            print(f"Warning: Error creating index: {e}")

        try:
            # Create DataFrame schema
            return pa.DataFrameSchema(**df_kwargs)
        except Exception as e:
            # Fallback to a minimal schema if that fails
            print(f"Warning: Error creating full DataFrameSchema: {e}")
            return pa.DataFrameSchema(columns=columns)

    def _build_index_from_meta(self, index_meta: Optional[Dict]) -> Optional[Union[pa.Index, List[pa.Index]]]:
        """Helper to reconstruct index schema from metadata."""
        if not index_meta:
            return None

        names = index_meta.get("names", [])
        dtypes = index_meta.get("dtypes", [])

        if len(names) == 1:
            return pa.Index(dtype=dtypes[0], name=names[0])
        
        return [pa.Index(dtype=dtype, name=name) for name, dtype in zip(names, dtypes)]

    def detect(self, obj: Any) -> bool:
        """Returns True if obj is a Pandera schema or type."""
        return (
            pa is not None and (
                isinstance(obj, (pa.DataFrameSchema, pa.SeriesSchema)) or
                (isinstance(obj, type) and issubclass(obj, (Series, DataFrame)))
            )
        )

    def _map_dtype_to_canonical(self, dtype_str: str) -> str:
        """Map pandas/numpy dtype strings to canonical type names."""
        dtype_map = {
            "int": "int",
            "int64": "int",
            "float": "float",
            "float64": "float",
            "bool": "bool",
            "str": "str",
            "object": "any",
            "datetime64": "datetime",
            "timedelta64": "timedelta",
            "category": "str",
            "string": "str"
        }
        
        # Handle numpy/pandas dtype strings
        dtype_str = str(dtype_str).lower()
        for key in dtype_map:
            if key in dtype_str:
                return dtype_map[key]
        return "any"

    def _pandera_check_to_constraint(self, check: pa.Check) -> Optional[Constraint]:
        """
        Convert a Pandera Check to a Constraint.
        
        Args:
            check: The Pandera check to convert
            
        Returns:
            A Constraint object, or None if conversion is not possible
        """
        if not check:
            return None

        # Try to get the check name
        check_name = None
        try:
            check_name = check.name or (check.check_fn if isinstance(check.check_fn, str) else None)
        except AttributeError:
            # If check doesn't have these attributes, try to get name from repr
            try:
                check_str = str(check)
                if 'greater_than_or_equal_to' in check_str:
                    check_name = 'greater_than_or_equal_to'
                elif 'less_than_or_equal_to' in check_str:
                    check_name = 'less_than_or_equal_to'
                elif 'str_length' in check_str:
                    check_name = 'str_length'
                elif 'str_matches' in check_str:
                    check_name = 'str_matches'
                elif 'isin' in check_str:
                    check_name = 'isin'
                elif 'in_range' in check_str:
                    check_name = 'in_range'
            except:
                pass

        if not check_name:
            return None

        # Handle custom checks
        if check_name.startswith("multiple_of_"):
            try:
                value = float(check_name.split("_")[-1])
                return MultipleOf(value=value)
            except (ValueError, IndexError):
                return None

        # Get constraint type from mapping
        constraint_type = self.REVERSE_CHECK_MAPPING.get(check_name)
        if not constraint_type:
            return None

        # Try to extract check_kwargs safely
        try:
            check_kwargs = check.check_kwargs or {}
        except AttributeError:
            # If check doesn't have check_kwargs, try to extract values from repr
            check_kwargs = {}
            check_str = str(check)
            
            # Extract values based on check name
            if check_name == 'greater_than_or_equal_to':
                try:
                    value = float(check_str.split('greater_than_or_equal_to(')[1].split(')')[0])
                    check_kwargs['min_value'] = value
                except:
                    pass
            elif check_name == 'less_than_or_equal_to':
                try:
                    value = float(check_str.split('less_than_or_equal_to(')[1].split(')')[0])
                    check_kwargs['max_value'] = value
                except:
                    pass
            elif check_name == 'str_length':
                if 'min_value=' in check_str:
                    try:
                        value = int(check_str.split('min_value=')[1].split(',')[0].strip(')'))
                        check_kwargs['min_value'] = value
                    except:
                        pass
                if 'max_value=' in check_str:
                    try:
                        value = int(check_str.split('max_value=')[1].split(',')[0].strip(')'))
                        check_kwargs['max_value'] = value
                    except:
                        pass
        
        # Handle special cases
        if check_name == "in_range":
            min_val = check_kwargs.get("min_value")
            max_val = check_kwargs.get("max_value")
            if min_val is not None:
                return MinValue(value=min_val)
            elif max_val is not None:
                return MaxValue(value=max_val)
            return None
        elif check_name == "str_length":
            min_len = check_kwargs.get("min_value")
            max_len = check_kwargs.get("max_value")
            if min_len is not None:
                return MinLength(value=min_len)
            elif max_len is not None:
                return MaxLength(value=max_len)
            return None
        elif check_name == "isin":
            allowed_values = check_kwargs.get("allowed_values", [])
            return OneOf(allowed_values=allowed_values)
        elif check_name == "str_matches":
            pattern = check_kwargs.get("pattern", "")
            return Pattern(pattern=pattern)
        elif check_name == "greater_than_or_equal_to":
            if "min_value" in check_kwargs:
                return MinValue(value=check_kwargs["min_value"])
            return None
        elif check_name == "less_than_or_equal_to":
            if "max_value" in check_kwargs:
                return MaxValue(value=check_kwargs["max_value"])
            return None

        # Standard case: get value using the first available kwarg
        for key in ["min_value", "max_value", "value"]:
            if key in check_kwargs:
                return constraint_type(value=check_kwargs[key])

        return None

    def _constraint_to_pandera_check(self, constraint: Constraint) -> Optional[pa.Check]:
        """
        Convert a Constraint to a Pandera Check.
        
        Args:
            constraint: The constraint to convert
            
        Returns:
            A Pandera Check object, or None if conversion is not possible
        """
        if not isinstance(constraint, Constraint):
            return None

        constraint_type = type(constraint)
        if constraint_type not in self.CHECK_MAPPING:
            return None

        check_name, value_key = self.CHECK_MAPPING[constraint_type]
        
        # Handle special cases first
        if constraint_type == MultipleOf:
            # Create custom check for MultipleOf
            def multiple_of_check(x):
                if pd.isna(x):
                    return True
                try:
                    return float(x) % float(constraint.value) == 0
                except (TypeError, ValueError):
                    return False
            return pa.Check(multiple_of_check, name=f"multiple_of_{constraint.value}")

        if check_name is None:
            return None

        # Get the check method from Pandera
        check_method = getattr(pa.Check, check_name, None)
        if check_method is None:
            return None

        # Handle special cases for different constraint types
        if constraint_type in (MinValue, MaxValue):
            kwargs = {}
            if constraint_type == MinValue:
                kwargs["min_value"] = constraint.value
            else:
                kwargs["max_value"] = constraint.value
            return check_method(**kwargs)
        elif constraint_type in (MinLength, MaxLength):
            kwargs = {}
            if constraint_type == MinLength:
                kwargs["min_value"] = constraint.value
            else:
                kwargs["max_value"] = constraint.value
            return check_method(**kwargs)
        elif constraint_type == Pattern:
            return check_method(constraint.pattern)
        elif constraint_type == OneOf:
            return check_method(constraint.allowed_values)
        else:
            # Standard case: pass the constraint value using the mapped key
            return check_method(**{value_key: constraint.value})
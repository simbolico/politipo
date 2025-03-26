# ./politipo/plugins/pandas/types.py

from typing import Any, Dict, Type, Union # Added Union
import pandas as pd
import numpy as np
from functools import lru_cache # Added lru_cache

from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError # Added error


class PandasTypeSystem(TypeSystem):
    """Type system implementation for Pandas DataFrames with enhanced dtype support."""

    name = "pandas"

    def __init__(self):
         # Optional: Check if pandas is installed
         if not hasattr(pd, 'DataFrame'):
             raise PolitipoError("Pandas is not installed. Cannot initialize PandasTypeSystem.")

    @lru_cache(maxsize=128)
    def to_canonical(self, df_or_type: Union[pd.DataFrame, Type, np.dtype, str]) -> CanonicalType:
        """
        Convert a Pandas DataFrame structure, dtype, or type string to canonical type.
        If a DataFrame is passed, represents the structure.
        If a dtype or type string is passed, represents that single type.
        """
        if isinstance(df_or_type, pd.DataFrame):
            # Represent DataFrame structure
            df = df_or_type
            return CanonicalType(
                kind="composite",
                name="DataFrame",
                params={
                    "columns": {
                        col: self._dtype_to_canonical(df[col].dtype)
                        for col in df.columns
                    }
                },
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "pandas_index": {
                        "name": df.index.name,
                        "dtype": str(df.index.dtype)
                    }
                })
            )
        else:
            # Represent a single dtype
            try:
                # Normalize input to a numpy dtype object
                if isinstance(df_or_type, str):
                    dtype = np.dtype(df_or_type)
                elif isinstance(df_or_type, type) and df_or_type in (int, float, str, bool):
                    # Handle basic python types sometimes used with pandas
                    map_py_pd = {int: np.int64, float: np.float64, str: object, bool: np.bool_}
                    dtype = np.dtype(map_py_pd[df_or_type])
                elif isinstance(df_or_type, pd.api.extensions.ExtensionDtype):
                    # Handle ExtensionDtypes (like StringDtype, Int64Dtype)
                    # Map them to their canonical equivalent
                    if isinstance(df_or_type, pd.StringDtype):
                        return CanonicalType(
                            kind="primitive",
                            name="str",
                            meta=TypeMeta(data={
                                "origin_system": self.name,
                                "original_dtype": str(df_or_type)
                            })
                        )
                    if isinstance(df_or_type, (pd.Int64Dtype, pd.Int32Dtype, pd.Int16Dtype, pd.Int8Dtype,
                                            pd.UInt64Dtype, pd.UInt32Dtype, pd.UInt16Dtype, pd.UInt8Dtype)):
                        return CanonicalType(
                            kind="primitive",
                            name="int",
                            meta=TypeMeta(data={
                                "origin_system": self.name,
                                "original_dtype": str(df_or_type)
                            })
                        )
                    if isinstance(df_or_type, (pd.Float64Dtype, pd.Float32Dtype)):
                        return CanonicalType(
                            kind="primitive",
                            name="float",
                            meta=TypeMeta(data={
                                "origin_system": self.name,
                                "original_dtype": str(df_or_type)
                            })
                        )
                    if isinstance(df_or_type, pd.BooleanDtype):
                        return CanonicalType(
                            kind="primitive",
                            name="bool",
                            meta=TypeMeta(data={
                                "origin_system": self.name,
                                "original_dtype": str(df_or_type)
                            })
                        )
                    if isinstance(df_or_type, pd.CategoricalDtype):
                        # Recursive call for category values dtype
                        cat_dtype = df_or_type.categories.dtype
                        return CanonicalType(
                            kind="container",
                            name="category",
                            params={"item_type": self._dtype_to_canonical(cat_dtype)},
                            meta=TypeMeta(data={
                                "origin_system": self.name,
                                "original_dtype": str(df_or_type)
                            })
                        )
                    if isinstance(df_or_type, pd.DatetimeTZDtype):
                        params = {'time_zone': str(df_or_type.tz)}
                        return CanonicalType(
                            kind="primitive",
                            name="datetime",
                            params=params,
                            meta=TypeMeta(data={
                                "origin_system": self.name,
                                "original_dtype": str(df_or_type)
                            })
                        )
                    # Add PeriodDtype etc. if needed
                    return CanonicalType(
                        kind="primitive",
                        name="any",
                        meta=TypeMeta(data={
                            "origin_system": self.name,
                            "original_dtype": str(df_or_type)
                        })
                    )
                elif isinstance(df_or_type, np.dtype):
                    dtype = df_or_type
                else:
                    raise TypeError(f"Input must be DataFrame, dtype or type string, got {type(df_or_type)}")

                return self._dtype_to_canonical(dtype)

            except Exception as e:
                # Fallback case - ensure TypeMeta is used
                return CanonicalType(
                    kind="primitive",
                    name="any",
                    meta=TypeMeta(data={
                        "origin_system": self.name,
                        "original_type": str(df_or_type)
                    })
                )


    @lru_cache(maxsize=128) # Added cache
    def from_canonical(self, canonical: CanonicalType) -> Union[Type[pd.DataFrame], np.dtype, str]:
        """
        Create an empty DataFrame structure, or get Pandas dtype/string from canonical.
        """
        if canonical.kind == "composite" and canonical.name == "DataFrame":
            # Return the DataFrame type itself, structure is in params
            return pd.DataFrame # Or potentially a factory function?
        else:
             # Return a dtype object or string representation
             try:
                 return self._canonical_to_dtype(canonical)
             except Exception as e:
                 raise ConversionError(f"Failed to convert canonical type '{canonical}' to Pandas dtype: {e}") from e


    def detect(self, obj: Any) -> bool:
        """Check if object is a Pandas DataFrame, dtype, or common pandas type string."""
        if isinstance(obj, pd.DataFrame): return True
        if isinstance(obj, np.dtype): return True
        if isinstance(obj, pd.api.extensions.ExtensionDtype): return True
        # Check common string representations used in pandas
        if isinstance(obj, str) and obj in ('int64', 'float64', 'bool', 'datetime64[ns]', 'string', 'object',
                                             'Int64', 'Float64', 'boolean', 'category'):
             return True
        # Check if it's the DataFrame class itself
        if obj is pd.DataFrame: return True
        return False

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic DataFrame type as the default Pandas target."""
        return CanonicalType(
            kind="composite",
            name="DataFrame",
            params={
                "columns": {},  # Empty columns dict for generic DataFrame
                "index": {"type": CanonicalType(kind="primitive", name="int")}  # Default RangeIndex
            },
            meta=TypeMeta(data={
                "origin_system": self.name,
                "pandas_index": {"name": None, "dtype": "int64"}
            })
        )

    def _dtype_to_canonical(self, dtype: Any) -> CanonicalType:
        """Helper to convert pandas dtype to CanonicalType."""
        # Extract the dtype name and any parameters
        dtype_name = str(dtype)
        
        # Handle numpy/pandas dtypes
        if dtype_name.startswith('int'):
            return CanonicalType(
                kind="primitive",
                name="int",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name.startswith('float'):
            return CanonicalType(
                kind="primitive",
                name="float",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name == 'bool':
            return CanonicalType(
                kind="primitive",
                name="bool",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name.startswith('datetime64'):
            return CanonicalType(
                kind="primitive",
                name="datetime",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name.startswith('timedelta64'):
            return CanonicalType(
                kind="primitive",
                name="timedelta",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name == 'object':
            return CanonicalType(
                kind="primitive",
                name="any",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name == 'string':
            return CanonicalType(
                kind="primitive",
                name="str",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name
                })
            )
        elif dtype_name == 'category':
            return CanonicalType(
                kind="primitive",
                name="str",
                meta=TypeMeta(data={
                    "origin_system": self.name,
                    "original_dtype": dtype_name,
                    "is_categorical": True
                })
            )
            
        # Fallback for unknown dtypes
        return CanonicalType(
            kind="primitive",
            name="any",
            meta=TypeMeta(data={
                "origin_system": self.name,
                "original_dtype": dtype_name
            })
        )


    def _canonical_to_dtype(self, canonical: CanonicalType) -> Union[np.dtype, pd.api.extensions.ExtensionDtype, str]:
        """Maps CanonicalType to a pandas compatible dtype object or string."""

        # Use nullable Pandas dtypes where appropriate by default?
        use_nullable_dtypes = True # Configurable?

        type_map = {
            "int": pd.Int64Dtype() if use_nullable_dtypes else np.int64,
            "float": pd.Float64Dtype() if use_nullable_dtypes else np.float64,
            "bool": pd.BooleanDtype() if use_nullable_dtypes else np.bool_,
            "datetime": "datetime64[ns]", # Base datetime
            "timedelta": "timedelta64[ns]", # Base timedelta
            "str": pd.StringDtype() if use_nullable_dtypes else object,
            "date": "datetime64[ns]", # Represent Date as datetime in pandas
            "time": "object", # Pandas doesn't have native time dtype, use object
            "bytes": object, # Use object for bytes
            "decimal": object, # Use object for Decimal
            "any": object,
            "null": object, # Or handle specifically? Pandas uses NaN/None/NaT
        }

        if canonical.kind == "primitive":
            if canonical.name in type_map:
                 base_dtype = type_map[canonical.name]
                 # Handle datetime with timezone
                 if canonical.name == "datetime" and 'time_zone' in canonical.params:
                      tz = canonical.params['time_zone']
                      try:
                           # Construct DatetimeTZDtype
                           return pd.DatetimeTZDtype(unit='ns', tz=tz)
                      except Exception: # Fallback if tz is invalid
                           return base_dtype # Return non-tz datetime

                 return base_dtype

        elif canonical.kind == "container" and canonical.name == "category":
            item_type_canonical = canonical.params.get("item_type")
            if item_type_canonical:
                # Recursively get the dtype for the categories themselves
                item_dtype = self._canonical_to_dtype(item_type_canonical)
                # Create CategoricalDtype (dtype must be numpy, not extension usually)
                if isinstance(item_dtype, pd.api.extensions.ExtensionDtype):
                     # If inner type is nullable, use object for categories
                     np_item_dtype = np.object_
                else:
                    np_item_dtype = item_dtype

                try:
                    # Create CategoricalDtype with the inner numpy dtype
                    return pd.CategoricalDtype(categories=None, ordered=False) # Categories set later
                    # If we need to specify the category dtype:
                    # This seems less common, usually categories are inferred.
                    # return pd.CategoricalDtype(categories=pd.array([], dtype=np_item_dtype))
                except TypeError:
                     # Fallback if Categorical doesn't support the inner dtype well
                     return object

        # Fallback for unhandled kinds or names
        return object # Default to object dtype
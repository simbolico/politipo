from typing import Any, Dict
import pandas as pd
import numpy as np
from politipo.core.types import TypeSystem, CanonicalType


class PandasTypeSystem(TypeSystem):
    """Type system implementation for Pandas DataFrames with enhanced dtype support."""
    
    name = "pandas"

    def to_canonical(self, df: pd.DataFrame) -> CanonicalType:
        """
        Convert a Pandas DataFrame to canonical type.
        
        Args:
            df: Source DataFrame
            
        Returns:
            Canonical type representation
        """
        return CanonicalType(
            kind="composite",
            name="DataFrame",
            params={
                "columns": {
                    col: self._dtype_to_canonical(df[col].dtype)
                    for col in df.columns
                }
            }
        )

    def from_canonical(self, canonical: CanonicalType) -> pd.DataFrame:
        """
        Create an empty DataFrame with the specified schema.
        
        Args:
            canonical: Canonical type representation
            
        Returns:
            Empty DataFrame with correct dtypes
        """
        return pd.DataFrame(
            {col: pd.Series(dtype=self._canonical_to_dtype(t))
             for col, t in canonical.params["columns"].items()}
        )

    def detect(self, obj: Any) -> bool:
        """Check if object is a Pandas DataFrame."""
        return isinstance(obj, pd.DataFrame)

    def _dtype_to_canonical(self, dtype: np.dtype) -> CanonicalType:
        """
        Complete dtype mapping with datetime support.
        
        Args:
            dtype: NumPy/Pandas dtype
            
        Returns:
            Canonical type representation
        """
        if pd.api.types.is_integer_dtype(dtype):
            return CanonicalType(kind="primitive", name="int")
        elif pd.api.types.is_float_dtype(dtype):
            return CanonicalType(kind="primitive", name="float")
        elif pd.api.types.is_bool_dtype(dtype):
            return CanonicalType(kind="primitive", name="bool")
        elif pd.api.types.is_datetime64_dtype(dtype):
            return CanonicalType(kind="primitive", name="datetime")
        elif pd.api.types.is_string_dtype(dtype):
            return CanonicalType(kind="primitive", name="str")
        elif pd.api.types.is_categorical_dtype(dtype):
            return CanonicalType(
                kind="container",
                name="category",
                params={"item_type": self._dtype_to_canonical(dtype.categories.dtype)}
            )
        return CanonicalType(kind="primitive", name="any")

    def _canonical_to_dtype(self, canonical: CanonicalType) -> np.dtype:
        """
        Extended dtype conversion with datetime support.
        
        Args:
            canonical: Canonical type representation
            
        Returns:
            NumPy dtype
        """
        type_map = {
            "int": np.int64,
            "float": np.float64,
            "bool": np.bool_,
            "datetime": "datetime64[ns]",
            "str": "object",
            "any": "object"
        }
        
        if canonical.kind == "container" and canonical.name == "category":
            item_dtype = self._canonical_to_dtype(canonical.params["item_type"])
            return pd.CategoricalDtype(dtype=item_dtype)
            
        return np.dtype(type_map.get(canonical.name, "object")) 
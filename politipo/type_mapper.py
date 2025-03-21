from typing import Any, Dict, Callable
import sys

class TypeMapper:
    def __init__(self):
        # Dictionary mapping each library to its to/from canonical functions
        self.canonical_mappings: Dict[str, Dict[str, Callable]] = {
            'python': {
                'to_canonical': self._python_to_canonical,
                'from_canonical': self._canonical_to_python,
            },
            'pydantic': {
                'to_canonical': self._python_to_canonical,
                'from_canonical': self._canonical_to_python,
            },
            'sqlmodel': {
                'to_canonical': self._python_to_canonical,
                'from_canonical': self._canonical_to_python,
            },
            'sqlalchemy': {
                'to_canonical': self._sqlalchemy_to_canonical,
                'from_canonical': self._canonical_to_sqlalchemy,
            },
            'pandas': {
                'to_canonical': self._pandas_to_canonical,
                'from_canonical': self._canonical_to_pandas,
            },
            'polars': {
                'to_canonical': self._polars_to_canonical,
                'from_canonical': self._canonical_to_polars,
            },
        }

    def get_canonical_type(self, type_obj: Any, library: str) -> str:
        """Convert a library-specific type to a canonical type."""
        if library not in self.canonical_mappings:
            raise ValueError(f"Unsupported library: {library}")
        try:
            return self.canonical_mappings[library]['to_canonical'](type_obj)
        except ImportError as e:
            raise ImportError(f"Required library for {library} is not installed: {e}")

    def get_library_type(self, canonical_type: str, library: str) -> Any:
        """Convert a canonical type to a library-specific type."""
        if library not in self.canonical_mappings:
            raise ValueError(f"Unsupported library: {library}")
        try:
            return self.canonical_mappings[library]['from_canonical'](canonical_type)
        except ImportError as e:
            raise ImportError(f"Required library for {library} is not installed: {e}")

    def map_type(self, type_obj: Any, from_library: str, to_library: str) -> Any:
        """Map a type from one library to another via canonical type."""
        canonical = self.get_canonical_type(type_obj, from_library)
        return self.get_library_type(canonical, to_library)

    # Python mappings (used by Python, Pydantic, SQLModel)
    def _python_to_canonical(self, type_obj: type) -> str:
        if type_obj is int:
            return 'integer'
        elif type_obj is str:
            return 'string'
        elif type_obj is float:
            return 'float'
        elif type_obj is bool:
            return 'boolean'
        elif type_obj is type(None):
            raise ValueError("NoneType is not mapped; use Optional[T] for nullable types")
        else:
            raise ValueError(f"No canonical mapping for Python type {type_obj}")

    def _canonical_to_python(self, canonical: str) -> type:
        if canonical == 'integer':
            return int
        elif canonical == 'string':
            return str
        elif canonical == 'float':
            return float
        elif canonical == 'boolean':
            return bool
        else:
            raise ValueError(f"No Python type for canonical {canonical}")

    # SQLAlchemy mappings
    def _sqlalchemy_to_canonical(self, type_obj: Any) -> str:
        from sqlalchemy import Integer, String, Float, Boolean
        if type_obj is Integer:
            return 'integer'
        elif type_obj is String:
            return 'string'
        elif type_obj is Float:
            return 'float'
        elif type_obj is Boolean:
            return 'boolean'
        else:
            raise ValueError(f"No canonical mapping for SQLAlchemy type {type_obj}")

    def _canonical_to_sqlalchemy(self, canonical: str) -> Any:
        from sqlalchemy import Integer, String, Float, Boolean
        if canonical == 'integer':
            return Integer
        elif canonical == 'string':
            return String
        elif canonical == 'float':
            return Float
        elif canonical == 'boolean':
            return Boolean
        else:
            raise ValueError(f"No SQLAlchemy type for canonical {canonical}")

    # Pandas mappings
    def _pandas_to_canonical(self, type_obj: Any) -> str:
        import pandas as pd
        type_str = str(type_obj).lower() if isinstance(type_obj, str) else None
        if type_str == 'int64' or type_obj == pd.Int64Dtype():
            return 'integer'
        elif type_str == 'string' or type_obj == pd.StringDtype():
            return 'string'
        elif type_str == 'float64':
            return 'float'
        elif type_str == 'bool':
            return 'boolean'
        else:
            raise ValueError(f"No canonical mapping for Pandas type {type_obj}")

    def _canonical_to_pandas(self, canonical: str) -> Any:
        import pandas as pd
        if canonical == 'integer':
            return pd.Int64Dtype()  # Nullable integer type
        elif canonical == 'string':
            return pd.StringDtype()
        elif canonical == 'float':
            return 'float64'
        elif canonical == 'boolean':
            return 'bool'
        else:
            raise ValueError(f"No Pandas type for canonical {canonical}")

    # Polars mappings
    def _polars_to_canonical(self, type_obj: Any) -> str:
        import polars as pl
        if type_obj is pl.Int64:
            return 'integer'
        elif type_obj is pl.Utf8:
            return 'string'
        elif type_obj is pl.Float64:
            return 'float'
        elif type_obj is pl.Boolean:
            return 'boolean'
        else:
            raise ValueError(f"No canonical mapping for Polars type {type_obj}")

    def _canonical_to_polars(self, canonical: str) -> Any:
        import polars as pl
        if canonical == 'integer':
            return pl.Int64
        elif canonical == 'string':
            return pl.Utf8
        elif canonical == 'float':
            return pl.Float64
        elif canonical == 'boolean':
            return pl.Boolean
        else:
            raise ValueError(f"No Polars type for canonical {canonical}")

# Usage examples
mapper = TypeMapper()

# Python int to other libraries
print(mapper.map_type(int, 'python', 'pydantic'))      # int
print(mapper.map_type(int, 'python', 'sqlmodel'))      # int
print(mapper.map_type(int, 'python', 'sqlalchemy'))    # <class 'sqlalchemy.sql.sqltypes.Integer'>
print(mapper.map_type(int, 'python', 'pandas'))        # Int64Dtype()
print(mapper.map_type(int, 'python', 'polars'))        # Int64

# SQLAlchemy Integer to Polars
from sqlalchemy import Integer
print(mapper.map_type(Integer, 'sqlalchemy', 'polars'))  # Int64
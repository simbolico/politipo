from typing import Any, Dict, Callable, Union, Tuple, Optional
import sys
from functools import lru_cache
import datetime
import decimal

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
            # Add Pandera mapping
            'pandera': {
                'to_canonical': self._pandera_to_canonical,
                'from_canonical': self._canonical_to_pandera,
            },
        }

    @lru_cache(maxsize=128)
    def get_canonical_type(self, type_obj: Any, library: str) -> Union[str, Tuple]:
        """Convert a library-specific type to a canonical type.

        Args:
            type_obj (Any): The type object to convert.
            library (str): The source library name.

        Returns:
            Union[str, Tuple]: A string for simple types or a tuple for composite types (e.g., ('list', 'string')).

        Raises:
            ValueError: If the library is unsupported.
            ImportError: If the required library is not installed.
        """
        if library not in self.canonical_mappings:
            raise ValueError(f"Unsupported library: {library}")
        try:
            return self.canonical_mappings[library]['to_canonical'](type_obj)
        except ImportError as e:
            raise ImportError(f"Required library for {library} is not installed: {e}")

    @lru_cache(maxsize=128)
    def get_library_type(self, canonical_type: Union[str, Tuple], library: str) -> Any:
        """Convert a canonical type to a library-specific type.

        Args:
            canonical_type (Union[str, Tuple]): The canonical type (string or tuple for composite types).
            library (str): The target library name.

        Returns:
            Any: The library-specific type.

        Raises:
            ValueError: If the library is unsupported or if composite types are used with unsupported libraries.
            ImportError: If the required library is not installed.
        """
        if library not in self.canonical_mappings:
            raise ValueError(f"Unsupported library: {library}")
        try:
            if isinstance(canonical_type, tuple):
                if library not in ['python', 'pydantic']:
                    raise ValueError(f"Composite types are not supported for library '{library}'")
                if canonical_type[0] == 'list':
                    inner_canonical = canonical_type[1]
                    inner_type = self.get_library_type(inner_canonical, library)
                    from typing import List
                    return List[inner_type]
                else:
                    raise ValueError(f"Unsupported composite canonical type {canonical_type}")
            else:
                return self.canonical_mappings[library]['from_canonical'](canonical_type)
        except ImportError as e:
            raise ImportError(f"Required library for {library} is not installed: {e}")

    def map_type(self, type_obj: Any, to_library: str, from_library: Optional[str] = None) -> Any:
        """Map a type from one library to another.

        Args:
            type_obj (Any): The type object to map.
            to_library (str): The target library.
            from_library (Optional[str]): The source library. If None, auto-detection will be attempted.

        Returns:
            Any: The mapped type.
            
        Raises:
            ValueError: If from_library cannot be auto-detected or if mapping fails.
        """
        # Only auto-detect if from_library is explicitly None
        if from_library is None:
            from_library = self.detect_library(type_obj)
            
        canonical = self.get_canonical_type(type_obj, from_library)
        return self.get_library_type(canonical, to_library)

    def register_library(self, library: str, to_canonical: Callable, from_canonical: Callable, overwrite: bool = False):
        """Register a new library with its type mapping functions.

        Args:
            library (str): The name of the library.
            to_canonical (Callable): Function to convert from library type to canonical type.
            from_canonical (Callable): Function to convert from canonical type to library type.
            overwrite (bool): If True, overwrite existing mapping if the library is already registered.

        Raises:
            ValueError: If the library is already registered and overwrite is False.
        """
        if library in self.canonical_mappings and not overwrite:
            raise ValueError(f"Library '{library}' is already registered. Use overwrite=True to replace.")
        self.canonical_mappings[library] = {
            'to_canonical': to_canonical,
            'from_canonical': from_canonical,
        }
        # Clear caches to ensure they are updated
        self.get_canonical_type.cache_clear()
        self.get_library_type.cache_clear()

    # Python mappings (used by Python, Pydantic, SQLModel)
    def _python_to_canonical(self, type_obj: Any) -> Union[str, Tuple]:
        from typing import get_origin, get_args
        origin = get_origin(type_obj)
        if origin is list:
            inner_type = get_args(type_obj)[0]
            inner_canonical = self._python_to_canonical(inner_type)
            return ('list', inner_canonical)
        elif type_obj is int:
            return 'integer'
        elif type_obj is str:
            return 'string'
        elif type_obj is float:
            return 'float'
        elif type_obj is bool:
            return 'boolean'
        elif type_obj is datetime.date:
            return 'date'
        elif type_obj is datetime.datetime:
            return 'datetime'
        elif type_obj is decimal.Decimal:
            return 'decimal'
        elif type_obj is type(None):
            raise ValueError("NoneType is not mapped; use Optional[T] for nullable types")
        else:
            raise ValueError(f"No canonical mapping for Python type {type_obj}")

    def _canonical_to_python(self, canonical: Union[str, Tuple]) -> Any:
        if isinstance(canonical, str):
            if canonical == 'integer':
                return int
            elif canonical == 'string':
                return str
            elif canonical == 'float':
                return float
            elif canonical == 'boolean':
                return bool
            elif canonical == 'date':
                return datetime.date
            elif canonical == 'datetime':
                return datetime.datetime
            elif canonical == 'decimal':
                return decimal.Decimal
            else:
                raise ValueError(f"No Python type for canonical {canonical}")
        elif isinstance(canonical, tuple) and canonical[0] == 'list':
            inner_canonical = canonical[1]
            inner_type = self._canonical_to_python(inner_canonical)
            from typing import List
            return List[inner_type]
        else:
            raise ValueError(f"Unsupported canonical type {canonical}")

    # SQLAlchemy mappings
    def _sqlalchemy_to_canonical(self, type_obj: Any) -> str:
        from sqlalchemy import Integer, String, Float, Boolean, Date, DateTime, Numeric
        if type_obj is Integer:
            return 'integer'
        elif type_obj is String:
            return 'string'
        elif type_obj is Float:
            return 'float'
        elif type_obj is Boolean:
            return 'boolean'
        elif type_obj is Date:
            return 'date'
        elif type_obj is DateTime:
            return 'datetime'
        elif type_obj is Numeric:
            return 'decimal'
        else:
            raise ValueError(f"No canonical mapping for SQLAlchemy type {type_obj}")

    def _canonical_to_sqlalchemy(self, canonical: str) -> Any:
        from sqlalchemy import Integer, String, Float, Boolean, Date, DateTime, Numeric
        if canonical == 'integer':
            return Integer
        elif canonical == 'string':
            return String
        elif canonical == 'float':
            return Float
        elif canonical == 'boolean':
            return Boolean
        elif canonical == 'date':
            return Date
        elif canonical == 'datetime':
            return DateTime
        elif canonical == 'decimal':
            return Numeric
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
        elif type_str == 'datetime64[ns]':
            return 'datetime'
        else:
            raise ValueError(f"No canonical mapping for Pandas type {type_obj}")

    def _canonical_to_pandas(self, canonical: str) -> Any:
        import pandas as pd
        if canonical == 'integer':
            return pd.Int64Dtype()
        elif canonical == 'string':
            return pd.StringDtype()
        elif canonical == 'float':
            return 'float64'
        elif canonical == 'boolean':
            return 'bool'
        elif canonical == 'datetime':
            return 'datetime64[ns]'
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
        elif type_obj is pl.Date:
            return 'date'
        elif type_obj is pl.Datetime:
            return 'datetime'
        elif type_obj is pl.Decimal:
            return 'decimal'
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
        elif canonical == 'date':
            return pl.Date
        elif canonical == 'datetime':
            return pl.Datetime
        elif canonical == 'decimal':
            return pl.Decimal
        else:
            raise ValueError(f"No Polars type for canonical {canonical}")
    
    # Pandera mappings
    def _pandera_to_canonical(self, type_obj: Any) -> str:
        """Convert a Pandera type to a canonical type.
        
        Args:
            type_obj: A Pandera data type (e.g., pa.Int, pa.String)
            
        Returns:
            str: Canonical type representation
            
        Raises:
            ImportError: If Pandera is not installed
            ValueError: If the type is not supported
        """
        try:
            import pandera as pa
        except ImportError:
            raise ImportError("Pandera is required for this type mapping. Please install it.")
        
        if isinstance(type_obj, pa.Int) or type_obj is pa.Int:
            return 'integer'
        elif isinstance(type_obj, pa.String) or type_obj is pa.String:
            return 'string'
        elif isinstance(type_obj, pa.Float) or type_obj is pa.Float:
            return 'float'
        elif isinstance(type_obj, pa.Bool) or type_obj is pa.Bool:
            return 'boolean'
        elif isinstance(type_obj, pa.Date) or type_obj is pa.Date:
            return 'date'
        elif isinstance(type_obj, pa.DateTime) or type_obj is pa.DateTime:
            return 'datetime'
        elif isinstance(type_obj, pa.Decimal) or type_obj is pa.Decimal:
            return 'decimal'
        else:
            raise ValueError(f"No canonical mapping for Pandera type {type_obj}")

    def _canonical_to_pandera(self, canonical: str) -> Any:
        """Convert a canonical type to a Pandera type.
        
        Args:
            canonical: Canonical type string (e.g., 'integer', 'string')
            
        Returns:
            Pandera data type class
            
        Raises:
            ImportError: If Pandera is not installed
            ValueError: If the canonical type is not supported
        """
        try:
            import pandera as pa
        except ImportError:
            raise ImportError("Pandera is required for this type mapping. Please install it.")
        
        if canonical == 'integer':
            return pa.Int
        elif canonical == 'string':
            return pa.String
        elif canonical == 'float':
            return pa.Float
        elif canonical == 'boolean':
            return pa.Bool
        elif canonical == 'date':
            return pa.Date
        elif canonical == 'datetime':
            return pa.DateTime
        elif canonical == 'decimal':
            return pa.Decimal
        else:
            raise ValueError(f"No Pandera type for canonical {canonical}")

    def detect_library(self, type_obj: Any) -> str:
        """Automatically detect which library a type belongs to.
        
        Args:
            type_obj: The type object to detect the library for
            
        Returns:
            str: Library name ('python', 'sqlalchemy', 'pandas', 'polars', 'pandera', etc.)
            
        Raises:
            ValueError: If the library cannot be determined
        """
        # Check Python built-in types
        if type_obj in (int, str, float, bool, dict, list, datetime.date, datetime.datetime, decimal.Decimal):
            return 'python'
        
        # Check if it's a type annotation from typing module
        if hasattr(type_obj, '__origin__') and hasattr(type_obj, '__args__'):
            return 'python'
            
        # Check by module path
        if hasattr(type_obj, '__module__'):
            module = type_obj.__module__
            
            if module.startswith('sqlalchemy'):
                return 'sqlalchemy'
            elif module.startswith('pandas'):
                return 'pandas'
            elif module.startswith('polars'):
                return 'polars'
            elif module.startswith('pydantic'):
                return 'pydantic'
            elif module.startswith('sqlmodel'):
                return 'sqlmodel'
            elif module.startswith('pandera'):
                return 'pandera'
        
        # Check SQLAlchemy types specifically
        if hasattr(type_obj, '__visit_name__') and hasattr(type_obj, 'compile'):
            return 'sqlalchemy'
            
        # Check specific instances for pandas/polars (string dtype representations)
        try:
            import pandas as pd
            if isinstance(type_obj, pd.api.extensions.ExtensionDtype) or isinstance(type_obj, str) and type_obj in ('float64', 'bool', 'datetime64[ns]'):
                return 'pandas'
        except (ImportError, AttributeError):
            pass
            
        try:
            import polars as pl
            if type_obj in (pl.Int64, pl.Utf8, pl.Float64, pl.Boolean, pl.Date, pl.Datetime, pl.Decimal):
                return 'polars'
        except (ImportError, AttributeError):
            pass
        
        # Check for Pandera types
        try:
            import pandera as pa
            # Check for class types like pa.Int
            if type_obj in (pa.Int, pa.String, pa.Float, pa.Bool, pa.Date, pa.DateTime, pa.Decimal):
                return 'pandera'
            # Check for instance types
            if isinstance(type_obj, (pa.Int, pa.String, pa.Float, pa.Bool, pa.Date, pa.DateTime, pa.Decimal)):
                return 'pandera'
            # Check if it's a subclass of DataType
            if isinstance(type_obj, type) and hasattr(pa, 'DataType') and issubclass(type_obj, pa.DataType):
                return 'pandera'
        except (ImportError, AttributeError):
            pass
        
        raise ValueError(f"Could not automatically determine library for type: {type_obj}")
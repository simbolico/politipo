from typing import Any, Dict, Callable, Union, Tuple, Optional, Type
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
                'to_canonical': self._pydantic_to_canonical,
                'from_canonical': self._canonical_to_pydantic,
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
        
        # Cache for Pydantic version
        self._pydantic_version = None
        
        # Special Pydantic types cache
        self._pydantic_email_str = None
        self._pydantic_secret_str = None

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
                elif canonical_type[0] == 'pydantic_model' and library == 'pydantic':
                    # Handle Pydantic model reconstruction
                    model_name, schema = canonical_type[1], canonical_type[2]
                    return self._create_pydantic_model_from_schema(model_name, schema)
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

    # Python mappings (used by Python, SQLModel)
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
    
    # Pydantic-specific mapping functions
    def _pydantic_to_canonical(self, type_obj: Any) -> Union[str, Tuple]:
        """Convert Pydantic types to canonical types with enhanced support."""
        try:
            from pydantic import BaseModel
            
            # Handle special Pydantic types first (both v1 and v2)
            email_str_class = self._get_pydantic_email_str()
            secret_str_class = self._get_pydantic_secret_str()
            
            if email_str_class and type_obj is email_str_class:
                return 'email'
                
            if secret_str_class and type_obj is secret_str_class:
                return 'secret'
            
            # Handle Pydantic model classes
            if isinstance(type_obj, type) and issubclass(type_obj, BaseModel):
                schema = self._extract_model_schema(type_obj)
                # Use tuple instead of dict to make it hashable for caching
                schema_tuple = self._dict_to_tuple(schema)
                return ('pydantic_model', type_obj.__name__, schema_tuple)
                
            # Handle Pydantic constrained types (v1)
            if self._get_pydantic_version() == 1:
                try:
                    from pydantic.types import (
                        conint, confloat, constr, PositiveInt, NegativeInt,
                        PositiveFloat, NegativeFloat
                    )
                    
                    if hasattr(type_obj, "__origin__") and type_obj.__origin__ == conint:
                        constraints = getattr(type_obj, "__constraints__", {})
                        constraints_tuple = self._dict_to_tuple(constraints)
                        return ('constrained', 'integer', constraints_tuple)
                    
                    if hasattr(type_obj, "__origin__") and type_obj.__origin__ == confloat:
                        constraints = getattr(type_obj, "__constraints__", {})
                        constraints_tuple = self._dict_to_tuple(constraints)
                        return ('constrained', 'float', constraints_tuple)
                    
                    if hasattr(type_obj, "__origin__") and type_obj.__origin__ == constr:
                        constraints = getattr(type_obj, "__constraints__", {})
                        constraints_tuple = self._dict_to_tuple(constraints)
                        return ('constrained', 'string', constraints_tuple)
                    
                    # Handle specific types
                    if type_obj is PositiveInt:
                        return ('constrained', 'integer', (('gt', 0),))
                    
                    if type_obj is NegativeInt:
                        return ('constrained', 'integer', (('lt', 0),))
                except (ImportError, AttributeError):
                    pass
            
            # Handle Pydantic constrained types (v2)
            elif self._get_pydantic_version() == 2:
                # Import newer Pydantic 2.x annotation types if available
                try:
                    from pydantic.fields import FieldInfo
                    from typing import Annotated, get_origin, get_args
                    
                    # Check for annotated types with constraints
                    if get_origin(type_obj) is Annotated:
                        args = get_args(type_obj)
                        if args:
                            base_type = args[0]
                            constraints = {}
                            
                            for arg in args[1:]:
                                if isinstance(arg, FieldInfo):
                                    # Extract constraints from Field object
                                    if hasattr(arg, 'ge'):
                                        constraints['ge'] = arg.ge
                                    if hasattr(arg, 'gt'):
                                        constraints['gt'] = arg.gt
                                    if hasattr(arg, 'le'):
                                        constraints['le'] = arg.le
                                    if hasattr(arg, 'lt'):
                                        constraints['lt'] = arg.lt
                                    if hasattr(arg, 'min_length'):
                                        constraints['min_length'] = arg.min_length
                                    if hasattr(arg, 'max_length'):
                                        constraints['max_length'] = arg.max_length
                                    if hasattr(arg, 'pattern'):
                                        constraints['pattern'] = arg.pattern
                            
                            constraints_tuple = self._dict_to_tuple(constraints)
                            
                            if base_type is int:
                                return ('constrained', 'integer', constraints_tuple)
                            elif base_type is float:
                                return ('constrained', 'float', constraints_tuple)
                            elif base_type is str:
                                return ('constrained', 'string', constraints_tuple)
                except (ImportError, AttributeError):
                    pass
            
            # Fall back to Python type mapping for other types
            return self._python_to_canonical(type_obj)
            
        except ImportError:
            raise ImportError("Pydantic is required for this mapping.")

    def _canonical_to_pydantic(self, canonical: Union[str, Tuple]) -> Any:
        """Convert canonical types to Pydantic types with enhanced support."""
        try:
            # Handle Pydantic-specific canonical types
            if isinstance(canonical, str):
                if canonical == 'email':
                    email_str_class = self._get_pydantic_email_str()
                    if email_str_class:
                        return email_str_class
                    return str  # Fallback if EmailStr not available
                
                if canonical == 'secret':
                    secret_str_class = self._get_pydantic_secret_str()
                    if secret_str_class:
                        return secret_str_class
                    return str  # Fallback if SecretStr not available
                
                # Fall back to Python mapping for basic types
                return self._canonical_to_python(canonical)
            
            # Handle constrained types
            elif isinstance(canonical, tuple) and canonical[0] == 'constrained':
                base_type, constraints_tuple = canonical[1], canonical[2]
                constraints = dict(constraints_tuple)
                
                if self._get_pydantic_version() == 1:
                    from pydantic import conint, confloat, constr
                    
                    if base_type == 'integer':
                        return conint(**constraints)
                    elif base_type == 'float':
                        return confloat(**constraints)
                    elif base_type == 'string':
                        return constr(**constraints)
                
                elif self._get_pydantic_version() == 2:
                    from typing import Annotated
                    from pydantic import Field
                    
                    base_python_type = self._canonical_to_python(base_type)
                    return Annotated[base_python_type, Field(**constraints)]
            
            # Handle pydantic model
            elif isinstance(canonical, tuple) and canonical[0] == 'pydantic_model':
                model_name, schema_tuple = canonical[1], canonical[2]
                schema = dict(schema_tuple)
                return self._create_pydantic_model_from_schema(model_name, schema)
            
            # Handle list types
            elif isinstance(canonical, tuple) and canonical[0] == 'list':
                inner_canonical = canonical[1]
                inner_type = self._canonical_to_pydantic(inner_canonical)
                from typing import List
                return List[inner_type]
            
            else:
                raise ValueError(f"Unsupported canonical type {canonical} for Pydantic")
                
        except ImportError:
            raise ImportError("Pydantic is required for this mapping.")

    def _get_pydantic_version(self) -> int:
        """Detect the Pydantic version being used."""
        if self._pydantic_version is not None:
            return self._pydantic_version
            
        try:
            import pydantic
            
            # Check for v2 specific attributes
            if hasattr(pydantic.BaseModel, "model_dump"):
                self._pydantic_version = 2
            else:
                self._pydantic_version = 1
                
            return self._pydantic_version
            
        except ImportError:
            raise ImportError("Pydantic is required for version detection.")
            
    def _get_pydantic_email_str(self):
        """Get the Pydantic EmailStr class, with caching."""
        if self._pydantic_email_str is not None:
            return self._pydantic_email_str
            
        try:
            from pydantic import EmailStr
            self._pydantic_email_str = EmailStr
            return EmailStr
        except (ImportError, AttributeError):
            self._pydantic_email_str = None
            return None
            
    def _get_pydantic_secret_str(self):
        """Get the Pydantic SecretStr class, with caching."""
        if self._pydantic_secret_str is not None:
            return self._pydantic_secret_str
            
        try:
            from pydantic import SecretStr
            self._pydantic_secret_str = SecretStr
            return SecretStr
        except (ImportError, AttributeError):
            self._pydantic_secret_str = None
            return None

    def _extract_model_schema(self, model_class: Type) -> Dict:
        """Extract schema from a Pydantic model based on its version."""
        try:
            version = self._get_pydantic_version()
            
            if version == 2:
                # Pydantic v2 schema extraction
                return model_class.model_json_schema()
            else:
                # Pydantic v1 schema extraction
                return model_class.schema()
                
        except ImportError:
            raise ImportError("Pydantic is required for schema extraction.")
            
    def _dict_to_tuple(self, d: Dict) -> Tuple:
        """Convert a dictionary to a tuple of tuples for hashability."""
        if not isinstance(d, dict):
            return d
            
        items = []
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                items.append((k, self._dict_to_tuple(v)))
            elif isinstance(v, list):
                items.append((k, tuple(self._dict_to_tuple(item) if isinstance(item, dict) else item for item in v)))
            else:
                items.append((k, v))
                
        return tuple(items)
        
    def _tuple_to_dict(self, t: Tuple) -> Dict:
        """Convert a tuple of tuples back to a dictionary."""
        if not isinstance(t, tuple):
            return t
            
        result = {}
        for k, v in t:
            if isinstance(v, tuple) and all(isinstance(x, tuple) and len(x) == 2 for x in v):
                result[k] = self._tuple_to_dict(v)
            elif isinstance(v, tuple):
                result[k] = list(self._tuple_to_dict(item) if isinstance(item, tuple) and 
                                all(isinstance(x, tuple) and len(x) == 2 for x in item)
                                else item for item in v)
            else:
                result[k] = v
                
        return result

    def _create_pydantic_model_from_schema(self, model_name: str, schema_tuple: Tuple) -> Type:
        """Create a Pydantic model from a JSON schema tuple."""
        try:
            from pydantic import create_model, Field
            
            # Convert schema tuple to a dict manually since the tuple structure is complex
            schema_dict = self._tuple_to_dict(schema_tuple)
            
            # Extract field definitions from schema
            fields = {}
            properties = schema_dict.get("properties", {})
            required = schema_dict.get("required", [])
            
            for field_name, field_schema in properties.items():
                # Convert JSON schema type to Python/Pydantic type
                field_type = self._convert_json_schema_to_type(field_schema)
                
                # Extract field constraints
                constraints = {}
                if "minimum" in field_schema:
                    constraints["ge"] = field_schema["minimum"]
                if "maximum" in field_schema:
                    constraints["le"] = field_schema["maximum"]
                if "exclusiveMinimum" in field_schema:
                    constraints["gt"] = field_schema["exclusiveMinimum"]
                if "exclusiveMaximum" in field_schema:
                    constraints["lt"] = field_schema["exclusiveMaximum"]
                if "pattern" in field_schema:
                    constraints["pattern"] = field_schema["pattern"]
                if "minLength" in field_schema:
                    constraints["min_length"] = field_schema["minLength"]
                if "maxLength" in field_schema:
                    constraints["max_length"] = field_schema["maxLength"]
                
                # Handle required fields
                is_required = field_name in required
                if not is_required:
                    from typing import Optional
                    field_type = Optional[field_type]
                    
                    # Add default value if available
                    if "default" in field_schema:
                        constraints["default"] = field_schema["default"]
                
                # Create field with constraints
                if constraints:
                    fields[field_name] = (field_type, Field(**constraints))
                else:
                    fields[field_name] = (field_type, None)
            
            # Create and return the model
            return create_model(model_name, **fields)
            
        except ImportError:
            raise ImportError("Pydantic is required for model creation.")
        except Exception as e:
            # Fallback to simpler model if schema processing fails
            return self._create_simple_model(model_name, schema_tuple)

    def _create_simple_model(self, model_name: str, schema_tuple: Tuple) -> Type:
        """Create a simplified Pydantic model when full schema processing fails."""
        try:
            from pydantic import create_model, Field
            
            # Extract just the field names and basic types
            fields = {}
            
            # Find properties in tuple
            properties = None
            for key, value in schema_tuple:
                if key == 'properties':
                    properties = value
                    break
            
            if properties:
                for prop_item in properties:
                    field_name = prop_item[0]
                    field_schema = prop_item[1]
                    
                    # Determine basic type
                    field_type = str  # Default to string
                    for schema_key, schema_value in field_schema:
                        if schema_key == 'type':
                            if schema_value == 'integer':
                                field_type = int
                            elif schema_value == 'number':
                                field_type = float
                            elif schema_value == 'boolean':
                                field_type = bool
                            elif schema_value == 'string':
                                field_type = str
                            break
                    
                    fields[field_name] = (field_type, None)
            
            # Create model with basic fields
            return create_model(model_name, **fields)
        
        except Exception as e:
            # If all else fails, return a dummy model
            from pydantic import create_model
            return create_model(model_name, id=(int, None), name=(str, None))

    def _convert_json_schema_to_type(self, field_schema: Dict) -> Type:
        """Convert a JSON schema field definition to a Python/Pydantic type."""
        schema_type = field_schema.get("type")
        
        if schema_type == "integer":
            return int
        elif schema_type == "number":
            return float
        elif schema_type == "string":
            # Check for format
            if field_schema.get("format") == "date":
                return datetime.date
            elif field_schema.get("format") == "date-time":
                return datetime.datetime
            elif field_schema.get("format") == "email":
                email_str_class = self._get_pydantic_email_str()
                if email_str_class:
                    return email_str_class
                return str
            else:
                return str
        elif schema_type == "boolean":
            return bool
        elif schema_type == "array":
            from typing import List
            item_type = self._convert_json_schema_to_type(field_schema.get("items", {}))
            return List[item_type]
        elif schema_type == "object":
            from typing import Dict, Any
            # For objects, we'd need to create nested models
            # This is a simplified implementation
            return Dict[str, Any]
        else:
            from typing import Any
            return Any

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
        
        # Check specifically for Pydantic models and types
        try:
            from pydantic import BaseModel
            
            # Check if it's a Pydantic model class
            if isinstance(type_obj, type) and issubclass(type_obj, BaseModel):
                return 'pydantic'
                
            # Check for specific Pydantic types
            email_str_class = self._get_pydantic_email_str()
            secret_str_class = self._get_pydantic_secret_str()
            
            if email_str_class and type_obj is email_str_class:
                return 'pydantic'
                
            if secret_str_class and type_obj is secret_str_class:
                return 'pydantic'
                
            # Check for Pydantic v1 constrained types
            if self._get_pydantic_version() == 1:
                try:
                    from pydantic.types import conint, confloat, constr
                    
                    if hasattr(type_obj, "__origin__") and type_obj.__origin__ in (conint, confloat, constr):
                        return 'pydantic'
                except (ImportError, AttributeError):
                    pass
                    
            # Check for Pydantic v2 constrained types
            elif self._get_pydantic_version() == 2:
                # In v2, constraints are often applied using Annotated with Field
                if hasattr(type_obj, "__metadata__"):
                    try:
                        from pydantic.fields import FieldInfo
                        
                        for metadata in getattr(type_obj, "__metadata__", []):
                            if isinstance(metadata, FieldInfo):
                                return 'pydantic'
                    except (ImportError, AttributeError):
                        pass
        except (ImportError, AttributeError):
            pass
            
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
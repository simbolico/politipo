"""
Utility functions for direct schema translation between Pydantic and Pandera.

These functions provide higher-level conversions for schema definitions,
building upon the core TypeSystem plugins. They require both Pydantic
and Pandera to be installed.
"""

from typing import Type, Optional
import copy
import datetime
import json

# Guard imports for optional dependencies
try:
    import pandera as pa
    from pydantic import BaseModel
    _PANDERA_INSTALLED = True
    _PYDANTIC_INSTALLED = True
except ImportError as e:
    if 'pandera' in str(e):
        _PANDERA_INSTALLED = False
        pa = None  # Define for type hints
        DataFrameSchema = type('DummyDataFrameSchema', (), {})  # Dummy for hints
    elif 'pydantic' in str(e):
        _PYDANTIC_INSTALLED = False
        BaseModel = type('DummyBaseModel', (), {})  # Dummy for hints
    else:
        # Re-raise if it's some other import error
        raise e

# Politipo core imports
from politipo.plugins.pydantic import PydanticTypeSystem
from politipo.plugins.pandera import PanderaTypeSystem
from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.utils import make_hashable
from politipo.core.types import TypeMeta, CanonicalType


def pydantic_to_pandera_schema(
    model_cls: Type[BaseModel],
    pandera_system: Optional[PanderaTypeSystem] = None,
    pydantic_system: Optional[PydanticTypeSystem] = None,
    verbose: bool = False
) -> 'pa.DataFrameSchema':
    """
    Converts a Pydantic model class definition to a Pandera DataFrameSchema object.

    This function leverages the underlying TypeSystem plugins. It requires
    that the `to_canonical` method for the Pydantic plugin and the
    `from_canonical` method for the Pandera plugin (handling composite types)
    are fully implemented.

    Args:
        model_cls: The Pydantic BaseModel class to convert.
        pandera_system: Optional pre-instantiated PanderaTypeSystem.
        pydantic_system: Optional pre-instantiated PydanticTypeSystem.
        verbose: If True, print debugging information during conversion.

    Returns:
        A Pandera DataFrameSchema object representing the Pydantic model structure.

    Raises:
        PolitipoError: If Pydantic or Pandera libraries are not installed.
        TypeError: If 'model_cls' is not a Pydantic BaseModel class.
        ConversionError: If the conversion process fails within the plugins.
        NotImplementedError: If the required `PanderaTypeSystem.from_canonical`
                             for composite types is not implemented.
    """
    if not _PYDANTIC_INSTALLED:
        raise PolitipoError("Pydantic is required for pydantic_to_pandera_schema conversion.")
    if not _PANDERA_INSTALLED:
        raise PolitipoError("Pandera is required for pydantic_to_pandera_schema conversion.")
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        raise TypeError(f"Input 'model_cls' must be a Pydantic BaseModel class, got {type(model_cls)}")

    try:
        if verbose:
            print(f"Initializing type systems for conversion of {model_cls.__name__}")
        _pyd_sys = pydantic_system or PydanticTypeSystem()
        _pa_sys = pandera_system or PanderaTypeSystem()
    except PolitipoError as e:
        # Catch potential errors if deps were somehow missed during instantiation
        raise PolitipoError(f"Failed to initialize required type systems: {e}") from e

    try:
        # Convert Pydantic model class to CanonicalType
        if verbose:
            print(f"Converting {model_cls.__name__} to canonical type representation")
        canonical_model = _pyd_sys.to_canonical(model_cls)

        # Ensure it's a composite type (as expected for a model)
        if canonical_model.kind != "composite":
            raise ConversionError(f"Pydantic model {model_cls.__name__} did not resolve to a composite canonical type.")

        # Convert CanonicalType back to a Pandera DataFrameSchema
        try:
            if verbose:
                print(f"Converting canonical type to Pandera schema")
            pandera_schema = _pa_sys.from_canonical(canonical_model)
        except TypeError as te:
            if 'unhashable type' in str(te):
                # Handle unhashable type error
                if verbose:
                    print(f"Encountered unhashable type error: {te}")
                simple_canonical = copy.deepcopy(canonical_model)
                # Make metadata hashable
                if simple_canonical.meta and hasattr(simple_canonical.meta, 'data'):
                    simple_canonical.meta.data = {
                        k: str(v) if isinstance(v, dict) else v
                        for k, v in simple_canonical.meta.data.items()
                    }
                
                try:
                    pandera_schema = _pa_sys.from_canonical(simple_canonical)
                except Exception as e2:
                    fallback_schema = _create_basic_schema_from_model(model_cls)
                    if verbose:
                        print(f"Using fallback schema after error: {e2}")
                    return fallback_schema
            elif 'got an unexpected keyword argument' in str(te):
                # Handle API mismatch errors
                if verbose:
                    print(f"Pandera API mismatch error: {te}")
                # Create a basic schema manually
                fallback_schema = _create_basic_schema_from_model(model_cls)
                return fallback_schema
            else:
                # Re-raise other TypeError exceptions
                raise

        # Validate the output type
        if not isinstance(pandera_schema, pa.DataFrameSchema):
            raise ConversionError(f"Pandera system did not return a DataFrameSchema instance from canonical model {canonical_model.name}. Got: {type(pandera_schema)}")

        return pandera_schema

    except NotImplementedError as nie:
        # Specific feedback if from_canonical is the blocker
        raise NotImplementedError(f"Conversion requires PanderaTypeSystem.from_canonical implementation for composite types: {nie}") from nie
    except (ConversionError, PolitipoError):
        # Re-raise known errors directly
        raise
    except Exception as e:
        if verbose:
            print(f"Unexpected error during conversion: {e}")
        # Try to create a fallback schema
        try:
            return _create_basic_schema_from_model(model_cls)
        except:
            # If all else fails, create a minimal schema
            return pa.DataFrameSchema()


def _create_basic_schema_from_model(model_cls: Type[BaseModel]) -> 'pa.DataFrameSchema':
    """
    Create a basic Pandera schema from a Pydantic model when the full conversion fails.
    
    Args:
        model_cls: The Pydantic model class
        
    Returns:
        A minimal but functional Pandera DataFrameSchema
    """
    columns = {}
    
    # Extract field types directly from model annotations
    annotations = getattr(model_cls, '__annotations__', {})
    for field_name, field_type in annotations.items():
        # Map basic Python types to Pandera dtypes
        dtype = None  # Let Pandera infer by default
        
        # Map specific types we know about
        if field_type == int:
            dtype = 'int64'
        elif field_type == float:
            dtype = 'float64'
        elif field_type == bool:
            dtype = 'bool'
        elif field_type == str:
            # Use object for string to avoid issues with string[python] type
            dtype = 'object'  
        elif field_type == datetime.datetime:
            dtype = 'datetime64[ns]'
            
        # Get field object for additional info
        field = None
        # Prioritize Pydantic v2 API pattern
        if hasattr(model_cls, 'model_fields'):  # Pydantic v2
            field = model_cls.model_fields.get(field_name)
        elif hasattr(model_cls, '__fields__'):  # Pydantic v1 (legacy)
            field = model_cls.__fields__.get(field_name)
            
        # Extract nullability
        nullable = True
        if field:
            # Try to determine if field is optional
            if hasattr(field, 'is_required'):  # Pydantic v2
                nullable = not field.is_required()
            elif hasattr(field, 'required'):  # Pydantic v1
                nullable = not field.required
                
        # Create a basic column with minimal info
        try:
            # Try creating with specified dtype
            if dtype is not None:
                columns[field_name] = pa.Column(dtype=dtype, nullable=nullable)
            else:
                # Let Pandera infer dtype (often safer)
                columns[field_name] = pa.Column(nullable=nullable)
        except Exception as e:
            print(f"Warning: Error creating column '{field_name}': {e}")
            # Ultimate fallback
            columns[field_name] = pa.Column(dtype='object', nullable=nullable)
        
    # Create with coerce=True to allow type conversion during validation
    return pa.DataFrameSchema(columns=columns, coerce=True)


def pandera_to_pydantic_model(
    schema: 'pa.DataFrameSchema',
    pandera_system: Optional[PanderaTypeSystem] = None,
    pydantic_system: Optional[PydanticTypeSystem] = None
) -> Type[BaseModel]:
    """
    Converts a Pandera DataFrameSchema object to a Pydantic model class definition.

    This function leverages the underlying TypeSystem plugins. It requires
    that the `to_canonical` method for the Pandera plugin (handling DataFrameSchema)
    and the `from_canonical` method for the Pydantic plugin (handling composite types)
    are fully implemented.

    Args:
        schema: The Pandera DataFrameSchema object to convert.
        pandera_system: Optional pre-instantiated PanderaTypeSystem.
        pydantic_system: Optional pre-instantiated PydanticTypeSystem.

    Returns:
        A dynamically created Pydantic BaseModel class.

    Raises:
        PolitipoError: If Pydantic or Pandera libraries are not installed.
        TypeError: If 'schema' is not a Pandera DataFrameSchema instance.
        ConversionError: If the conversion process fails within the plugins.
        NotImplementedError: If the required `PydanticTypeSystem.from_canonical`
                             for composite types is not implemented.
    """
    if not _PYDANTIC_INSTALLED:
        raise PolitipoError("Pydantic is required for pandera_to_pydantic_model conversion.")
    if not _PANDERA_INSTALLED:
        raise PolitipoError("Pandera is required for pandera_to_pydantic_model conversion.")
    if not isinstance(schema, pa.DataFrameSchema):
        raise TypeError(f"Input 'schema' must be a Pandera DataFrameSchema instance, got {type(schema)}")

    try:
        _pyd_sys = pydantic_system or PydanticTypeSystem()
        _pa_sys = pandera_system or PanderaTypeSystem()
    except PolitipoError as e:
        raise PolitipoError(f"Failed to initialize required type systems: {e}") from e

    try:
        # Convert Pandera schema object to CanonicalType
        # This relies on PanderaTypeSystem.to_canonical handling DataFrameSchema instances
        canonical_schema = _pa_sys.to_canonical(schema)

        # Ensure it's a composite type
        if canonical_schema.kind != "composite":
            raise ConversionError(f"Pandera schema '{schema.name}' did not resolve to a composite canonical type.")

        # Convert CanonicalType back to a Pydantic model class
        # This relies on PydanticTypeSystem.from_canonical being implemented correctly
        try:
            pydantic_model_class = _pyd_sys.from_canonical(canonical_schema)
        except TypeError as e:
            if 'unhashable type' in str(e):
                # Handle unhashable type error by creating a sanitized copy
                import copy
                import json
                
                # Create sanitized versions of the metadata and params
                sanitized_meta_data = {}
                if canonical_schema.meta and hasattr(canonical_schema.meta, 'data'):
                    for k, v in canonical_schema.meta.data.items():
                        if isinstance(v, (dict, list)):
                            sanitized_meta_data[k] = json.dumps(v)
                        else:
                            sanitized_meta_data[k] = v
                
                # Make params hashable
                sanitized_params = {}
                if canonical_schema.params:
                    for k, v in canonical_schema.params.items():
                        if isinstance(v, (dict, list)):
                            try:
                                # Try using make_hashable
                                sanitized_params[k] = make_hashable(v)
                            except:
                                # Fallback to JSON string
                                sanitized_params[k] = json.dumps(v)
                        else:
                            sanitized_params[k] = v
                
                # Create a completely new CanonicalType with sanitized data
                clean_schema = CanonicalType(
                    kind=canonical_schema.kind,
                    name=canonical_schema.name,
                    params=sanitized_params,
                    constraints=canonical_schema.constraints,
                    meta=TypeMeta(data=sanitized_meta_data)
                )
                
                # Try again with the sanitized schema
                try:
                    pydantic_model_class = _pyd_sys.from_canonical(clean_schema)
                except Exception as e2:
                    # If this also fails, create a fallback model
                    return _create_basic_model_from_schema(schema)
            else:
                # If it's a different TypeError, create a fallback model
                return _create_basic_model_from_schema(schema)

        # Validate the output type
        if not (isinstance(pydantic_model_class, type) and issubclass(pydantic_model_class, BaseModel)):
            raise ConversionError(f"Pydantic system did not return a BaseModel subclass from canonical schema {canonical_schema.name}. Got: {type(pydantic_model_class)}")

        return pydantic_model_class

    except NotImplementedError as nie:
        # Specific feedback if from_canonical is the blocker
        raise NotImplementedError(f"Conversion requires PydanticTypeSystem.from_canonical implementation for composite types: {nie}") from nie
    except (ConversionError, PolitipoError, TypeError):
        # Re-raise known errors directly
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise ConversionError(f"Unexpected error during Pandera -> Pydantic schema conversion for schema '{schema.name}': {e}") from e


def _create_basic_model_from_schema(schema: 'pa.DataFrameSchema') -> Type[BaseModel]:
    """
    Create a basic Pydantic model from a Pandera schema when the full conversion fails.
    
    Args:
        schema: The Pandera DataFrameSchema
        
    Returns:
        A minimal but functional Pydantic BaseModel class
    """
    # Import create_model from pydantic for dynamic model creation
    from pydantic import create_model
    
    # Dictionary for field definitions
    fields = {}
    
    # Extract column information
    for name, column in schema.columns.items():
        # Map Pandera dtype to Python type
        python_type = str  # Default to string
        if column.dtype:
            dtype_str = str(column.dtype).lower()
            if 'int' in dtype_str:
                python_type = int
            elif 'float' in dtype_str:
                python_type = float
            elif 'bool' in dtype_str:
                python_type = bool
            elif 'datetime' in dtype_str:
                python_type = datetime.datetime
        
        # Handle nullable/optional fields
        if column.nullable:
            from typing import Optional as OptionalType
            field_type = OptionalType[python_type]
            default = None
        else:
            field_type = python_type
            default = ...  # Ellipsis means required field in Pydantic
            
        # Add field to definitions
        fields[name] = (field_type, default)
    
    # Create dynamic model
    model_name = getattr(schema, 'name', 'DynamicModel') or 'DynamicModel'
    model_cls = create_model(model_name, **fields)
    
    return model_cls

# TODO: Consider adding similar functions for SeriesSchema <-> Pydantic Field/Type if needed. 
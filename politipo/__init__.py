"""Politipo: Type conversion and validation for data models."""

# Core components
from politipo.core.types import CanonicalType, TypeMeta
from politipo.core.conversion import ConversionEngine
from politipo.core.errors import ConversionError, PolitipoError

# Plugin type systems
from politipo.plugins.python import PythonTypeSystem
from politipo.plugins.pydantic import PydanticTypeSystem
from politipo.plugins.sqlalchemy import SQLAlchemyTypeSystem, SQLAlchemyModelTypeSystem
from politipo.plugins.sqlmodel import SQLModelTypeSystem
from politipo.plugins.pandas import PandasTypeSystem
from politipo.plugins.polars import PolarsTypeSystem

# Optional plugins that depend on external packages
try:
    from politipo.plugins.pandera import PanderaTypeSystem
    _HAS_PANDERA = True
except ImportError:
    _HAS_PANDERA = False
    PanderaTypeSystem = None  # Define as None if import fails

# Optional schema tools
try:
    # Only try importing if dependencies are met
    if _HAS_PANDERA and PydanticTypeSystem is not None:
        from .schema_tools import pydantic_to_pandera_schema, pandera_to_pydantic_model
        _HAS_SCHEMA_TOOLS = True
    else:
        _HAS_SCHEMA_TOOLS = False
except ImportError:
    _HAS_SCHEMA_TOOLS = False
    # Define dummies if needed for type checking elsewhere, though __all__ controls export
    def pydantic_to_pandera_schema(*args, **kwargs): raise ImportError("Missing dependencies for schema_tools")
    def pandera_to_pydantic_model(*args, **kwargs): raise ImportError("Missing dependencies for schema_tools")


__all__ = [
    # Core
    'CanonicalType',
    'TypeMeta',
    'ConversionEngine',
    'ConversionError',
    'PolitipoError',
    # Type Systems
    'PythonTypeSystem',
    'PydanticTypeSystem',
    'SQLAlchemyTypeSystem',
    'SQLAlchemyModelTypeSystem',
    'SQLModelTypeSystem',
    'PandasTypeSystem',
    'PolarsTypeSystem',
]

# Add optional type systems and tools if available
if _HAS_PANDERA:
    __all__.append('PanderaTypeSystem')
if _HAS_SCHEMA_TOOLS:
    __all__.extend([
        'pydantic_to_pandera_schema',
        'pandera_to_pydantic_model',
    ])
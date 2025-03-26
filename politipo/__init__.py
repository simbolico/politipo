"""Politipo: Type Harmony for Python."""

from politipo.core.conversion import ConversionEngine
from politipo.core.types import TypeSystem, CanonicalType
from politipo.core.errors import ConversionError, PolitipoError

# Plugin type systems
from politipo.plugins.python import PythonTypeSystem
from politipo.plugins.pydantic import PydanticTypeSystem
from politipo.plugins.pandas import PandasTypeSystem
from politipo.plugins.polars import PolarsTypeSystem
from politipo.plugins.sqlalchemy import SQLAlchemyTypeSystem, SQLAlchemyModelTypeSystem
from politipo.plugins.sqlmodel import SQLModelTypeSystem
from politipo.plugins.pandera import PanderaTypeSystem

__all__ = [
    # Core components
    "ConversionEngine",
    "TypeSystem",
    "CanonicalType",
    "ConversionError",
    "PolitipoError",
    
    # Type systems
    "PythonTypeSystem",
    "PydanticTypeSystem",
    "PandasTypeSystem",
    "PolarsTypeSystem",
    "SQLAlchemyTypeSystem",
    "SQLAlchemyModelTypeSystem",
    "SQLModelTypeSystem",
    "PanderaTypeSystem",
]
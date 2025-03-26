from dataclasses import dataclass
from typing import Optional
from politipo.core.types import CanonicalType, TypeSystem


@dataclass
class ConversionContext:
    """Context object for type conversion operations."""
    source: CanonicalType
    target: CanonicalType
    source_type_system: TypeSystem
    target_type_system: Optional[TypeSystem]
    strict: bool = True  # Whether to enforce strict validation 
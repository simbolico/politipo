from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from politipo.core.types import CanonicalType
    from politipo.core.conversion.engine import ConversionContext


class ConversionStrategy(ABC):
    """Base class for all type conversion strategies."""

    @abstractmethod
    def can_handle(
        self, 
        source: "CanonicalType", 
        target: "CanonicalType"
    ) -> bool:
        """
        Whether this strategy can handle the conversion.
        
        Args:
            source: Source type to convert from
            target: Target type to convert to
            
        Returns:
            True if this strategy can handle the conversion
        """
        pass

    @abstractmethod
    def convert(
        self, 
        value: Any,
        context: "ConversionContext"
    ) -> Any:
        """
        Perform the actual conversion.
        
        Args:
            value: The value to convert
            context: Conversion context with type information
            
        Returns:
            The converted value
        """
        pass 
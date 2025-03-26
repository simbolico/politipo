from abc import ABC, abstractmethod
from typing import Any
from politipo.core.types import CanonicalType
from politipo.core.conversion.context import ConversionContext


class ConversionStrategy(ABC):
    """Base class for type conversion strategies."""

    @abstractmethod
    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """
        Check if this strategy can handle the conversion.

        Args:
            source: Source canonical type
            target: Target canonical type

        Returns:
            True if this strategy can handle the conversion
        """
        pass

    @abstractmethod
    def convert(self, value: Any, context: ConversionContext) -> Any:
        """
        Convert a value using this strategy.

        Args:
            value: The value to convert
            context: Conversion context with source/target types and settings

        Returns:
            The converted value

        Raises:
            ConversionError: If conversion fails
        """
        pass 
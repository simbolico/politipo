from abc import ABC, abstractmethod
from typing import Any

from politipo.core.types import CanonicalType


class ConversionStrategy(ABC):
    """Base class for all conversion strategies."""

    @abstractmethod
    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """
        Returns True if this strategy can handle the conversion.

        Args:
            source: Source canonical type
            target: Target canonical type

        Returns:
            True if this strategy can handle the conversion
        """
        pass

    @abstractmethod
    def convert(self, value: Any, target: CanonicalType, strict: bool = True) -> Any:
        """
        Convert a value to the target type.

        Args:
            value: The value to convert
            target: Target canonical type
            strict: Whether to enforce strict type checking

        Returns:
            The converted value

        Raises:
            ConversionError: If conversion fails
        """
        pass

    def get_priority(self) -> int:
        """
        Returns the priority of this strategy (higher = more specific).

        Default implementations should return 0.
        More specific implementations (e.g., direct conversions) should return higher values.
        """
        return 0

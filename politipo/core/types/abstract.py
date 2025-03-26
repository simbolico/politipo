from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimpleTypeMeta:
    """Concrete implementation for storing type metadata as a dictionary."""
    data: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "SimpleTypeMeta") -> "SimpleTypeMeta":
        """
        Combines two metadata objects, with 'other' overriding.
        
        Args:
            other: The metadata to merge with this one
            
        Returns:
            A new SimpleTypeMeta with merged data
        """
        merged_data = {**self.data, **other.data}
        return SimpleTypeMeta(data=merged_data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Helper method to access metadata.
        
        Args:
            key: The key to look up
            default: Value to return if key not found
            
        Returns:
            The value for the key or the default
        """
        return self.data.get(key, default)


# Make TypeMeta an alias for SimpleTypeMeta for potential future flexibility
TypeMeta = SimpleTypeMeta


class Constraint(ABC):
    """Base class for all type constraints."""
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Returns True if value satisfies the constraint."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the constraint for storage."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        """Deserializes a constraint."""
        pass 
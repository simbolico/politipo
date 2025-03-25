from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable


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


class TypeMeta(ABC):
    """Metadata attached to canonical types (e.g., SQL 'NOT NULL')."""
    
    @abstractmethod
    def merge(self, other: "TypeMeta") -> "TypeMeta":
        """Combines two metadata objects."""
        pass 
from typing import Protocol, runtime_checkable, Any, Dict
from abc import abstractmethod


@runtime_checkable
class TypeSystem(Protocol):
    """Protocol that all type system plugins must implement."""
    __slots__ = ()  # Prevent dynamic attribute creation
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the type system (e.g., 'pydantic')."""
        pass

    @abstractmethod
    def to_canonical(self, type_obj: Any) -> "CanonicalType":
        """
        Converts a native type to CanonicalType.
        
        Args:
            type_obj: The native type object to convert
            
        Returns:
            Canonical representation of the type
            
        Raises:
            ConversionError: If conversion fails
        """
        pass

    @abstractmethod
    def from_canonical(self, canonical: "CanonicalType") -> Any:
        """
        Reconstructs a native type from CanonicalType.
        
        Args:
            canonical: The canonical type to convert from
            
        Returns:
            Native type representation
            
        Raises:
            ConversionError: If conversion fails
        """
        pass

    @abstractmethod
    def detect(self, obj: Any) -> bool:
        """
        Returns True if the object belongs to this type system.
        
        Args:
            obj: Object to check
            
        Returns:
            True if this type system can handle the object
        """
        pass

    @abstractmethod
    def get_default_canonical(self) -> "CanonicalType":
        """
        Returns a default/generic CanonicalType for this system.
        
        Returns:
            A CanonicalType representing the most generic/default type for this system.
            For example:
            - Pandas: DataFrame with no columns
            - Pydantic: dict
            - Python: Any
            - SQLModel: SQLModel base class
            - Polars: DataFrame with no columns
        
        This is used when the target is specified only as a string system name,
        without any specific type information.
        """
        pass


@runtime_checkable
class ConstraintProtocol(Protocol):
    """Protocol for constraint implementations."""
    __slots__ = ()

    def validate(self, value: Any) -> bool:
        """
        Validate a value against this constraint.
        
        Args:
            value: Value to validate
            
        Returns:
            True if value satisfies the constraint
            
        Raises:
            ConstraintViolationError: If validation fails with details
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize constraint to dictionary.
        
        Returns:
            Dictionary representation of the constraint
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintProtocol":
        """
        Create constraint from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            New constraint instance
            
        Raises:
            ValueError: If data is invalid
        """
        ... 
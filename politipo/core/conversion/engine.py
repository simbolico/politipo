from dataclasses import dataclass
from typing import Dict, Type, Any, Optional, Union, List
from politipo.core.types import CanonicalType, TypeSystem
from politipo.core.conversion.graph import TypeGraph
from politipo.core.conversion.strategies.base import ConversionStrategy


@dataclass
class ConversionContext:
    """Context object passed to conversion strategies."""
    source: CanonicalType
    target: CanonicalType
    source_type_system: TypeSystem
    target_type_system: TypeSystem
    strict: bool = True


class ConversionEngine:
    """Core engine for managing type conversions between different systems."""

    def __init__(self):
        self._type_systems: Dict[str, TypeSystem] = {}
        self._strategies: Dict[str, ConversionStrategy] = {}
        self._graph = TypeGraph()

    def register_type_system(self, type_system: TypeSystem) -> None:
        """
        Register a new type system plugin.
        
        Args:
            type_system: The type system to register
            
        Raises:
            ValueError: If a type system with the same name is already registered
        """
        if type_system.name in self._type_systems:
            raise ValueError(f"TypeSystem '{type_system.name}' already registered")
        self._type_systems[type_system.name] = type_system
        self._graph.add_type_system(type_system)

    def register_strategy(self, name: str, strategy: ConversionStrategy) -> None:
        """
        Register a new conversion strategy.
        
        Args:
            name: Unique name for the strategy
            strategy: The strategy implementation
            
        Raises:
            ValueError: If a strategy with the same name exists
        """
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' already registered")
        self._strategies[name] = strategy

    def convert(
        self,
        value: Any,
        target: Union[str, TypeSystem],
        *,
        strict: bool = True
    ) -> Any:
        """
        Convert a value to a target type system.
        
        Args:
            value: The value to convert
            target: Target type system name or instance
            strict: Whether to enforce strict type checking
            
        Returns:
            The converted value
            
        Raises:
            ValueError: If no conversion path is found
        """
        # Detect source type system
        source_system = self._detect_source_system(value)
        
        # Resolve target system
        target_system = (
            self._type_systems[target] if isinstance(target, str) 
            else target
        )

        # Get canonical types
        source_type = source_system.to_canonical(value)
        target_type = target_system.from_canonical(source_type)

        # Create context
        context = ConversionContext(
            source=source_type,
            target=target_type,
            source_type_system=source_system,
            target_type_system=target_system,
            strict=strict
        )

        # Find conversion path
        path = self._graph.find_path(
            source=source_system,
            target=target_system,
            value=value
        )

        # Execute conversion chain
        current_value = value
        for strategy_name in path:
            strategy = self._strategies[strategy_name]
            current_value = strategy.convert(current_value, context)

        return current_value

    def _detect_source_system(self, value: Any) -> TypeSystem:
        """Find the type system that can handle the given value."""
        for system in self._type_systems.values():
            if system.detect(value):
                return system
        raise ValueError(f"No TypeSystem found for value: {type(value)}") 
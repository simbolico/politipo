from typing import Any, Dict, List, Optional, Type, Union
from politipo.core.types import TypeSystem, CanonicalType
from politipo.core.errors import ConversionError
from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.conversion.context import ConversionContext


class ConversionEngine:
    """Core engine for type conversion operations."""

    def __init__(self, strategies: Optional[List[ConversionStrategy]] = None):
        self._strategies = strategies or []
        self._type_systems: Dict[str, TypeSystem] = {}

    def register_strategy(self, strategy: Union[str, ConversionStrategy], strategy_impl: Optional[ConversionStrategy] = None) -> None:
        """
        Register a new conversion strategy.

        Args:
            strategy: Either the strategy instance or a name for the strategy
            strategy_impl: The strategy implementation if a name was provided

        Raises:
            ValueError: If strategy_impl is not provided when strategy is a string
        """
        if isinstance(strategy, str):
            if strategy_impl is None:
                raise ValueError("strategy_impl must be provided when registering with a name")
            self._strategies.append(strategy_impl)
        else:
            self._strategies.append(strategy)

    def register_type_system(self, type_system: TypeSystem) -> None:
        """
        Register a new type system.

        Args:
            type_system: The type system to register

        Raises:
            ValueError: If a type system with the same name is already registered
        """
        if type_system.name in self._type_systems:
            raise ValueError(f"TypeSystem '{type_system.name}' already registered")
        self._type_systems[type_system.name] = type_system

    def convert(
        self,
        value: Any,
        target: Union[str, Type, CanonicalType, TypeSystem],
        strict: bool = True
    ) -> Any:
        """
        Convert a value to the target type.
        
        Args:
            value: The value to convert
            target: Target type specification, can be:
                   - str: Name of target type system (e.g., "pandas", "pydantic")
                   - Type: Native type class (e.g., MyPydanticModel, pd.DataFrame)
                   - CanonicalType: Pre-constructed canonical type
                   - TypeSystem: Type system instance (advanced use)
            strict: Whether to enforce strict type checking
            
        Returns:
            The converted value
            
        Raises:
            ConversionError: If conversion fails
        """
        # Resolve source canonical type from value's type
        source_canonical = self._resolve_canonical(value, None, is_source=True)
        
        # Resolve target canonical type from specification
        target_canonical = self._resolve_canonical(None, target, is_source=False)
        
        # Find appropriate strategy
        strategy = self._find_strategy(source_canonical, target_canonical)
        if not strategy:
            raise ConversionError(
                f"No conversion strategy found from {source_canonical.name} to {target_canonical.name}"
            )
            
        # Execute conversion
        try:
            # Resolve source and target type systems for the context
            source_system = self._detect_system_for_type(type(value)) or self._get_type_system("python")
            target_system = (
                self._detect_system_for_type(target) if isinstance(target, type)
                else self._get_type_system(target) if isinstance(target, str)
                else target if isinstance(target, TypeSystem)
                else None
            )

            context = ConversionContext(
                source=source_canonical,
                target=target_canonical,
                source_type_system=source_system,
                target_type_system=target_system,
                strict=strict
            )
            
            return strategy.convert(value, context)
        except Exception as e:
            # Add more context to the error message
            raise ConversionError(f"Conversion failed using strategy {type(strategy).__name__}: {e}") from e

    def _resolve_canonical(
        self,
        value_or_type: Any,  # Can be the value (for source) or the type (for target)
        type_spec: Optional[Union[str, Type, CanonicalType, TypeSystem]],
        is_source: bool
    ) -> CanonicalType:
        """
        Resolve type specification to canonical type.
        
        Args:
            value_or_type: The value (for source) or type (for target)
            type_spec: Type specification
            is_source: Whether resolving source or target type
            
        Returns:
            Resolved CanonicalType
            
        Raises:
            ConversionError: If resolution fails
        """
        if isinstance(type_spec, CanonicalType):
            return type_spec

        system: Optional[TypeSystem] = None
        native_type: Optional[Type] = None

        if isinstance(type_spec, TypeSystem):
            system = type_spec
        elif isinstance(type_spec, str):
            system = self._get_type_system(type_spec)
            if not system:
                raise ConversionError(f"Type system '{type_spec}' not registered")
        elif isinstance(type_spec, type):
            native_type = type_spec
            # Need a way to detect the system from the native type
            system = self._detect_system_for_type(native_type)
            if not system:
                raise ConversionError(f"Cannot determine TypeSystem for type {native_type}")
        elif type_spec is None and is_source:
            # For source, detect from value's type
            value_type = type(value_or_type)
            system = self._detect_system_for_type(value_type)
            if not system:
                # Fallback to Python system for basic types
                system = self._get_type_system("python")
            native_type = value_type
        else:
            raise ConversionError(f"Invalid type specification: {type_spec}")

        try:
            if is_source:
                # For source, always use the actual type
                return system.to_canonical(native_type or type(value_or_type))
            else:
                # For target, use native_type if provided, otherwise get default
                if native_type:
                    return system.to_canonical(native_type)
                else:
                    return system.get_default_canonical()
        except Exception as e:
            context = "source" if is_source else "target"
            raise ConversionError(f"Failed to resolve {context} type: {e}") from e

    def _detect_system_for_type(self, native_type: Type) -> Optional[TypeSystem]:
        """Find the registered TypeSystem that detects the given native type."""
        for system in self._type_systems.values():
            if system.detect(native_type):
                return system
        return None

    def _get_type_system(self, type_spec: Union[CanonicalType, TypeSystem, str]) -> Optional[TypeSystem]:
        """Get TypeSystem instance from specification."""
        if isinstance(type_spec, TypeSystem):
            return type_spec
        elif isinstance(type_spec, str):
            if type_spec not in self._type_systems:
                raise ConversionError(f"Type system '{type_spec}' not registered")
            return self._type_systems[type_spec]
        elif isinstance(type_spec, CanonicalType):
            # Extract type system from canonical type's metadata if available
            system_name = type_spec.metadata.get("type_system")
            if system_name and system_name in self._type_systems:
                return self._type_systems[system_name]
        return None

    def _find_strategy(self, source: CanonicalType, target: CanonicalType) -> Optional[ConversionStrategy]:
        """Find appropriate strategy for conversion."""
        for strategy in self._strategies:
            if strategy.can_handle(source, target):
                return strategy
        return None 
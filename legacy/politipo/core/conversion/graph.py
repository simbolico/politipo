from collections import deque
from typing import Any

from politipo.core.types import TypeSystem


class TypeGraph:
    """Graph structure for managing type system conversion paths."""

    def __init__(self):
        self._nodes: dict[str, TypeSystem] = {}
        self._edges: dict[str, dict[str, str]] = {}  # {source: {target: strategy_name}}

    def add_type_system(self, system: TypeSystem) -> None:
        """
        Register a type system node.

        Args:
            system: The type system to add
        """
        self._nodes[system.name] = system
        if system.name not in self._edges:
            self._edges[system.name] = {}

    def register_strategy(self, source: TypeSystem, target: TypeSystem, strategy: str) -> None:
        """
        Add a conversion path between systems.

        Args:
            source: Source type system
            target: Target type system
            strategy: Name of the strategy to use
        """
        if source.name not in self._edges:
            self._edges[source.name] = {}
        self._edges[source.name][target.name] = strategy

    def find_path(self, source: TypeSystem, target: TypeSystem, value: Any) -> list[str]:
        """
        Find optimal conversion path using BFS.

        Args:
            source: Source type system
            target: Target type system
            value: The value being converted (for optimization hints)

        Returns:
            List of strategy names forming the conversion path

        Raises:
            ValueError: If no path exists between source and target
        """
        if source.name == target.name:
            return []

        # Check for direct strategy
        if target.name in self._edges.get(source.name, {}):
            return [self._edges[source.name][target.name]]

        # BFS for path
        queue = deque([(source.name, [])])
        visited = {source.name}

        while queue:
            current, path = queue.popleft()

            for next_system, strategy in self._edges[current].items():
                if next_system == target.name:
                    return path + [strategy]

                if next_system not in visited:
                    visited.add(next_system)
                    queue.append((next_system, path + [strategy]))

        raise ValueError(f"No conversion path found from {source.name} to {target.name}")

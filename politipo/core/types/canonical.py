from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional, List
from .abstract import Constraint, SimpleTypeMeta


@dataclass(frozen=True)
class CanonicalType:
    """Immutable representation of a type across all systems."""
    kind: Literal["primitive", "container", "composite", "generic"]
    name: str  # e.g., "int", "DataFrame", "List"
    params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Constraint] = field(default_factory=dict)
    meta: Optional[SimpleTypeMeta] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.kind == "container" and "item_type" not in self.params:
            raise ValueError("Container types must specify 'item_type'")
        
        if self.kind == "generic" and not self.params.get("type_params"):
            raise ValueError("Generic types require type_params")

    def with_constraint(self, constraint: Constraint) -> "CanonicalType":
        """Returns a new CanonicalType with an added constraint."""
        new_constraints = {**self.constraints, constraint.__class__.__name__: constraint}
        return CanonicalType(
            kind=self.kind,
            name=self.name,
            params=self.params,
            constraints=new_constraints,
            meta=self.meta
        ) 
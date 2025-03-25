from typing import Any, Dict, Type, Union, get_origin, get_args
from pydantic import BaseModel, Field, create_model
from politipo.core.types import TypeSystem, CanonicalType


class PydanticTypeSystem(TypeSystem):
    """Type system implementation for Pydantic models with nested model support."""
    
    name = "pydantic"

    def to_canonical(self, model: Type[BaseModel]) -> CanonicalType:
        """
        Convert a Pydantic model to canonical type.
        
        Args:
            model: Pydantic model class
            
        Returns:
            Canonical type representation
        """
        fields = {}
        for name, field in model.model_fields.items():
            field_type = self._py_type_to_canonical(field.annotation)
            constraints = self._field_to_constraints(field)
            
            fields[name] = {
                "type": field_type,
                "constraints": constraints,
                "required": field.is_required()
            }
            
        return CanonicalType(
            kind="composite",
            name=model.__name__,
            params={"fields": fields}
        )

    def from_canonical(self, canonical: CanonicalType) -> Type[BaseModel]:
        """
        Create a Pydantic model from canonical type.
        
        Args:
            canonical: Canonical type representation
            
        Returns:
            Generated Pydantic model class
        """
        fields = {}
        for name, field_info in canonical.params["fields"].items():
            py_type = self._canonical_to_py_type(field_info["type"])
            field_kwargs = self._constraints_to_field_kwargs(
                field_info["constraints"]
            )
            
            if not field_info.get("required", True):
                field_kwargs["default"] = None
                
            fields[name] = (py_type, Field(**field_kwargs))
            
        return create_model(canonical.name, **fields)

    def detect(self, obj: Any) -> bool:
        """Check if object is a Pydantic model."""
        return isinstance(obj, BaseModel) or (
            isinstance(obj, type) and issubclass(obj, BaseModel)
        )

    def _py_type_to_canonical(self, py_type: Type) -> CanonicalType:
        """
        Handle nested models and generic containers.
        
        Args:
            py_type: Python type to convert
            
        Returns:
            Canonical type representation
        """
        # Handle Pydantic models recursively
        if isinstance(py_type, type) and issubclass(py_type, BaseModel):
            return self.to_canonical(py_type)
        
        # Handle Optional[T]
        if get_origin(py_type) is Union:
            args = get_args(py_type)
            if type(None) in args:  # Optional[T]
                return self._py_type_to_canonical(
                    next(t for t in args if t is not type(None))
                )
        
        # Handle List[Model], Dict[str, Model], etc.
        origin = get_origin(py_type)
        if origin in (list, set, tuple, dict):
            args = get_args(py_type)
            if origin is dict:
                return CanonicalType(
                    kind="container",
                    name="dict",
                    params={
                        "key_type": self._py_type_to_canonical(args[0]),
                        "value_type": self._py_type_to_canonical(args[1])
                    }
                )
            return CanonicalType(
                kind="container",
                name=origin.__name__,
                params={"item_type": self._py_type_to_canonical(args[0])}
            )
        
        # Primitive types
        type_map = {
            str: ("primitive", "str"),
            int: ("primitive", "int"),
            float: ("primitive", "float"),
            bool: ("primitive", "bool")
        }
        if py_type in type_map:
            kind, name = type_map[py_type]
            return CanonicalType(kind=kind, name=name)
        
        return CanonicalType(kind="primitive", name="any")

    def _canonical_to_py_type(self, canonical: CanonicalType) -> Type:
        """Convert canonical type to Python type."""
        if canonical.kind == "primitive":
            type_map = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "any": Any
            }
            return type_map.get(canonical.name, Any)
            
        if canonical.kind == "container":
            container_map = {
                "list": list,
                "set": set,
                "tuple": tuple,
                "dict": dict
            }
            container = container_map[canonical.name]
            if canonical.name == "dict":
                key_type = self._canonical_to_py_type(
                    canonical.params["key_type"]
                )
                value_type = self._canonical_to_py_type(
                    canonical.params["value_type"]
                )
                return Dict[key_type, value_type]
            else:
                item_type = self._canonical_to_py_type(
                    canonical.params["item_type"]
                )
                return container[item_type]
            
        return Any

    def _field_to_constraints(self, field: Any) -> Dict[str, Any]:
        """Extract constraints from Pydantic field."""
        constraints = {}
        
        if field.ge is not None:
            constraints["min_value"] = field.ge
        if field.le is not None:
            constraints["max_value"] = field.le
        if field.min_length is not None:
            constraints["min_length"] = field.min_length
        if field.max_length is not None:
            constraints["max_length"] = field.max_length
            
        return constraints

    def _constraints_to_field_kwargs(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Convert constraints to Pydantic field arguments."""
        kwargs = {}
        
        if "min_value" in constraints:
            kwargs["ge"] = constraints["min_value"]
        if "max_value" in constraints:
            kwargs["le"] = constraints["max_value"]
        if "min_length" in constraints:
            kwargs["min_length"] = constraints["min_length"]
        if "max_length" in constraints:
            kwargs["max_length"] = constraints["max_length"]
            
        return kwargs 
# ./politipo/plugins/pydantic/types.py

import json
import sys
from functools import lru_cache
from typing import (
    Any,
    Optional,
    Union,
    get_args,
    get_origin,
)

try:
    from pydantic import BaseModel, EmailStr, Field, SecretStr, create_model
    from pydantic.fields import FieldInfo

    # Try importing specific v2/v1 things for constraint mapping
    try:  # Pydantic v2
        from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
        from pydantic.fields import FieldInfo as ModelField  # Use FieldInfo as ModelField in v2
        from pydantic_core import PydanticUndefined

        _PYDANTIC_V2 = True
    except ImportError:  # Pydantic v1
        from pydantic.fields import ModelField

        _PYDANTIC_V2 = False
        PydanticUndefined = None  # Define for compatibility check
except ImportError:
    # Allow type checking if pydantic not installed
    BaseModel = None
    Field = None
    FieldInfo = None
    ModelField = None
    EmailStr = None
    SecretStr = None
    _PYDANTIC_V2 = False
    PydanticUndefined = None
    # Define dummies for type hints if needed
    if "annotated_types" not in sys.modules:
        Gt = Ge = Lt = Le = MinLen = MaxLen = MultipleOf = type("DummyAnnotatedType", (), {})
    if "pydantic.fields" not in sys.modules:
        ModelField = type("DummyModelField", (), {})


from politipo.core.errors import ConversionError, PolitipoError
from politipo.core.types import CanonicalType, TypeMeta, TypeSystem

# Import constraint classes
from politipo.core.types.constraints import (
    Constraint,
    GreaterThan,
    LessThan,
    MaxLength,
    MaxValue,
    MinLength,
    MinValue,
    MultipleOf,
    OneOf,
    Pattern,
)

# Import PythonTypeSystem to delegate primitive/container mapping
from politipo.plugins.python import PythonTypeSystem

# Helper dict for constraint mapping
_PYDANTIC_CONSTRAINT_MAP = {
    # Field attributes (v1 & v2) / Annotated types (v2)
    "gt": GreaterThan,
    "ge": MinValue,  # Maps ge to MinValue (inclusive)
    "lt": LessThan,
    "le": MaxValue,  # Maps le to MaxValue (inclusive)
    "min_length": MinLength,
    "max_length": MaxLength,
    "pattern": Pattern,
    "multiple_of": MultipleOf,
    "allowed_values": OneOf,  # Add OneOf mapping
    # Annotated types specific mapping (ensure correct class is checked)
    Gt: GreaterThan,
    Ge: MinValue,
    Lt: LessThan,
    Le: MaxValue,
    MinLen: MinLength,
    MaxLen: MaxLength,
    MultipleOf: MultipleOf,
}


class PydanticTypeSystem(TypeSystem):
    """Type system implementation for Pydantic models with enhanced features."""

    name = "pydantic"

    def __init__(self):
        if BaseModel is None:
            raise PolitipoError("Pydantic is not installed. Cannot initialize PydanticTypeSystem.")
        # Use PythonTypeSystem for handling primitives and standard containers
        self._python_system = PythonTypeSystem()

    def get_default_canonical(self) -> CanonicalType:
        """Returns a generic dict type as the default Pydantic target."""
        return CanonicalType(
            kind="container",
            name="dict",
            params={
                "key_type": CanonicalType(kind="primitive", name="str"),
                "value_type": CanonicalType(kind="primitive", name="any"),
            },
            meta=TypeMeta(data={"origin_system": self.name}),
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: type[BaseModel]) -> CanonicalType:
        """Convert a Pydantic model or type to CanonicalType."""

        # Handle specific Pydantic types first
        if type_obj is EmailStr:
            return CanonicalType(kind="primitive", name="str", params={"format": "email"})
        if type_obj is SecretStr:
            return CanonicalType(kind="primitive", name="str", params={"format": "secret"})
        # Add more specific Pydantic types here (UUID, Url, etc.)

        # Handle Pydantic Model Classes
        if isinstance(type_obj, type) and issubclass(type_obj, BaseModel):
            fields = {}
            model_fields_dict = {}
            required_set = set()

            try:  # Pydantic v2+
                model_fields_dict = getattr(type_obj, "model_fields", {})
                required_set = {
                    name for name, field in model_fields_dict.items() if field.is_required()
                }
            except AttributeError:
                try:  # Pydantic v1
                    model_fields_dict = getattr(type_obj, "__fields__", {})
                    required_set = {
                        name for name, field in model_fields_dict.items() if field.required
                    }
                except AttributeError:
                    raise ConversionError(f"Cannot inspect fields for Pydantic model {type_obj}")

            for name, field in model_fields_dict.items():
                # Delegate type conversion (handles primitives, containers, nested models)
                # Need field.annotation (v2) or field.outer_type_ (v1)
                field_annotation = getattr(field, "annotation", getattr(field, "outer_type_", Any))
                field_type_canonical = self._map_pydantic_annotation_to_canonical(field_annotation)

                # Extract constraints
                constraints = self._extract_constraints(field)

                fields[name] = {
                    "type": field_type_canonical,
                    "constraints": constraints,
                    "required": name in required_set,
                    # Add default value, description etc. if needed
                    "description": getattr(field, "description", None),
                }

            # Store full schema for reconstruction fidelity
            try:
                schema = type_obj.model_json_schema() if _PYDANTIC_V2 else type_obj.schema()
                # Convert schema to tuple for hashability if used directly in params
                # params = {"fields": fields, "json_schema_tuple": self._dict_to_tuple(schema)}
                # Storing as string is simpler for caching
                params = {"fields": fields, "json_schema_str": json.dumps(schema)}
            except Exception:
                # Fallback if schema generation fails
                params = {"fields": fields}

            # Create metadata with schema_type for Pandera compatibility
            meta_data = {
                "origin_system": self.name,
                "schema_type": "dataframe",  # Set schema_type to dataframe for Pydantic models
                "model_name": type_obj.__name__,
            }

            return CanonicalType(
                kind="composite",
                name=type_obj.__name__,
                params=params,
                meta=TypeMeta(data=meta_data),
            )

        # Handle Annotated types not part of a model field (less common)
        origin = get_origin(type_obj)
        if (
            origin is Union
            and getattr(sys.modules.get("typing"), "Annotated", None)
            and type(None) not in get_args(type_obj)
        ):
            args = get_args(type_obj)
            if len(args) > 1 and isinstance(args[1], FieldInfo):
                # It's likely Annotated[Type, Field(...)]
                base_type = args[0]
                field_info = args[1]
                base_canonical = self._map_pydantic_annotation_to_canonical(base_type)
                constraints = self._extract_constraints(field_info)
                # Merge constraints into base canonical type
                new_constraints = {**base_canonical.constraints, **constraints}
                return CanonicalType(
                    kind=base_canonical.kind,
                    name=base_canonical.name,
                    params=base_canonical.params,
                    constraints=new_constraints,
                    meta=base_canonical.meta,
                )

        # If not a model or specific Pydantic type, delegate to Python system
        return self._python_system.to_canonical(type_obj)

    def _map_pydantic_annotation_to_canonical(self, annotation: type) -> CanonicalType:
        """Helper to convert Pydantic field annotation, handling nesting."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Optional[T]
        if origin is Union and type(None) in args:
            inner_type = next((arg for arg in args if arg is not type(None)), Any)
            # Recursively call for inner type
            return self._map_pydantic_annotation_to_canonical(
                inner_type
            )  # Nullability handled by 'required'

        # Handle containers (List[Model], Dict[str, Model], etc.)
        if origin in (list, list, tuple, tuple, set, set, frozenset, frozenset):
            container_name = origin.__name__.lower().replace("[]", "")
            item_type = (
                self._map_pydantic_annotation_to_canonical(args[0])
                if args
                else CanonicalType(kind="primitive", name="any")
            )
            return CanonicalType(
                kind="container", name=container_name, params={"item_type": item_type}
            )

        if origin in (dict, dict):
            key_type = (
                self._map_pydantic_annotation_to_canonical(args[0])
                if args
                else CanonicalType(kind="primitive", name="any")
            )
            value_type = (
                self._map_pydantic_annotation_to_canonical(args[1])
                if args and len(args) > 1
                else CanonicalType(kind="primitive", name="any")
            )
            return CanonicalType(
                kind="container",
                name="dict",
                params={"key_type": key_type, "value_type": value_type},
            )

        # Handle nested Pydantic models (recursive call to main method)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self.to_canonical(annotation)

        # Handle specific Pydantic types
        if annotation is EmailStr:
            return CanonicalType(kind="primitive", name="str", params={"format": "email"})
        if annotation is SecretStr:
            return CanonicalType(kind="primitive", name="str", params={"format": "secret"})

        # Otherwise, assume it's a standard Python type and delegate
        return self._python_system.to_canonical(annotation)

    @lru_cache(maxsize=128)
    def from_canonical(self, canonical: CanonicalType) -> type:
        """
        Reconstructs a Pydantic type or model from CanonicalType.

        Args:
            canonical: The canonical type representation to convert.

        Returns:
            A Pydantic type (for primitives/containers) or model class (for composites).

        Raises:
            ConversionError: If reconstruction fails or required data is missing.
        """
        # Handle Pydantic Model reconstruction from fields
        if canonical.kind == "composite":
            fields_info = canonical.params.get("fields")
            if not fields_info:
                raise ConversionError(
                    f"Cannot reconstruct Pydantic model '{canonical.name}' without field definitions."
                )

            fields_for_create_model = {}
            for name, field_info in fields_info.items():
                try:
                    # Get field type
                    field_type_canonical = field_info["type"]
                    py_type = self._canonical_to_pydantic_type(field_type_canonical)

                    # Get constraints and convert to field kwargs
                    constraints = field_info.get("constraints", {})
                    field_kwargs = self._constraints_to_field_kwargs(constraints)

                    # Add description if present
                    if field_info.get("description"):
                        field_kwargs["description"] = field_info["description"]

                    # Handle required/optional and defaults
                    is_required = field_info.get("required", True)
                    default_value = field_info.get("default", ...)

                    if is_required:
                        # Required fields might still have defaults in Pydantic
                        current_default = default_value if default_value is not ... else ...
                        field_definition = Field(default=current_default, **field_kwargs)
                    else:
                        # Optional field: Ensure type hint is Optional and default is None unless specified
                        if get_origin(py_type) is not Union or type(None) not in get_args(py_type):
                            py_type = Optional[py_type]
                        # Use canonical default if provided, otherwise default optional fields to None
                        final_default = default_value if default_value is not ... else None
                        field_definition = Field(default=final_default, **field_kwargs)

                    fields_for_create_model[name] = (py_type, field_definition)

                except Exception as e:
                    raise ConversionError(
                        f"Error processing field '{name}' for Pydantic model '{canonical.name}': {e}"
                    ) from e

            # Handle Model Config
            config_dict = canonical.meta.get("pydantic_config") if canonical.meta else None
            config_kwarg = {}
            if config_dict:
                if _PYDANTIC_V2:
                    # For v2, pass model_config directly
                    config_kwarg["model_config"] = config_dict
                else:
                    # For v1, create Config class
                    config_kwarg["__config__"] = type("Config", (), config_dict)

            try:
                # Create the model
                model = create_model(canonical.name, **fields_for_create_model, **config_kwarg)
                # Add model docstring if available
                if canonical.meta and canonical.meta.get("description"):
                    model.__doc__ = canonical.meta.get("description")
                return model
            except Exception as e:
                raise ConversionError(
                    f"Failed to create Pydantic model '{canonical.name}': {e}"
                ) from e

        # Handle Container Types
        if canonical.kind == "container":
            container_map: dict[str, type] = {
                "list": list,
                "tuple": tuple,
                "set": set,
                "frozenset": frozenset,
                "dict": dict,
            }
            if canonical.name in container_map:
                container = container_map[canonical.name]
                if canonical.name == "dict":
                    key_type = self._canonical_to_pydantic_type(
                        canonical.params.get(
                            "key_type", CanonicalType(kind="primitive", name="any")
                        )
                    )
                    value_type = self._canonical_to_pydantic_type(
                        canonical.params.get(
                            "value_type", CanonicalType(kind="primitive", name="any")
                        )
                    )
                    return container[key_type, value_type]
                else:
                    item_type = self._canonical_to_pydantic_type(
                        canonical.params.get(
                            "item_type", CanonicalType(kind="primitive", name="any")
                        )
                    )
                    return container[item_type]

        # Handle Primitive Types (including special formats)
        if canonical.kind == "primitive":
            if canonical.params.get("format") == "email":
                return EmailStr
            if canonical.params.get("format") == "secret":
                return SecretStr

            # Delegate basic primitives to Python system
            return self._python_system.from_canonical(canonical)

        # Fallback
        return Any

    def _canonical_to_pydantic_type(self, canonical: CanonicalType) -> type:
        """Helper to convert canonical back to a type usable in Pydantic."""
        # Handle nested models recursively
        if canonical.kind == "composite":
            return self.from_canonical(canonical)
        # Handle containers recursively
        if canonical.kind == "container":
            return self.from_canonical(canonical)
        # Handle primitives (including formats)
        if canonical.kind == "primitive":
            if canonical.params.get("format") == "email":
                return EmailStr
            if canonical.params.get("format") == "secret":
                return SecretStr
            # Delegate basic primitives
            return self._python_system.from_canonical(canonical)

        return Any  # Fallback

    def detect(self, obj: Any) -> bool:
        """Returns True if the object is a Pydantic model class or specific type."""
        if BaseModel is None:
            return False

        # Check for specific Pydantic types
        if obj in (EmailStr, SecretStr):  # Add more specific types if needed
            return True

        # Check if it's a Pydantic model class
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            return True

        # Check for Annotated types originating from Pydantic Field usage? (Harder to detect reliably)

        return False

    def _extract_constraints(self, field: FieldInfo | ModelField) -> dict[str, Constraint]:
        """Extracts constraints from a Pydantic FieldInfo or ModelField."""
        constraints = {}
        if field is None:
            return constraints

        # Common attributes/metadata keys
        constraint_keys = [
            "gt",
            "ge",
            "lt",
            "le",
            "min_length",
            "max_length",
            "pattern",
            "multiple_of",
            "allowed_values",
        ]

        # Pydantic v2: Constraints often in metadata or direct attributes
        if _PYDANTIC_V2 and isinstance(field, FieldInfo):
            # Check metadata (e.g., from Annotated)
            if field.metadata:
                for item in field.metadata:
                    # Check annotated_types like Gt(0)
                    for anno_type_cls, core_constraint_cls in _PYDANTIC_CONSTRAINT_MAP.items():
                        # Check if item is instance of anno_type_cls if anno_type_cls is a type
                        if isinstance(anno_type_cls, type) and isinstance(item, anno_type_cls):
                            # Extract value (e.g., item.gt, item.min_length)
                            # Assumes annotated types have single attr with constraint value
                            constraint_value = next(
                                (
                                    getattr(item, attr)
                                    for attr in constraint_keys
                                    if hasattr(item, attr)
                                ),
                                None,
                            )
                            if constraint_value is not None:
                                constraints[core_constraint_cls.__name__] = core_constraint_cls(
                                    value=constraint_value
                                )
                                break  # Found constraint for this item
            # Check direct attributes on FieldInfo
            for key in constraint_keys:
                if key in _PYDANTIC_CONSTRAINT_MAP:
                    value = getattr(field, key, None)
                    # Pydantic v2 uses PydanticUndefined for unset values
                    if value is not None and value is not PydanticUndefined:
                        constraint_cls = _PYDANTIC_CONSTRAINT_MAP[key]
                        if key == "allowed_values":
                            constraints[constraint_cls.__name__] = constraint_cls(
                                allowed_values=value
                            )
                        else:
                            constraints[constraint_cls.__name__] = constraint_cls(value=value)

        # Pydantic v1: Constraints often direct attributes on ModelField or in field.type_.__constraints__
        elif not _PYDANTIC_V2 and isinstance(field, ModelField):
            for key in constraint_keys:
                if key in _PYDANTIC_CONSTRAINT_MAP:
                    value = getattr(
                        field.field_info, key, None
                    )  # Constraints are on field_info in v1
                    if value is not None:
                        constraint_cls = _PYDANTIC_CONSTRAINT_MAP[key]
                        if key == "allowed_values":
                            constraints[constraint_cls.__name__] = constraint_cls(
                                allowed_values=value
                            )
                        else:
                            constraints[constraint_cls.__name__] = constraint_cls(value=value)
            # Also check __constraints__ on the type itself (e.g., for conint)
            if hasattr(field.type_, "__constraints__"):
                for constraint in field.type_.__constraints__:
                    # Handle Pydantic v1 constraint objects
                    if hasattr(constraint, "allowed_values"):
                        constraints["OneOf"] = OneOf(allowed_values=constraint.allowed_values)
                    elif hasattr(constraint, "gt"):
                        constraints["GreaterThan"] = GreaterThan(value=constraint.gt)
                    elif hasattr(constraint, "ge"):
                        constraints["MinValue"] = MinValue(value=constraint.ge)
                    elif hasattr(constraint, "lt"):
                        constraints["LessThan"] = LessThan(value=constraint.lt)
                    elif hasattr(constraint, "le"):
                        constraints["MaxValue"] = MaxValue(value=constraint.le)
                    elif hasattr(constraint, "min_length"):
                        constraints["MinLength"] = MinLength(value=constraint.min_length)
                    elif hasattr(constraint, "max_length"):
                        constraints["MaxLength"] = MaxLength(value=constraint.max_length)
                    elif hasattr(constraint, "pattern"):
                        constraints["Pattern"] = Pattern(pattern=constraint.pattern)
                    elif hasattr(constraint, "multiple_of"):
                        constraints["MultipleOf"] = MultipleOf(value=constraint.multiple_of)

        return constraints

    def _constraints_to_field_kwargs(self, constraints: dict[str, Constraint]) -> dict[str, Any]:
        """Convert constraints back to Pydantic field kwargs."""
        kwargs = {}
        for constraint in constraints.values():
            if isinstance(constraint, GreaterThan):
                kwargs["gt"] = constraint.value
            elif isinstance(constraint, LessThan):
                kwargs["lt"] = constraint.value
            elif isinstance(constraint, MinValue):
                kwargs["ge"] = constraint.value
            elif isinstance(constraint, MaxValue):
                kwargs["le"] = constraint.value
            elif isinstance(constraint, MinLength):
                kwargs["min_length"] = constraint.value
            elif isinstance(constraint, MaxLength):
                kwargs["max_length"] = constraint.value
            elif isinstance(constraint, Pattern):
                kwargs["pattern"] = constraint.pattern
            elif isinstance(constraint, MultipleOf):
                kwargs["multiple_of"] = constraint.value
            elif isinstance(constraint, OneOf):
                kwargs["allowed_values"] = constraint.allowed_values
        return kwargs

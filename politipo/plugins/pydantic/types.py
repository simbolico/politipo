# ./politipo/plugins/pydantic/types.py

from typing import Any, Dict, Type, Union, get_origin, get_args, Optional, List, Tuple, Set, FrozenSet
from functools import lru_cache
import datetime
import decimal
import sys
import json
import inspect

try:
    from pydantic import BaseModel, Field, create_model, EmailStr, SecretStr
    from pydantic.fields import FieldInfo
    # Try importing specific v2/v1 things for constraint mapping
    try: # Pydantic v2
        from annotated_types import Gt, Ge, Lt, Le, MinLen, MaxLen, MultipleOf
        from pydantic_core import PydanticUndefined
        from pydantic.fields import FieldInfo as ModelField  # Use FieldInfo as ModelField in v2
        _PYDANTIC_V2 = True
    except ImportError: # Pydantic v1
        from pydantic.fields import ModelField
        _PYDANTIC_V2 = False
        PydanticUndefined = None # Define for compatibility check
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
    if 'annotated_types' not in sys.modules:
        Gt = Ge = Lt = Le = MinLen = MaxLen = MultipleOf = type('DummyAnnotatedType', (), {})
    if 'pydantic.fields' not in sys.modules:
         ModelField = type('DummyModelField', (), {})


from politipo.core.types import TypeSystem, CanonicalType, TypeMeta
from politipo.core.errors import ConversionError, PolitipoError
# Import constraint classes
from politipo.core.types.constraints import (
    MinValue, MaxValue, MinLength, MaxLength,
    GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual,
    Pattern, MultipleOf, Constraint
)
# Import PythonTypeSystem to delegate primitive/container mapping
from politipo.plugins.python import PythonTypeSystem

# Helper dict for constraint mapping
_PYDANTIC_CONSTRAINT_MAP = {
    # Field attributes (v1 & v2) / Annotated types (v2)
    'gt': GreaterThan,
    'ge': MinValue, # Maps ge to MinValue (inclusive)
    'lt': LessThan,
    'le': MaxValue, # Maps le to MaxValue (inclusive)
    'min_length': MinLength,
    'max_length': MaxLength,
    'pattern': Pattern,
    'multiple_of': MultipleOf,
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
                "value_type": CanonicalType(kind="primitive", name="any")
            },
            meta=TypeMeta(data={"origin_system": self.name})
        )

    @lru_cache(maxsize=128)
    def to_canonical(self, type_obj: Type[BaseModel]) -> CanonicalType:
        """Convert a Pydantic model or type to CanonicalType."""

        # Handle specific Pydantic types first
        if type_obj is EmailStr:
             return CanonicalType(kind="primitive", name="str", params={"format": "email"})
        if type_obj is SecretStr:
             return CanonicalType(kind="primitive", name="str", params={"format": "secret"})

        # Handle Pydantic Model Classes
        if isinstance(type_obj, type) and issubclass(type_obj, BaseModel):
            fields_canonical = {}
            model_fields_dict = {}
            required_set = set()

            try: # Pydantic v2+
                model_fields_dict = getattr(type_obj, 'model_fields', {})
                required_set = {name for name, field in model_fields_dict.items() if field.is_required()}
            except AttributeError:
                 try: # Pydantic v1
                     model_fields_dict = getattr(type_obj, '__fields__', {})
                     required_set = {name for name, field in model_fields_dict.items() if field.required}
                 except AttributeError:
                     raise ConversionError(f"Cannot inspect fields for Pydantic model {type_obj}")

            for name, field in model_fields_dict.items():
                field_annotation = getattr(field, 'annotation', getattr(field, 'outer_type_', Any))
                field_type_canonical = self._map_pydantic_annotation_to_canonical(field_annotation)
                constraints = self._extract_constraints(field)

                # Extract default value
                default_value = ...  # Ellipsis represents no default set
                if _PYDANTIC_V2:
                    # In v2, default is directly on FieldInfo
                    raw_default = getattr(field, 'default', PydanticUndefined)
                    if raw_default is not PydanticUndefined:
                        default_value = raw_default
                else:
                    # In v1, default is on field_info within ModelField
                    field_info_v1 = getattr(field, 'field_info', None)
                    if field_info_v1:
                       raw_default = getattr(field_info_v1, 'default', ...)
                       if raw_default is not ...:
                           default_value = raw_default

                fields_canonical[name] = {
                    "type": field_type_canonical,
                    "constraints": constraints,
                    "required": name in required_set,
                    "default": default_value,
                    "description": getattr(field, 'description', None),
                }

            # Extract model config
            model_config_dict = {}
            if _PYDANTIC_V2:
                # Pydantic v2 stores config in model_config attribute (dict)
                cfg = getattr(type_obj, 'model_config', {})
                if cfg:
                    try:
                       # Only store serializable config for broader compatibility
                       model_config_dict = json.loads(json.dumps(cfg))
                    except TypeError:
                        model_config_dict = {"error": "Config not JSON serializable"}
            else:
                # Pydantic v1 uses a Config class
                config_class = getattr(type_obj, 'Config', None)
                if config_class:
                    model_config_dict = {
                        key: getattr(config_class, key)
                        for key in dir(config_class)
                        if not key.startswith('_') and not callable(getattr(config_class, key))
                    }

            # Extract model docstring for description
            model_description = inspect.getdoc(type_obj)

            meta_data = {
                "origin_system": self.name,
                "pydantic_config": model_config_dict,
                "description": model_description,
            }

            # Store full schema for reconstruction fidelity
            try:
                schema = type_obj.model_json_schema() if _PYDANTIC_V2 else type_obj.schema()
                params = {"fields": fields_canonical, "json_schema_str": json.dumps(schema)}
            except Exception:
                # Fallback if schema generation fails
                params = {"fields": fields_canonical}

            return CanonicalType(
                kind="composite",
                name=type_obj.__name__,
                params=params,
                meta=TypeMeta(data=meta_data)
            )

        # Handle Annotated types not part of a model field (less common)
        origin = get_origin(type_obj)
        if origin is Union and getattr(sys.modules.get('typing'), 'Annotated', None) and type(None) not in get_args(type_obj):
             args = get_args(type_obj)
             if len(args) > 1 and isinstance(args[1], FieldInfo):
                  # It's likely Annotated[Type, Field(...)]
                  base_type = args[0]
                  field_info = args[1]
                  base_canonical = self._map_pydantic_annotation_to_canonical(base_type)
                  constraints = self._extract_constraints(field_info)
                  # Merge constraints into base canonical type
                  new_constraints = {**base_canonical.constraints, **constraints}
                  return CanonicalType(kind=base_canonical.kind, name=base_canonical.name, params=base_canonical.params, constraints=new_constraints, meta=base_canonical.meta)


        # If not a model or specific Pydantic type, delegate to Python system
        return self._python_system.to_canonical(type_obj)


    def _map_pydantic_annotation_to_canonical(self, annotation: Type) -> CanonicalType:
         """Helper to convert Pydantic field annotation, handling nesting."""
         origin = get_origin(annotation)
         args = get_args(annotation)

         # Handle Optional[T]
         if origin is Union and type(None) in args:
             inner_type = next((arg for arg in args if arg is not type(None)), Any)
             # Recursively call for inner type
             return self._map_pydantic_annotation_to_canonical(inner_type) # Nullability handled by 'required'

         # Handle containers (List[Model], Dict[str, Model], etc.)
         if origin in (list, List, tuple, Tuple, set, Set, frozenset, FrozenSet):
             container_name = origin.__name__.lower().replace('[]','')
             item_type = self._map_pydantic_annotation_to_canonical(args[0]) if args else CanonicalType(kind="primitive", name="any")
             return CanonicalType(kind="container", name=container_name, params={"item_type": item_type})

         if origin in (dict, Dict):
             key_type = self._map_pydantic_annotation_to_canonical(args[0]) if args else CanonicalType(kind="primitive", name="any")
             value_type = self._map_pydantic_annotation_to_canonical(args[1]) if args and len(args)>1 else CanonicalType(kind="primitive", name="any")
             return CanonicalType(kind="container", name="dict", params={"key_type": key_type, "value_type": value_type})

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
    def from_canonical(self, canonical: CanonicalType) -> Type:
        """Reconstructs a Pydantic type or model from CanonicalType."""

        # Handle Pydantic Model reconstruction from full schema if available
        if canonical.kind == "composite" and "json_schema_str" in canonical.params:
            try:
                schema_dict = json.loads(canonical.params["json_schema_str"])
                # Requires a helper function similar to TypeMapper._create_pydantic_model_from_schema
                return self._create_model_from_schema(canonical.name, schema_dict)
            except Exception:
                 # Fallback to field-based reconstruction if schema fails
                 pass # Continue to logic below

        # Handle Pydantic Model reconstruction from fields
        if canonical.kind == "composite":
            fields_def = {}
            field_params = canonical.params.get("fields", {})
            if not field_params: # Cannot reconstruct without fields
                 raise ConversionError(f"Cannot reconstruct Pydantic model '{canonical.name}' without field definitions in params.")

            for name, field_info in field_params.items():
                field_type_canonical = field_info["type"]
                py_type = self._canonical_to_pydantic_type(field_type_canonical)

                constraints = field_info.get("constraints", {})
                field_kwargs = self._constraints_to_field_kwargs(constraints)

                if 'description' in field_info:
                     field_kwargs['description'] = field_info['description']

                # Handle optional vs required
                is_required = field_info.get("required", True)
                if is_required:
                    # Field(...) implies required unless default is set
                    if not field_kwargs:
                         fields_def[name] = (py_type, ...) # Ellipsis means required
                    else:
                         fields_def[name] = (py_type, Field(..., **field_kwargs))
                else:
                    # Make it Optional[py_type] and provide default None
                    # Note: py_type might already be Optional if derived from Optional[T]
                    if get_origin(py_type) is not Union or type(None) not in get_args(py_type):
                         py_type = Optional[py_type]
                    # Set default to None unless another default exists in kwargs
                    field_kwargs.setdefault('default', None)
                    fields_def[name] = (py_type, Field(**field_kwargs))

            return create_model(canonical.name, **fields_def)

        # Handle Container Types
        if canonical.kind == "container":
            container_map: Dict[str, Type] = {"list": List, "tuple": Tuple, "set": Set, "frozenset": FrozenSet, "dict": Dict}
            if canonical.name in container_map:
                container = container_map[canonical.name]
                if canonical.name == "dict":
                    key_type = self._canonical_to_pydantic_type(canonical.params.get("key_type", CanonicalType(kind="primitive", name="any")))
                    value_type = self._canonical_to_pydantic_type(canonical.params.get("value_type", CanonicalType(kind="primitive", name="any")))
                    return container[key_type, value_type]
                else:
                    item_type = self._canonical_to_pydantic_type(canonical.params.get("item_type", CanonicalType(kind="primitive", name="any")))
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


    def _canonical_to_pydantic_type(self, canonical: CanonicalType) -> Type:
         """Helper to convert canonical back to a type usable in Pydantic."""
         # Handle nested models recursively
         if canonical.kind == "composite":
             return self.from_canonical(canonical)
         # Handle containers recursively
         if canonical.kind == "container":
             return self.from_canonical(canonical)
         # Handle primitives (including formats)
         if canonical.kind == "primitive":
              if canonical.params.get("format") == "email": return EmailStr
              if canonical.params.get("format") == "secret": return SecretStr
              # Delegate basic primitives
              return self._python_system.from_canonical(canonical)

         return Any # Fallback


    def detect(self, obj: Any) -> bool:
        """Returns True if the object is a Pydantic model class or specific type."""
        if BaseModel is None: return False

        # Check for specific Pydantic types
        if obj in (EmailStr, SecretStr): # Add more specific types if needed
            return True

        # Check if it's a Pydantic model class
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            return True

        # Check for Annotated types originating from Pydantic Field usage? (Harder to detect reliably)

        return False


    def _extract_constraints(self, field: Union[FieldInfo, ModelField]) -> Dict[str, Constraint]:
        """Extracts constraints from a Pydantic FieldInfo or ModelField."""
        constraints = {}
        if field is None: return constraints

        # Common attributes/metadata keys
        constraint_keys = ['gt', 'ge', 'lt', 'le', 'min_length', 'max_length', 'pattern', 'multiple_of']

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
                              constraint_value = next((getattr(item, attr) for attr in constraint_keys if hasattr(item, attr)), None)
                              if constraint_value is not None:
                                   constraints[core_constraint_cls.__name__] = core_constraint_cls(value=constraint_value)
                                   break # Found constraint for this item
            # Check direct attributes on FieldInfo
            for key in constraint_keys:
                 if key in _PYDANTIC_CONSTRAINT_MAP:
                     value = getattr(field, key, None)
                     # Pydantic v2 uses PydanticUndefined for unset values
                     if value is not None and value is not PydanticUndefined:
                          constraint_cls = _PYDANTIC_CONSTRAINT_MAP[key]
                          constraints[constraint_cls.__name__] = constraint_cls(value=value)

        # Pydantic v1: Constraints often direct attributes on ModelField or in field.type_.__constraints__
        elif not _PYDANTIC_V2 and isinstance(field, ModelField):
             for key in constraint_keys:
                  if key in _PYDANTIC_CONSTRAINT_MAP:
                       value = getattr(field.field_info, key, None) # Constraints are on field_info in v1
                       if value is not None:
                            constraint_cls = _PYDANTIC_CONSTRAINT_MAP[key]
                            constraints[constraint_cls.__name__] = constraint_cls(value=value)
             # Also check __constraints__ on the type itself (e.g., for conint)
             if hasattr(field.type_, '__constraints__'):
                 pyd_constraints = field.type_.__constraints__
                 for key, value in pyd_constraints.items():
                      if key in _PYDANTIC_CONSTRAINT_MAP and value is not None:
                           constraint_cls = _PYDANTIC_CONSTRAINT_MAP[key]
                           constraints[constraint_cls.__name__] = constraint_cls(value=value)

        return constraints


    def _constraints_to_field_kwargs(self, constraints: Dict[str, Constraint]) -> Dict[str, Any]:
        """Convert core constraints back to Pydantic Field kwargs."""
        kwargs = {}
        # Inverse mapping (Core Constraint -> Pydantic Kwarg name)
        # Note: MinValue maps to 'ge', MaxValue maps to 'le'
        inverse_map = {
            GreaterThan: 'gt', MinValue: 'ge', LessThan: 'lt', MaxValue: 'le',
            MinLength: 'min_length', MaxLength: 'max_length', Pattern: 'pattern',
            MultipleOf: 'multiple_of',
            # Map GreaterThanOrEqual back to ge, LessThanOrEqual back to le
            GreaterThanOrEqual: 'ge', LessThanOrEqual: 'le',
        }
        for constraint_obj in constraints.values():
            kwarg_name = inverse_map.get(type(constraint_obj))
            if kwarg_name:
                 # Special handling for pattern if needed (regex vs pattern string)
                 value = constraint_obj.regex if isinstance(constraint_obj, Pattern) else constraint_obj.value
                 kwargs[kwarg_name] = value
        return kwargs

    # Helper similar to TypeMapper's - needed for schema reconstruction
    def _create_model_from_schema(self, model_name: str, schema: Dict) -> Type[BaseModel]:
         """Creates a Pydantic model from a JSON schema dictionary."""
         # This logic would be adapted from TypeMapper._create_pydantic_model_from_schema
         # It needs to parse schema['properties'], schema['required'] etc.
         # And map JSON schema types/formats/constraints back to Pydantic types/Field kwargs
         # Using self._canonical_to_pydantic_type and self._constraints_to_field_kwargs might help here
         # For brevity, returning a dummy model here. Replace with full implementation.
         print(f"Warning: _create_model_from_schema not fully implemented. Creating dummy model for {model_name}.")
         fields_from_schema = {}
         properties = schema.get('properties', {})
         required = schema.get('required', [])
         for name, prop_schema in properties.items():
             # Basic type mapping example
             json_type = prop_schema.get('type')
             py_type: Type = Any
             if json_type == 'integer': py_type = int
             elif json_type == 'number': py_type = float
             elif json_type == 'string': py_type = str
             elif json_type == 'boolean': py_type = bool
             # TODO: Handle formats, constraints, arrays, objects recursively
             is_req = name in required
             default = ... if is_req else prop_schema.get('default', None)
             fields_from_schema[name] = (py_type, default)

         if not fields_from_schema: # Create dummy if parse failed
              return create_model(model_name, __config__=None, dummy_field=(Any, None))
         else:
              return create_model(model_name, **fields_from_schema)
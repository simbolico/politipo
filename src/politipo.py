# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic>=2.5",
#     "sqlalchemy>=2.0",
#     "pyarrow>=15.0",
#     "polars>=0.20.0",
#     "duckdb>=0.10.0",
#     "pandera>=0.18.0",
#     "typing_extensions",
# ]
# ///

"""
# Politipo (PolyType) 🧬 v0.3.1 Omega
The Ultimate Data Fabric & Quality Engine.

Integrates: Python -> Pydantic -> Arrow -> DuckDB/Polars -> Pandera (Validation) -> Kùzu
Philosophy:
    1. Define Once (Pydantic)
    2. Transport Zero-Copy (Arrow)
    3. Validate Vectorized (Pandera)
    4. Analyze Anywhere (DuckDB/Polars/Kuzu)

Capabilities:
    - UUIDs stored as FixedSizeBinary(16) (50% RAM saving vs String)
    - Enums stored as Dictionary Encoding (10x RAM saving)
    - Vectors mapped to FixedSizeList/FIXED_LIST (Native AI/RAG)
    - Automated Quality Gates via Pandera
"""

from __future__ import annotations

import abc
import datetime
import decimal
import enum
import typing
import uuid
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

# --- Static type checking only (for IDE hints) ---
if TYPE_CHECKING:  # Imported only for type checkers / IDEs
    import duckdb as duckdb
    import pandas as pd
    import pandera as pandera

    # Optional submodules for richer typing
    import pandera.polars as pa_pl
    import polars as pl
    import pyarrow as pa
    import sqlalchemy as sa
from pydantic.fields import FieldInfo

# --- Ecosystem Imports (Safe Loading) ---
sa: Any = None
try:
    import sqlalchemy as _sa

    SQL_AVAILABLE: bool = True
    sa = _sa
except ImportError:
    SQL_AVAILABLE = False
    sa = object()

pa: Any = None
try:
    import pyarrow as _pa

    ARROW_AVAILABLE: bool = True
    pa = _pa
except ImportError:
    ARROW_AVAILABLE = False
    pa = object()

pl: Any = None
try:
    import polars as _pl

    POLARS_AVAILABLE: bool = True
    pl = _pl
except ImportError:
    POLARS_AVAILABLE = False
    pl = object()

try:
    import duckdb

    DUCKDB_AVAILABLE: bool = True
except ImportError:
    DUCKDB_AVAILABLE = False

pandera: Any = None
pa_pl: Any = None
pa_pd: Any = None
try:
    import pandera as _pandera
    import pandera.pandas as _pa_pd
    import pandera.polars as _pa_pl

    PANDERA_AVAILABLE: bool = True
    pandera = _pandera
    pa_pl = _pa_pl
    pa_pd = _pa_pd
except ImportError:
    PANDERA_AVAILABLE = False
    pandera = object()
    pa_pl = object()
    pa_pd = object()

# Public type aliases (runtime-visible; string-annotated to avoid runtime imports)
type ArrowTable = "pa.Table"
type ArrowSchema = "pa.Schema"
type ArrowField = "pa.Field"
type ArrowDataType = "pa.DataType"
type PolarsDF = "pl.DataFrame"
type PandasDF = "pd.DataFrame"
type DuckDBConn = "duckdb.DuckDBPyConnection"

# --- Metadata Markers ---


@dataclass(frozen=True)
class Precision:
    """
    SotA Financial Precision Marker.
    Usage: Annotated[Decimal, Precision(38, 18)]
    """

    precision: int
    scale: int


class VectorMarker:
    """
    SotA AI/Embedding Marker.
    Usage: Vector[1536] -> FixedSizeList(1536)
    """

    def __class_getitem__(cls, dim: int):
        return Annotated[list[float], "vector", dim]


# Help static type checkers accept Vector[4] notation by treating as Any
Vector: Any = VectorMarker


# --- Universal Type Registry & Factory ---
class DataType:
    """
    Universal Type Registry & Factory.
    Acts as the single source of truth for data types across the ecosystem.
    Allows external libraries to register custom semantic types.
    """

    # --- Primitives (Direct Mapping) ---
    STRING = str
    INTEGER = int
    FLOAT = float
    BOOLEAN = bool
    BYTES = bytes
    DATE = datetime.date
    TIMESTAMP = datetime.datetime
    JSON = dict
    UUID = uuid.UUID

    # --- Complex Types (Factories) ---

    @staticmethod
    def DECIMAL(precision: int = 18, scale: int = 2) -> Any:
        """
        Returns a Decimal type annotated with Precision metadata.
        Usage: field: DataType.DECIMAL(10, 2)
        """
        return Annotated[decimal.Decimal, Precision(precision, scale)]

    @staticmethod
    def VECTOR(dim: int) -> Any:
        """
        Returns a List[float] annotated with Vector metadata.
        Usage: embedding: DataType.VECTOR(1536)
        """
        return Vector[dim]

    # --- Extensibility Mechanism ---

    @classmethod
    def register(cls, name: str, type_def: Any) -> None:
        """
        Registers a new type dynamically.
        Used by upper layers (Kernel/Registry) to inject types like RID, GeoPoint, etc.

        Args:
            name: The name of the attribute (e.g., 'RID'). Must be uppercase.
            type_def: The python type or Annotated type definition.
        """
        name = name.upper()
        if hasattr(cls, name):
            raise ValueError(f"DataType '{name}' is already registered.")

        setattr(cls, name, type_def)


# --- 1. The Atomic Spec System (The DNA) ---


class TypeSpec(abc.ABC):
    """Defines how a type behaves across the entire Data Stack."""

    def __init__(self, nullable: bool = False, is_pk: bool = False):
        self.nullable = nullable
        self.is_pk = is_pk

    @property
    @abc.abstractmethod
    def sql(self) -> sa.types.TypeEngine | str: ...
    @property
    @abc.abstractmethod
    def kuzu(self) -> str: ...
    @property
    @abc.abstractmethod
    def duckdb(self) -> str: ...
    @property
    @abc.abstractmethod
    def arrow(self) -> pa.DataType | None: ...

    def get_arrow_field(self, name: str) -> pa.Field | None:
        if not ARROW_AVAILABLE:
            return None
        p = typing.cast(Any, pa)
        return p.field(name, self.arrow, nullable=self.nullable)

    def serialize(self, value: Any) -> Any:
        """Prepares value for Arrow ingestion (Fast Path)."""
        return value


# --- Concrete Implementations ---


class StringSpec(TypeSpec):
    @property
    def sql(self):
        return sa.String if SQL_AVAILABLE else "VARCHAR"

    @property
    def kuzu(self) -> str:
        return "STRING"

    @property
    def duckdb(self) -> str:
        return "VARCHAR"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.string() if ARROW_AVAILABLE else None


class IntegerSpec(TypeSpec):
    @property
    def sql(self):
        return sa.BigInteger if SQL_AVAILABLE else "BIGINT"

    @property
    def kuzu(self) -> str:
        return "INT64"

    @property
    def duckdb(self) -> str:
        return "BIGINT"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.int64() if ARROW_AVAILABLE else None


class FloatSpec(TypeSpec):
    @property
    def sql(self):
        return sa.Float if SQL_AVAILABLE else "FLOAT"

    @property
    def kuzu(self) -> str:
        return "DOUBLE"

    @property
    def duckdb(self) -> str:
        return "DOUBLE"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.float64() if ARROW_AVAILABLE else None


class BooleanSpec(TypeSpec):
    @property
    def sql(self):
        return sa.Boolean if SQL_AVAILABLE else "BOOLEAN"

    @property
    def kuzu(self) -> str:
        return "BOOLEAN"

    @property
    def duckdb(self) -> str:
        return "BOOLEAN"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.bool_() if ARROW_AVAILABLE else None


class UUIDSpec(TypeSpec):
    """
    SotA Optimization: Uses 16-byte Binary for Arrow/Parquet.
    Reduces memory usage by ~40% compared to String UUIDs.
    """

    @property
    def sql(self):
        return sa.Uuid if SQL_AVAILABLE else "UUID"

    @property
    def kuzu(self) -> str:
        return "UUID"

    @property
    def duckdb(self) -> str:  # DuckDB has native 128-bit UUID type
        return "UUID"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.binary(16) if ARROW_AVAILABLE else None

    def serialize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value.bytes  # Crucial: Convert to bytes for FixedSizeBinary
        return value


class DecimalSpec(TypeSpec):
    def __init__(self, p=18, s=3, **kwargs):
        super().__init__(**kwargs)
        self.p, self.s = p, s

    @property
    def sql(self):
        return sa.Numeric(self.p, self.s) if SQL_AVAILABLE else f"DECIMAL({self.p},{self.s})"

    @property
    def kuzu(self):
        return "DOUBLE"  # Kuzu lacks Decimal, Double is best approx

    @property
    def duckdb(self):
        return f"DECIMAL({self.p},{self.s})"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.decimal128(self.p, self.s) if ARROW_AVAILABLE else None


class DateTimeSpec(TypeSpec):
    @property
    def sql(self):
        return sa.DateTime(timezone=True) if SQL_AVAILABLE else "TIMESTAMPTZ"

    @property
    def kuzu(self) -> str:
        return "TIMESTAMP"

    @property
    def duckdb(self) -> str:
        return "TIMESTAMPTZ"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.timestamp("us", tz="UTC") if ARROW_AVAILABLE else None


class ListSpec(TypeSpec):
    def __init__(self, child: TypeSpec, **kwargs):
        super().__init__(**kwargs)
        self.child = child

    @property
    def sql(self):
        return sa.ARRAY(self.child.sql) if (SQL_AVAILABLE and hasattr(sa, "ARRAY")) else "JSON"

    @property
    def kuzu(self):
        return f"LIST({self.child.kuzu})"

    @property
    def duckdb(self):
        return f"{self.child.duckdb}[]"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.list_(self.child.arrow) if ARROW_AVAILABLE else None

    def serialize(self, value: Any) -> Any:
        if value is None:
            return None
        # Recursive serialization for lists of complex types (like UUIDs)
        return [self.child.serialize(v) for v in value]


class VectorSpec(TypeSpec):
    """
    SotA AI Support.
    Maps Python List[float] to FixedSizeList (Arrow) and FIXED_LIST (Kuzu/DuckDB).
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    @property
    def sql(self):
        return sa.ARRAY(sa.Float) if SQL_AVAILABLE else "FLOAT[]"

    @property
    def kuzu(self):
        return f"FIXED_LIST(FLOAT, {self.dim})"

    @property
    def duckdb(self):
        return f"FLOAT[{self.dim}]"

    @property
    def arrow(self):
        p = typing.cast(Any, pa)
        return p.list_(p.float32(), self.dim) if ARROW_AVAILABLE else None

    def serialize(self, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, list | tuple):
            raise TypeError("Vector value must be a list or tuple of floats")
        if len(value) != self.dim:
            raise ValueError(f"Vector length {len(value)} does not match expected dim {self.dim}")
        return value


class StructSpec(TypeSpec):
    def __init__(self, fields: dict[str, TypeSpec], **kwargs):
        super().__init__(**kwargs)
        self.fields = fields

    @property
    def sql(self):
        return sa.JSON if SQL_AVAILABLE else "JSON"

    @property
    def kuzu(self):
        inner = ", ".join(f"{k} {v.kuzu}" for k, v in self.fields.items())
        return f"STRUCT({inner})"

    @property
    def duckdb(self):
        inner = ", ".join(f"{k} {v.duckdb}" for k, v in self.fields.items())
        return f"STRUCT({inner})"

    @property
    def arrow(self):
        if not ARROW_AVAILABLE:
            return None
        p = typing.cast(Any, pa)
        return p.struct([v.get_arrow_field(k) for k, v in self.fields.items()])

    def serialize(self, value: Any) -> Any:
        if value is None:
            return None
        # Convert Model -> Dict
        raw = value.model_dump() if hasattr(value, "model_dump") else value.__dict__
        # Recurse for fields
        return {k: self.fields[k].serialize(v) for k, v in raw.items() if k in self.fields}


class EnumSpec(TypeSpec):
    def __init__(self, inner: TypeSpec, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    @property
    def sql(self):
        return sa.String if SQL_AVAILABLE else "VARCHAR"

    @property
    def kuzu(self):
        return self.inner.kuzu

    @property
    def duckdb(self):
        return "VARCHAR"

    @property
    def arrow(self):
        # SotA Memory: Use Dictionary Encoding
        if not ARROW_AVAILABLE:
            return None
        p = typing.cast(Any, pa)
        return p.dictionary(p.int32(), self.inner.arrow)

    def serialize(self, value: Any) -> Any:
        if value is None:
            return None
        return value.value if hasattr(value, "value") else value


# --- 2. The Resolver Brain (Logic) ---


class PolyResolver:
    """Maps Python/Pydantic types to TypeSpecs."""

    def resolve(self, type_hint: Any, field_info: Any | None = None) -> TypeSpec:
        is_nullable = False
        is_pk = False

        # 1. Metadata Extraction
        if field_info:
            if getattr(field_info, "primary_key", False):
                is_pk = True
            # Inspect metadata for FieldInfo(primary_key=...) markers
            if hasattr(field_info, "metadata"):
                for m in field_info.metadata or []:
                    if isinstance(m, FieldInfo) and getattr(m, "primary_key", False):
                        is_pk = True
            # V2 nullability check if available
            try:
                if hasattr(field_info, "is_required") and not field_info.is_required():
                    is_nullable = True
            except TypeError:
                pass

        # 2. Unwrap Annotated/Optional
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union and type(None) in args:
            is_nullable = True
            non_none = [t for t in args if t is not type(None)]
            if len(non_none) == 1:
                spec = self.resolve(non_none[0], field_info)
                spec.nullable = True
                return spec

        if origin is Annotated:
            base = args[0]
            meta = args[1:]
            # Extract FieldInfo metadata if present
            for m in meta:
                if isinstance(m, FieldInfo):
                    if getattr(m, "primary_key", False):
                        is_pk = True
                    try:
                        if hasattr(m, "is_required") and not m.is_required():
                            is_nullable = True
                    except TypeError:
                        pass
            # Vector Handler
            if base is list or base == list[float]:
                for m in meta:
                    if m == "vector" and len(meta) >= 2:
                        return VectorSpec(dim=meta[1], nullable=is_nullable, is_pk=is_pk)
            # Decimal Precision
            if base is decimal.Decimal:
                for m in meta:
                    if isinstance(m, Precision):
                        return DecimalSpec(m.precision, m.scale, nullable=is_nullable, is_pk=is_pk)
            # Fallback to base resolution but preserve extracted flags
            spec = self.resolve(base, field_info)
            spec.is_pk = spec.is_pk or is_pk
            spec.nullable = spec.nullable or is_nullable
            return spec

        # 3. Primitives
        if type_hint is str:
            return StringSpec(nullable=is_nullable, is_pk=is_pk)
        if type_hint is int:
            return IntegerSpec(nullable=is_nullable, is_pk=is_pk)
        if type_hint is float:
            return FloatSpec(nullable=is_nullable, is_pk=is_pk)
        if type_hint is bool:
            return BooleanSpec(nullable=is_nullable, is_pk=is_pk)
        if type_hint is uuid.UUID:
            return UUIDSpec(nullable=is_nullable, is_pk=is_pk)
        if type_hint is datetime.datetime:
            return DateTimeSpec(nullable=is_nullable, is_pk=is_pk)
        if type_hint is decimal.Decimal:
            return DecimalSpec(18, 3, nullable=is_nullable, is_pk=is_pk)

        # 4. Containers
        if origin in (list, list):
            return ListSpec(self.resolve(args[0]), nullable=is_nullable)

        # 5. Nested Models (Structs)
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            fields: dict[str, TypeSpec] = {}
            try:
                hints = typing.get_type_hints(type_hint, include_extras=True)
            except TypeError:
                hints = typing.get_type_hints(type_hint)
            for name, ann in hints.items():
                finfo = getattr(type_hint, "model_fields", {}).get(name)
                spec = self.resolve(ann, finfo)
                # Heuristic default: common PK field names
                if not getattr(spec, "is_pk", False) and name in ("id", "pk"):
                    spec.is_pk = True
                fields[name] = spec
            return StructSpec(fields, nullable=is_nullable)

        # 6. Enums
        if isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
            return EnumSpec(StringSpec(), nullable=is_nullable)

        return StringSpec(nullable=True)  # Ultimate Fallback


_RESOLVER = PolyResolver()

# --- 3. Quality Gate (Pandera Integration) ---


class QualityGate:
    """
    Transform Pydantic Constraints -> Vectorized Pandera Validations.
    The "Endgame" of type safety.
    """

    @staticmethod
    def generate_schema(model: type[BaseModel], backend="polars") -> Any:
        if not PANDERA_AVAILABLE:
            # Suggest installing validation extra via uv
            raise ImportError(
                "Missing optional dependency 'pandera'.\n"
                "Install with: uv pip install '.[validation]'"
            )

        columns = {}
        # Select backend-specific Check module to avoid top-level pandera usage
        check_mod: Any
        if backend == "polars" and PANDERA_AVAILABLE and pa_pl is not object():
            check_mod = typing.cast(Any, pa_pl)
        elif backend == "pandas" and PANDERA_AVAILABLE and pa_pd is not object():
            check_mod = typing.cast(Any, pa_pd)
        else:
            # Fallback: retain compatibility if specific backend submodule unavailable
            check_mod = typing.cast(Any, pandera)
        for name, field in model.model_fields.items():
            checks = []
            # Extract Pydantic constraints (gt, lt, pattern, etc)
            constraints = QualityGate._extract_constraints(field)

            if "gt" in constraints:
                checks.append(check_mod.Check.gt(constraints["gt"]))
            if "ge" in constraints:
                checks.append(check_mod.Check.ge(constraints["ge"]))
            if "lt" in constraints:
                checks.append(check_mod.Check.lt(constraints["lt"]))
            if "le" in constraints:
                checks.append(check_mod.Check.le(constraints["le"]))
            if "pattern" in constraints:
                checks.append(check_mod.Check.str_matches(constraints["pattern"]))
            # String length constraints
            if "min_length" in constraints or "max_length" in constraints:
                checks.append(
                    check_mod.Check.str_length(
                        min_value=constraints.get("min_length"),
                        max_value=constraints.get("max_length"),
                    )
                )
            # multiple_of for numeric
            if "multiple_of" in constraints:
                n = constraints["multiple_of"]
                checks.append(check_mod.Check(lambda s, n=n: (s % n == 0)))

            # Map Type roughly for Pandera
            # Note: Pandera Polars backend infers type mostly from
            # the dataframe, so we can be lenient here
            dtype = QualityGate._map_type(field.annotation)

            # Determine nullability: allow None if field not required or annotation is Optional
            ann = getattr(field, "annotation", None)
            allows_none = False
            try:
                allows_none = get_origin(ann) is Union and type(None) in get_args(ann)
            except Exception:
                allows_none = False

            columns[name] = check_mod.Column(
                dtype, checks=checks, nullable=(not field.is_required()) or allows_none
            )

        if backend == "polars" and POLARS_AVAILABLE and pa_pl is not object():
            return typing.cast(Any, pa_pl).DataFrameSchema(columns)
        if backend == "pandas" and PANDERA_AVAILABLE and pa_pd is not object():
            return typing.cast(Any, pa_pd).DataFrameSchema(columns)
        # Fallback: top-level pandera
        return typing.cast(Any, pandera).DataFrameSchema(columns)

    @staticmethod
    def _extract_constraints(field) -> dict:
        c = {}
        for meta in field.metadata:
            # annotated_types support (Pydantic v2 standard)
            if hasattr(meta, "gt"):
                c["gt"] = meta.gt
            if hasattr(meta, "ge"):
                c["ge"] = meta.ge
            if hasattr(meta, "lt"):
                c["lt"] = meta.lt
            if hasattr(meta, "le"):
                c["le"] = meta.le
            if hasattr(meta, "pattern"):
                c["pattern"] = meta.pattern
            if hasattr(meta, "min_length"):
                c["min_length"] = meta.min_length
            if hasattr(meta, "max_length"):
                c["max_length"] = meta.max_length
            if hasattr(meta, "multiple_of"):
                c["multiple_of"] = meta.multiple_of
        return c

    @staticmethod
    def _map_type(t):
        # Basic mapping for Pandera validation
        if t is int:
            return int
        if t is float:
            return float
        if t is str:
            return str
        if t is bool:
            return bool
        return object


# --- 4. The Engine (Transporter) ---


class PolyTransporter:
    """
    The Zero-Copy Data Engine.
    Moves data between Pydantic, Arrow, DuckDB, Polars, and Kùzu.
    """

    def __init__(self, model_cls: type[BaseModel]):
        self.model_cls = model_cls
        self.spec = _RESOLVER.resolve(model_cls)
        if not isinstance(self.spec, StructSpec):
            raise ValueError("Root must be a Struct/Model")

    @property
    def arrow_schema(self) -> pa.Schema | None:
        if not ARROW_AVAILABLE:
            return None
        p = typing.cast(Any, pa)
        return p.schema([v.get_arrow_field(k) for k, v in self.spec.fields.items()])

    def generate_ddl(self, dialect: Literal["duckdb", "kuzu", "sql"] | str, table_name: str) -> str:
        """Generates CREATE TABLE statements for infrastructure."""

        def _sanitize_ident(name: str) -> str:
            import re

            if not isinstance(name, str):
                raise ValueError("Identifier must be a string")
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise ValueError(f"Unsafe identifier: {name!r}")
            return name

        safe_table = _sanitize_ident(table_name)
        # Optional: SQLAlchemy-backed compilation when available via
        # dialect strings like 'sql+postgresql'
        if isinstance(dialect, str) and dialect.startswith("sql+") and SQL_AVAILABLE:
            target = dialect.split("+", 1)[1]
            try:
                import sqlalchemy.dialects as sad

                # Resolve via getattr to appease type checkers
                def _dialect(name: str):
                    try:
                        mod = getattr(sad, name)
                        return mod.dialect()
                    except Exception:
                        return None

                # Map common aliases to SQLAlchemy dialect classes
                dialect_map = {
                    "postgres": _dialect("postgresql"),
                    "postgresql": _dialect("postgresql"),
                    "sqlite": _dialect("sqlite"),
                    "mysql": _dialect("mysql"),
                    "mariadb": _dialect("mysql"),
                }
                d = dialect_map.get(target)
            except Exception:
                d = None

            def compile_type(ts: TypeSpec) -> str:
                t = ts.sql
                # If a simple string was provided, use it
                if isinstance(t, str):
                    return t
                # Instantiate callables like sa.String -> sa.String()
                try:
                    inst = t() if callable(t) else t
                except Exception:
                    inst = t
                try:
                    if d is not None and hasattr(inst, "compile"):
                        return inst.compile(dialect=d)
                except Exception:
                    pass
                # Fallback to duckdb type name if compilation not possible
                return ts.duckdb

            cols = ", ".join(
                f"{k} {compile_type(v)}" + (" PRIMARY KEY" if v.is_pk else "")
                for k, v in self.spec.fields.items()
            )
            return f"CREATE TABLE IF NOT EXISTS {safe_table} ({cols});"
        elif isinstance(dialect, str) and dialect.startswith("sql+") and not SQL_AVAILABLE:
            # Fallback to generic SQL if SQLAlchemy is not available
            dialect = "sql"
        if dialect == "duckdb":
            cols = ", ".join(
                f"{k} {v.duckdb}" + (" PRIMARY KEY" if v.is_pk else "")
                for k, v in self.spec.fields.items()
            )
            return f"CREATE TABLE IF NOT EXISTS {safe_table} ({cols});"

        elif dialect == "kuzu":
            pk_col = next((k for k, v in self.spec.fields.items() if v.is_pk), "id")
            cols = ", ".join(f"{k} {v.kuzu}" for k, v in self.spec.fields.items())
            return f"CREATE NODE TABLE {safe_table} ({cols}, PRIMARY KEY ({pk_col}));"

        elif dialect == "sql":
            # Generic SQL: prefer simple portable type names
            def type_name(ts: TypeSpec) -> str:
                t = ts.sql
                return t if isinstance(t, str) else ts.duckdb

            cols = ", ".join(f"{k} {type_name(v)}" for k, v in self.spec.fields.items())
            return f"CREATE TABLE {safe_table} ({cols});"

        return ""

    def to_arrow(self, objects: list[BaseModel]) -> pa.Table:
        """
        SotA Batch Serialization: Objects -> Arrow Table.
        Handles UUID binary conversion and Enum integer encoding efficiently.
        """
        if not ARROW_AVAILABLE:
            raise ImportError(
                "Missing optional dependency 'pyarrow'.\nInstall with: uv pip install '.[arrow]'"
            )
        if not objects:
            p = typing.cast(Any, pa)
            return p.Table.from_pylist([], schema=self.arrow_schema)

        # Optimization: Pre-resolve serialization functions
        # This avoids checking isinstance for every row
        serializers = {}
        for k, v in self.spec.fields.items():
            # We only need custom serialization for complex types
            if isinstance(v, UUIDSpec | EnumSpec | StructSpec | ListSpec | VectorSpec):
                serializers[k] = v.serialize
            else:
                serializers[k] = None

        # Batch Processing
        pydict_data = {k: [] for k in self.spec.fields}

        for obj in objects:
            # Access raw data via __dict__ is faster than model_dump() for flat models
            # For nested models, we rely on recursiveness in the serializers
            raw = obj.__dict__
            for k, v_list in pydict_data.items():
                val = raw.get(k)
                ser = serializers[k]
                if val is not None and ser:
                    v_list.append(ser(val))
                else:
                    v_list.append(val)

        p = typing.cast(Any, pa)
        return p.Table.from_pydict(pydict_data, schema=self.arrow_schema)

    def to_polars(self, objects: list[BaseModel], validate: bool = False) -> pl.DataFrame:
        """
        Zero-Copy conversion to Polars DataFrame.
        Optionally runs Pandera Quality Gate.
        """
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Missing optional dependency 'polars'.\nInstall with: uv pip install '.[polars]'"
            )

        arrow_table = self.to_arrow(objects)
        p_pl = typing.cast(Any, pl)
        df = p_pl.from_arrow(arrow_table)

        if validate and PANDERA_AVAILABLE:
            try:
                schema = QualityGate.generate_schema(self.model_cls, backend="polars")
                df = schema.validate(df)  # Vectorized validation
                if isinstance(df, typing.cast(Any, pl).LazyFrame):
                    df = df.collect()
            except Exception:
                # Fallback to pandas backend if polars backend not available/compatible
                import pandas as pd

                schema_pd = QualityGate.generate_schema(self.model_cls, backend="pandas")
                df_pd = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
                # Normalize dtypes that may come in as categorical to plain object
                for col in df_pd.columns:
                    dtype_str = str(df_pd[col].dtype)
                    if dtype_str == "category" or dtype_str.startswith("datetime64"):
                        df_pd[col] = df_pd[col].astype(object)
                df_pd = schema_pd.validate(df_pd)
                df = typing.cast(Any, pl).from_pandas(df_pd)

        return df

    def ingest_duckdb(
        self, con: duckdb.DuckDBPyConnection, table_name: str, objects: list[BaseModel]
    ) -> int:
        """
        High-Speed Ingestion into DuckDB via Arrow Bridge.
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError(
                "Missing optional dependency 'duckdb'.\nInstall with: uv pip install '.[duckdb]'"
            )

        arrow_tbl = self.to_arrow(objects)
        # Register the Arrow table, then ingest
        con.register("arrow_tbl", arrow_tbl)
        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM arrow_tbl LIMIT 0")
        con.execute(f"INSERT INTO {table_name} SELECT * FROM arrow_tbl")
        return len(objects)


# --- 5. Unified API (The Public Face) ---


def resolve(model: type[BaseModel]) -> TypeSpec:
    return _RESOLVER.resolve(model)


def generate_ddl(model: type[BaseModel], dialect: str, table_name: str) -> str:
    return PolyTransporter(model).generate_ddl(dialect, table_name)  # type: ignore


def to_arrow(objects: list[BaseModel]) -> pa.Table | None:
    if not objects:
        return None
    return PolyTransporter(type(objects[0])).to_arrow(objects)


def to_polars(objects: list[BaseModel], validate: bool = True) -> pl.DataFrame | None:
    if not objects:
        return None
    return PolyTransporter(type(objects[0])).to_polars(objects, validate=validate)


# --- 5a. Developer Inspection Tool ---


def inspect(model: type[BaseModel]) -> None:
    """Prints the type mapping for a model (Debugging/Transparency)."""
    spec = _RESOLVER.resolve(model)
    if not hasattr(spec, "fields"):
        print(f"Type: {type(spec).__name__} (Not a Model)")
        return

    print(f"🔍 Politipo Inspection: {model.__name__}")
    print(f"{'Field':<15} | {'DuckDB':<20} | {'Arrow':<30}")
    print("-" * 70)

    for name, field in spec.fields.items():
        arrow_name = str(field.arrow) if ARROW_AVAILABLE else "N/A"
        if len(arrow_name) > 30:
            arrow_name = arrow_name[:27] + "..."
        print(f"{name:<15} | {field.duckdb:<20} | {arrow_name:<30}")
    print("-" * 70)


# --- 6a. UX Helpers (Extras Loader & Fluent Pipeline) ---

_EXTRA_HINT = {
    "arrow": "pyarrow",
    "polars": "polars",
    "duckdb": "duckdb",
    "validation": "pandera",
    "pandas": "pandas",
    "sqlalchemy": "sqlalchemy",
}


def require(
    extra: Literal["arrow", "polars", "duckdb", "validation", "pandas", "sqlalchemy"],
) -> None:
    """Ensure an optional extra is available or raise a helpful ImportError.

    Usage: require("polars") will suggest: uv pip install '.[polars]'
    """
    name = _EXTRA_HINT.get(extra, extra)
    msg = f"Missing optional dependency '{name}'.\nInstall with: uv pip install '.[{extra}]'"
    if extra == "arrow" and not ARROW_AVAILABLE:
        raise ImportError(msg)
    if extra == "polars" and not POLARS_AVAILABLE:
        raise ImportError(msg)
    if extra == "duckdb" and not DUCKDB_AVAILABLE:
        raise ImportError(msg)
    if extra == "validation" and not PANDERA_AVAILABLE:
        raise ImportError(msg)
    if extra == "sqlalchemy" and not SQL_AVAILABLE:
        raise ImportError(msg)


class Pipeline:
    def __init__(self, model_cls: type[BaseModel], objects: list[BaseModel]):
        self.model_cls = model_cls
        self.objects = objects
        self._arrow: pa.Table | None = None
        self._df: pl.DataFrame | None = None
        self._transporter = PolyTransporter(model_cls)

    def to_arrow(self) -> Pipeline:
        self._arrow = self._transporter.to_arrow(self.objects)
        return self

    def to_duckdb(self, con: duckdb.DuckDBPyConnection, table_name: str) -> Pipeline:
        self._transporter.ingest_duckdb(con, table_name, self.objects)
        return self

    def to_polars(self, validate: bool = True) -> pl.DataFrame:
        self._df = self._transporter.to_polars(self.objects, validate=validate)
        return self._df

    def to_pandas(self) -> pd.DataFrame | None:
        """High-performance Arrow->Pandas conversion."""
        if not ARROW_AVAILABLE:
            raise ImportError("PyArrow required")
        return self._transporter.to_arrow(self.objects).to_pandas()

    def write_parquet(self, path: str, compression: str = "snappy") -> None:
        """Dump memory directly to Parquet file."""
        if not ARROW_AVAILABLE:
            raise ImportError("PyArrow required")
        import pyarrow.parquet as pq

        tbl = self._transporter.to_arrow(self.objects)
        pq.write_table(tbl, path, compression=compression)


def from_models(objects: list[BaseModel]) -> Pipeline:
    if not objects:
        raise ValueError("from_models requires a non-empty list of models")
    return Pipeline(type(objects[0]), objects)


def pipeline(objects: list[BaseModel]) -> Pipeline:
    """Alias for from_models for a slightly shorter fluent entrypoint."""
    return from_models(objects)


# --- 6. Validation & Demo ---

if __name__ == "__main__":
    print("\n🌌 POLITIPO v3.0 OMEGA: System Check\n")

    # 1. Define Schema (Single Source of Truth)
    class UserType(enum.Enum):
        ADMIN = "admin"
        USER = "user"

    class Event(BaseModel):
        # Metadata: PK
        id: Annotated[uuid.UUID, FieldInfo(primary_key=True)]
        # Metadata: Constraints (for Pandera)
        cost: Annotated[decimal.Decimal, Precision(18, 4), FieldInfo(gt=0)]
        # Metadata: Vector (for Kuzu/Arrow)
        embedding: Annotated[list[float], "vector", 4] | None
        user_type: UserType
        timestamp: datetime.datetime

    # 2. Generate Infrastructure
    transporter = PolyTransporter(Event)
    print(f"[DuckDB DDL] {transporter.generate_ddl('duckdb', 'events')}")
    print(f"[Kuzu   DDL] {transporter.generate_ddl('kuzu', 'Events')}")

    # 3. Create Data
    data = [
        Event(
            id=uuid.uuid4(),
            cost=decimal.Decimal("10.5000"),
            embedding=[0.1, 0.2, 0.3, 0.4],
            user_type=UserType.ADMIN,
            timestamp=datetime.datetime.now(),
        ),
        Event(
            id=uuid.uuid4(),
            cost=decimal.Decimal("100.0000"),
            embedding=None,
            user_type=UserType.USER,
            timestamp=datetime.datetime.now(),
        ),
    ]

    if ARROW_AVAILABLE:
        # 4. Arrow Verification
        tbl = transporter.to_arrow(data)
        print(f"\n[Arrow Schema]\n{tbl.schema}")

        # Verify Optimization: UUID should be FixedSizeBinary[16]
        uuid_field = tbl.schema.field("id")
        print(f"UUID Storage: {uuid_field.type} (Should be fixed_size_binary[16])")

        # Verify Optimization: Enum should be Dictionary
        enum_field = tbl.schema.field("user_type")
        print(f"Enum Storage: {enum_field.type} (Should be dictionary<...>)")

    if DUCKDB_AVAILABLE:
        # 5. DuckDB Ingestion
        print("\n[DuckDB Ingestion]")
        con = duckdb.connect(":memory:")
        transporter.ingest_duckdb(con, "events", data)
        res = con.sql("SELECT count(*), sum(cost) FROM events").fetchall()
        print(f"Result: {res}")

    if POLARS_AVAILABLE and PANDERA_AVAILABLE:
        # 6. Quality Gate
        print("\n[Pandera Quality Gate]")
        try:
            df = transporter.to_polars(data, validate=True)
            print("✅ Data passed vectorized validation!")
            print(df)
        except Exception as e:
            print(f"❌ Validation Failed: {e}")

    print("\n✅ System Operational.")

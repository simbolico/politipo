# politipo — PolyType Data Fabric

[![CI](https://github.com/kevinsaltarelli/politipo/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinsaltarelli/politipo/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-95%25%2B-brightgreen)

PolyType: Pydantic → Arrow → DuckDB/Polars with vectorized validation (Pandera). Define your schema once with Pydantic v2, then move and validate data across modern analytics stacks with zero-copy where possible.

Highlights
- Define once: Pydantic v2 + Annotated metadata
- Transport fast: uuid as fixed-size binary; enums as dictionary; vectors as FixedSizeList
- Validate vectorized: Pandera (Polars backend, robust Pandas fallback)
- Analyze anywhere: DuckDB + Polars; Kùzu DDL generation
- Lean installs: uv-only workflow; extras loaded on demand

Install

Use uv (recommended):

```
uv pip install politipo
```

Optional extras:
- Arrow/Parquet: `uv pip install 'politipo[arrow]'`
- Polars: `uv pip install 'politipo[polars]'`
- DuckDB: `uv pip install 'politipo[duckdb]'`
- Validation (Pandera): `uv pip install 'politipo[validation]'`
- Pandas fallback: `uv pip install 'politipo[pandas]'`
- SQLAlchemy types: `uv pip install 'politipo[sqlalchemy]'`
- Everything: `uv pip install 'politipo[all]'`

Quick Start

```python
from typing import Annotated
import uuid, datetime, decimal, enum
from pydantic import BaseModel, Field
import politipo as pt

class UserType(enum.Enum):
    ADMIN = "admin"
    USER = "user"

class Event(BaseModel):
    id: Annotated[uuid.UUID, pt.FieldInfo(primary_key=True)]  # PK
    cost: Annotated[decimal.Decimal, pt.Precision(18, 4), Field(gt=0)]  # constraints
    embedding: pt.Vector[4] | None  # vector type
    user_type: UserType  # enum
    timestamp: datetime.datetime

# Generate DDL
print(pt.generate_ddl(Event, "duckdb", "events"))
print(pt.generate_ddl(Event, "kuzu", "Events"))

# Create data
rows = [
    Event(
        id=uuid.uuid4(),
        cost=decimal.Decimal("10.5000"),
        embedding=[0.1, 0.2, 0.3, 0.4],
        user_type=UserType.ADMIN,
        timestamp=datetime.datetime.now(),
    )
]

# Convert to Arrow (install extra: 'arrow')
tbl = pt.to_arrow(rows)  # pyarrow.Table
print(tbl.schema)

# Convert to Polars with validation (install extras: 'polars', 'validation', 'pandas')
df = pt.to_polars(rows, validate=True)
print(df)
```

Why SotA

- Single source of truth: Pydantic + Annotated drives Arrow/DuckDB/Polars/Pandera
- Memory-aware storage: uuid binary(16), enums dictionary encoding, vectors as FixedSizeList
- Graceful DX: missing extras raise actionable “uv pip install '.[extra]'” hints
- UV-native: lock, sync, run, build, publish

API Overview

- Precision: `Annotated[Decimal, pt.Precision(p, s)]` → Arrow decimal128, DuckDB DECIMAL(p,s)
- Vector[N]: `pt.Vector[N]` → Arrow FixedSizeList(float32, N); DuckDB FLOAT[N]
- Resolver → Specs: Maps model fields to TypeSpecs (String/Int/Float/Bool/UUID/Decimal/List/Struct/Enum)
- PolyTransporter:
  - `to_arrow(models)` → `pyarrow.Table`
  - `to_polars(models, validate=True)` → `polars.DataFrame`
  - `ingest_duckdb(con, table, models)` → insert via Arrow
  - `generate_ddl(dialect, table)` → DuckDB/Kùzu/portable SQL
- QualityGate:
  - `generate_schema(model, backend='polars'|'pandas')` → Pandera schema
  - Extracts gt/ge/lt/le/pattern, min_length/max_length, multiple_of
- Pipeline (fluent):
  - `pt.from_models(rows).to_arrow().to_duckdb(con, 'tbl').to_polars(validate=True)`

Recipes

- Financial decimals: `Precision(38, 18)` for high-precision columns; DuckDB DECIMAL(38,18).
- Embeddings: `Vector[1536]` stored as FixedSizeList(float32, 1536); Arrow-friendly for RAG.
- Enums as dictionary: memory-efficient encoded storage via Arrow dictionary type.
- Kùzu DDL: `generate_ddl('kuzu', 'Node')` produces STRUCT-based Node table schema with PK.

Missing-Extras UX

If a dependency is missing, you’ll see a helpful error:

```
Missing optional dependency 'pyarrow'.
Install with: uv pip install '.[arrow]'
```

Benchmarks (excerpt)

- UUID binary vs string UUIDs: smaller footprint, faster scans.
- Enums as dictionary vs strings: reduced RAM.
- Arrow → DuckDB ingest: high throughput via Arrow bridge.

Run local benchmarks with uv, then compare using your data shape. (Optional scripts incoming.)

Onboarding: Step-by-Step

- Create a venv and sync dev tools:
  - `uv python pin 3.13 && uv sync --group dev`
- Install extras you need:
  - Arrow: `uv pip install '.[arrow]'`
  - Polars: `uv pip install '.[polars]'`
  - DuckDB: `uv pip install '.[duckdb]'`
  - Validation: `uv pip install '.[validation]'` (+ `.[pandas]` for fallback)
- Define a Pydantic model with Annotated metadata (Precision, Field constraints, Vector[N]).
- Generate DDL and create a table (DuckDB/Kùzu).
- Use `pt.pipeline(rows)` to to_arrow().to_duckdb().to_polars(validate=True).
- Missing dependency? Error messages tell exactly which extra to install.

Development (uv-only)

- Lint/format/type: `make lint`, `make fmt-check`, `make type`
- Tests: `make test` (src/), coverage: `make cov`
- Pre-commit: `make pre-commit`

Releases

- Semantic versioning
- Build with uv: `make build`
- Publish with uv (requires `UV_PUBLISH_TOKEN`): `make publish`
- Release flow: `make release` then push tags: `git push --follow-tags` (publishing can be automated in CI)
user_sqlmodel = converter.convert({"id": 1, "name": "Alice"})
```

#### Converting to SQLAlchemy Models
Convert data to SQLAlchemy models (requires the `sqlalchemy` library):

```python
converter = TypeConverter(from_type=dict, to_type=UserSQLAlchemy)
user_sqlalchemy = converter.convert({"id": 1, "name": "Alice"})
```

#### Converting to DataFrames
Convert lists of `dict`, Pydantic models, or SQLModel instances to Pandas or Polars DataFrames:

```python
import pandas as pd
converter = TypeConverter(from_type=User, to_type=pd.DataFrame)
df = converter.convert([user1, user2])  # user1, user2 are User instances
```

#### Collection Conversions
Convert DataFrames to lists of models:

```python
converter = TypeConverter(from_type=pd.DataFrame, to_type=User)
users = converter.convert(df)  # df is a Pandas DataFrame
```

### TypeMapper Usage

#### Basic Type Mapping
To map a type from one library to another, create a `TypeMapper` instance and use the `map_type` method:

```python
mapper = TypeMapper()
# Auto-detect the source library (if possible)
target_type = mapper.map_type(source_type, to_library='target_library')

# Or explicitly specify the source library
target_type = mapper.map_type(source_type, to_library='target_library', from_library='source_library')
```

#### Mapping Python Types
Map between Python built-in types and other libraries:

```python
# Map Python int to SQLAlchemy Integer (auto-detection)
sqlalchemy_type = mapper.map_type(int, to_library='sqlalchemy')
# Import needed only when using the result
from sqlalchemy import Integer
assert sqlalchemy_type is Integer
```

#### Mapping SQLAlchemy Types
Map SQLAlchemy types to other libraries:

```python
from sqlalchemy import String
# Map SQLAlchemy String to Polars Utf8 (auto-detection)
polars_type = mapper.map_type(String, to_library='polars')
# Import needed only when using the result
import polars as pl
assert polars_type is pl.Utf8
```

#### Mapping Pandas Types
Map Pandas types to other libraries:

```python
import pandas as pd
# Map Pandas StringDtype to Python str (auto-detection)
python_type = mapper.map_type(pd.StringDtype(), to_library='python')
assert python_type is str
```

#### Mapping Polars Types
Map Polars types to other libraries:

```python
import polars as pl
# Map Polars Boolean to SQLAlchemy Boolean (auto-detection)
sqlalchemy_type = mapper.map_type(pl.Boolean, to_library='sqlalchemy')
# Import needed only when using the result
from sqlalchemy import Boolean
assert sqlalchemy_type is Boolean
```

#### Getting Canonical Types
Access the canonical type representation directly:

```python
# Get canonical type for Python int
canonical = mapper.get_canonical_type(int, 'python')  # Returns 'integer'

# Get library type from canonical
python_type = mapper.get_library_type('integer', 'python')  # Returns int
```

#### Automatic Library Detection
The `TypeMapper` class now supports automatic detection of the source library for type objects. This makes it easier to map types between libraries without needing to explicitly specify the source library.

### Using Library Auto-detection

```python
from politipo import TypeMapper
from sqlalchemy import Integer
import polars as pl

mapper = TypeMapper()

# Auto-detect sqlalchemy as the source library
pl_type = mapper.map_type(Integer, to_library='polars')
assert pl_type is pl.Int64

# Auto-detect python as the source library
sqlalchemy_type = mapper.map_type(int, to_library='sqlalchemy')
assert sqlalchemy_type is Integer
```

The `detect_library` method identifies the library a type belongs to by checking:
- Built-in Python types
- Type annotations from typing module
- Module paths and attributes
- Library-specific type patterns

When automatic detection is used, you only need to specify the target library. If detection fails, a `ValueError` is raised with a helpful message.

---

## API Reference

### `TypeConverter` Class
The core class for performing type conversions.

#### `__init__(self, from_type: Type, to_type: Type)`
Initializes the converter with the source and target types.

- **Parameters**:
  - `from_type`: The type of the input data (e.g., `dict`, `User`, `pd.DataFrame`).
  - `to_type`: The desired output type (e.g., `User`, `dict`, `pd.DataFrame`).

#### `convert(self, data: Any) -> Any`
Converts the input data from the source type to the target type.

- **Parameters**:
  - `data`: The data to convert (e.g., a dictionary, model instance, or dataframe).
- **Returns**: The converted data in the target type.
- **Raises**:
  - `ValueError`: If the conversion is unsupported or the data is invalid.
  - `ImportError`: If a required dependency is missing.

#### `convert_single(self, data: Any, coerce: bool = False) -> Any`
Converts a single item to the target type. This is an enhanced method for cases where you specifically want to convert an individual item rather than a collection.

- **Parameters**:
  - `data`: The data to convert (e.g., a dictionary or model instance).
  - `coerce`: If True, allows type coercion for stricter types like Pydantic models.
- **Returns**: The converted data in the target type.
- **Raises**:
  - `ValueError`: If the conversion is unsupported or the data is invalid.
  - `ImportError`: If a required dependency is missing.

#### `convert_collection(self, data: List[Any], coerce: bool = False) -> Any`
Converts a collection of items to the target type. This method is specialized for handling collections and provides better error messages for collection-specific conversions.

- **Parameters**:
  - `data`: The list of data to convert.
  - `coerce`: If True, allows type coercion for stricter types like Pydantic models.
- **Returns**: The converted collection in the target type.
- **Raises**:
  - `TypeError`: If data is not a list (except for DataFrames).
  - `ValueError`: If the conversion is unsupported or the data is invalid.
  - `ImportError`: If a required dependency is missing.

### `TypeMapper` Class
The core class for mapping types between different libraries.

#### `__init__(self)`
Initializes the mapper with supported library mappings.

#### `get_canonical_type(self, type_obj: Any, library: str) -> str`
Converts a library-specific type to a canonical type representation.

- **Parameters**:
  - `type_obj`: The type object to convert (e.g., `int`, `sqlalchemy.Integer`).
  - `library`: The source library name (e.g., 'python', 'sqlalchemy').
- **Returns**: A string representing the canonical type (e.g., 'integer', 'string').
- **Raises**:
  - `ValueError`: If the type or library is unsupported.
  - `ImportError`: If a required dependency is missing.

#### `get_library_type(self, canonical_type: str, library: str) -> Any`
Converts a canonical type to a library-specific type.

- **Parameters**:
  - `canonical_type`: The canonical type string (e.g., 'integer', 'string').
  - `library`: The target library name (e.g., 'python', 'sqlalchemy').
- **Returns**: A type object from the specified library (e.g., `int`, `sqlalchemy.Integer`).
- **Raises**:
  - `ValueError`: If the canonical type or library is unsupported.
  - `ImportError`: If a required dependency is missing.

#### `map_type(self, type_obj: Any, to_library: str, from_library: Optional[str] = None) -> Any`
Maps a type from one library to another via the canonical type system.

- **Parameters**:
  - `type_obj`: The type object to map (e.g., `int`, `sqlalchemy.Integer`).
  - `to_library`: The target library name (e.g., 'python', 'pandas').
  - `from_library`: The source library name (e.g., 'python', 'sqlalchemy'). If None, the library will be auto-detected.
- **Returns**: A type object from the target library.
- **Raises**:
  - `ValueError`: If the type or library is unsupported or auto-detection fails.
  - `ImportError`: If a required dependency is missing.

#### `detect_library(self, type_obj: Any) -> str`
Automatically detects which library a type object belongs to.

- **Parameters**:
  - `type_obj`: The type object to detect the library for (e.g., `int`, `sqlalchemy.Integer`).
- **Returns**: A string representing the detected library (e.g., 'python', 'sqlalchemy', 'pandas', 'polars').
- **Raises**:
  - `ValueError`: If the library cannot be detected.

#### `_convert_nested(self, data: Any, target_type: Type) -> Any`
Recursively converts nested data structures like dictionaries and lists while preserving their structure.

- **Parameters**:
  - `data`: The nested data structure to convert.
  - `target_type`: The target type for individual elements.
- **Returns**: The converted nested structure with the same hierarchy.

---

## Error Handling

The library provides custom exceptions for better error handling:

- `TypeConversionError`: Base exception for all conversion errors (subclasses `ValueError`).
- `UnsupportedTypeError`: Raised when a conversion is not supported.
- `MissingLibraryError`: Raised when a required library is not installed (subclasses `ImportError`).

These can be caught in your code:

```python
from politipo.type_converter import TypeConversionError, MissingLibraryError

try:
    result = converter.convert(data)
except TypeConversionError as e:
    print(f"Conversion error: {e}")
except MissingLibraryError as e:
    print(f"Missing dependency: {e}")
```

## Best Practices

- **Use Type Hints**: Always specify `from_type` and `to_type` with type hints to ensure clarity and leverage runtime type checking.
- **Handle Errors**: Wrap conversions in try-except blocks to catch errors appropriately:
  ```python
  try:
      result = converter.convert(data)
  except ValueError as e:
      print(f"Conversion error: {e}")
  except ImportError as e:
      print(f"Missing dependency: {e}")
  ```
- **Use Specialized Methods**: For clearer code intent, use `convert_single()` for individual items and `convert_collection()` for collections.
- **Process Nested Structures**: Use `_convert_nested()` when dealing with complex nested data structures that need to maintain their hierarchy.

---

## Troubleshooting

- **Missing Library Errors**: If you encounter an `ImportError`, install the required library (e.g., `pip install pydantic` for Pydantic models).
- **Unsupported Type Errors**: Ensure both `from_type` and `to_type` are supported by `politipo`. Check the Features section for supported types.
- **Data Validation Errors**: Verify that the input data matches the structure expected by the target type (e.g., required fields are present).
- **Type Mapping Errors**: If a type mapping fails, check that both the source and target libraries are supported and that the specific type has a defined mapping.

---

## Supported Type Mappings

The following table outlines the canonical type mappings supported by `TypeMapper`:

| Canonical Type | Python   | SQLAlchemy | Pandas        | Polars   |
|----------------|----------|------------|---------------|----------|
| 'integer'      | int      | Integer    | Int64Dtype()  | Int64    |
| 'string'       | str      | String     | StringDtype() | Utf8     |
| 'float'        | float    | Float      | 'float64'     | Float64  |
| 'boolean'      | bool     | Boolean    | 'bool'        | Boolean  |

Note that:
- Each mapping only requires the respective library to be installed when actually used
- Type mapping is bidirectional (e.g., you can map from 'python' to 'sqlalchemy' and vice versa)
- NoneType is not supported directly; use Optional[T] for nullable types in Python typing

---

## Conversion Table

Below is an updated Markdown table that includes all available combinations of `from_type` to `to_type` supported by the `politipo` library's `TypeConverter` class, with two additional columns: "Example From" and "Example To". These columns provide concrete examples of the input (`from_type`) and output (`to_type`) for each conversion, based on the sample data from `test_politipo.py`.

---

| From Type             | To Type               | Description                                                                                  | Supported | Example From                                      | Example To                                        |
|-----------------------|-----------------------|----------------------------------------------------------------------------------------------|-----------|--------------------------------------------------|--------------------------------------------------|
| `dict`                | `dict`                | Returns the input dictionary unchanged                                                       | ✅        | `{"id": 1, "name": "Alice"}`                     | `{"id": 1, "name": "Alice"}`                     |
| `Pydantic`            | `dict`                | Converts a Pydantic model instance to a dictionary using `.model_dump()`                     | ✅        | `UserPydantic(id=1, name="Alice")`               | `{"id": 1, "name": "Alice"}`                     |
| `SQLModel`            | `dict`                | Converts an SQLModel instance to a dictionary using `.model_dump()`                          | ✅        | `UserSQLModel(id=1, name="Alice")`               | `{"id": 1, "name": "Alice"}`                     |
| `SQLAlchemy`          | `dict`                | Extracts attributes from an SQLAlchemy model into a dictionary                               | ✅        | `UserSQLAlchemy(id=1, name="Alice")`             | `{"id": 1, "name": "Alice"}`                     |
| `dict`                | `Pydantic`            | Validates a dictionary into a Pydantic model using `.model_validate()`                       | ✅        | `{"id": 1, "name": "Alice"}`                     | `UserPydantic(id=1, name="Alice")`               |
| `Pydantic`            | `Pydantic`            | Converts one Pydantic model to another by dumping and validating                             | ✅        | `UserPydantic(id=1, name="Alice")`               | `UserPydantic(id=1, name="Alice")`               |
| `SQLModel`            | `Pydantic`            | Converts an SQLModel instance to a Pydantic model via dictionary                             | ✅        | `UserSQLModel(id=1, name="Alice")`               | `UserPydantic(id=1, name="Alice")`               |
| `SQLAlchemy`          | `Pydantic`            | Converts an SQLAlchemy model to a Pydantic model via dictionary                              | ✅        | `UserSQLAlchemy(id=1, name="Alice")`             | `UserPydantic(id=1, name="Alice")`               |
| `dict`                | `SQLModel`            | Validates a dictionary into an SQLModel instance using `.model_validate()`                   | ✅        | `{"id": 1, "name": "Alice"}`                     | `UserSQLModel(id=1, name="Alice")`               |
| `Pydantic`            | `SQLModel`            | Converts a Pydantic model to an SQLModel instance via dictionary                             | ✅        | `UserPydantic(id=1, name="Alice")`               | `UserSQLModel(id=1, name="Alice")`               |
| `SQLModel`            | `SQLModel`            | Returns the input SQLModel instance unchanged if types match                                 | ✅        | `UserSQLModel(id=1, name="Alice")`               | `UserSQLModel(id=1, name="Alice")`               |
| `SQLAlchemy`          | `SQLModel`            | Converts an SQLAlchemy model to an SQLModel instance via dictionary                          | ✅        | `UserSQLAlchemy(id=1, name="Alice")`             | `UserSQLModel(id=1, name="Alice")`               |
| `dict`                | `SQLAlchemy`          | Creates an SQLAlchemy model instance from a dictionary using keyword arguments               | ✅        | `{"id": 1, "name": "Alice"}`                     | `UserSQLAlchemy(id=1, name="Alice")`             |
| `Pydantic`            | `SQLAlchemy`          | Converts a Pydantic model to an SQLAlchemy model via dictionary                              | ✅        | `UserPydantic(id=1, name="Alice")`               | `UserSQLAlchemy(id=1, name="Alice")`             |
| `SQLModel`            | `SQLAlchemy`          | Converts an SQLModel instance to an SQLAlchemy model via dictionary                          | ✅        | `UserSQLModel(id=1, name="Alice")`               | `UserSQLAlchemy(id=1, name="Alice")`             |
| `SQLAlchemy`          | `SQLAlchemy`          | Returns the input SQLAlchemy instance unchanged if types match                               | ✅        | `UserSQLAlchemy(id=1, name="Alice")`             | `UserSQLAlchemy(id=1, name="Alice")`             |
| `list[dict]`          | `pd.DataFrame`        | Creates a Pandas DataFrame from a list of dictionaries                                       | ✅        | `[ {"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"} ]` | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[Pydantic]`      | `pd.DataFrame`        | Creates a Pandas DataFrame from a list of Pydantic models                                    | ✅        | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[SQLModel]`      | `pd.DataFrame`        | Creates a Pandas DataFrame from a list of SQLModel instances                                 | ✅        | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[SQLAlchemy]`    | `pd.DataFrame`        | Converts a list of SQLAlchemy model instances to a Pandas DataFrame                          | ✅        | `[UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]` | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[dict]`          | `pl.DataFrame`        | Creates a Polars DataFrame from a list of dictionaries                                       | ✅        | `[ {"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"} ]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[Pydantic]`      | `pl.DataFrame`        | Creates a Polars DataFrame from a list of Pydantic models                                    | ✅        | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[SQLModel]`      | `pl.DataFrame`        | Creates a Polars DataFrame from a list of SQLModel instances                                 | ✅        | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[SQLAlchemy]`    | `pl.DataFrame`        | Converts a list of SQLAlchemy model instances to a Polars DataFrame                          | ✅        | `[UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `pd.DataFrame`        | `list[Pydantic]`      | Converts a Pandas DataFrame to a list of Pydantic models                                     | ✅        | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` |
| `pd.DataFrame`        | `list[SQLModel]`      | Converts a Pandas DataFrame to a list of SQLModel instances                                  | ✅        | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` |
| `pd.DataFrame`        | `list[SQLAlchemy]`    | Converts a Pandas DataFrame to a list of SQLAlchemy model instances                          | ✅        | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]` |
| `pl.DataFrame`        | `list[Pydantic]`      | Converts a Polars DataFrame to a list of Pydantic models                                     | ✅        | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` |
| `pl.DataFrame`        | `list[SQLModel]`      | Converts a Polars DataFrame to a list of SQLModel instances                                  | ✅        | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` |
| `pl.DataFrame`        | `list[SQLAlchemy]`    | Converts a Polars DataFrame to a list of SQLAlchemy model instances                          | ✅        | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]` |
| `str`                 | `dict`                | Not supported; raises ValueError with "Unsupported from_type" message                        | ❌        | `"invalid"`                                      | N/A                                              |
| `dict`                | `list`                | Not supported; raises ValueError with "Unsupported to_type" message                          | ❌        | `{"id": 1, "name": "Alice"}`                     | N/A                                              |


---

### Notes on the Table

1. **Type Definitions**:
   - `Pydantic`: Instances of `UserPydantic` (inherits from `pydantic.BaseModel`).
   - `SQLModel`: Instances of `UserSQLModel` (inherits from `sqlmodel.SQLModel`).
   - `SQLAlchemy`: Instances of `UserSQLAlchemy` (inherits from `sqlalchemy.orm.declarative_base()`).
   - `pd.DataFrame`: Pandas DataFrame (`pandas.DataFrame`).
   - `pl.DataFrame`: Polars DataFrame (`polars.DataFrame`).

2. **Examples**:
   - **From**: The "Example From" column uses the exact sample data from `test_type_converter.py` where applicable (e.g., `sample_dict`, `sample_pydantic`).
   - **To**: The "Example To" column shows the expected output, simplified for readability. For DataFrames, the output is shown as the constructor call that matches the content, though actual instances would have additional metadata (e.g., column types).
   - **N/A**: For unsupported conversions, no example output is provided as they raise exceptions.

3. **Supported Combinations**:
   - The ✅ entries are fully implemented and tested, with examples drawn from the test suite.
   - For DataFrame-related conversions, the `from_type` involves lists or DataFrames, and the `to_type` may result in lists, reflecting the library's handling of collections.

4. **Unsupported Combinations**:
   - The ❌ entries show examples of inputs that would fail, with "N/A" for outputs since they aren't produced.
   - Unsupported conversions will raise a specific ValueError with a descriptive message explaining the issue.
   - The error messages clearly indicate whether the source type (`from_type`) or target type (`to_type`) is unsupported.

---

## Development

This project uses uv for dependency management and ruff/black/ty for quality.

### Setup

1) Install uv (see https://docs.astral.sh/uv/)
2) Sync dev dependencies:

```
uv sync --group dev
```

### Common tasks

```
# Lint
uv run ruff check .

# Auto-fix lint
uv run ruff check . --fix

# Format
uv run black .

# Format check
uv run black --check .

# Type-check (ty)
uv run ty

# Run tests against src/
PYTHONPATH=src uv run pytest -q src
```

## Repository Layout

- New library: `src/politipo.py` with tests under `src/`.
- Legacy code and examples: `legacy/` (excluded from lint/CI packaging scope).

## Install With uv Only

- Base install (minimal deps):
```
uv pip install .
```

- Install extras only when needed:
```
# Arrow/Parquet
uv pip install '.[arrow]'

# Polars
uv pip install '.[polars]'

# DuckDB
uv pip install '.[duckdb]'

# Validation (Pandera) and optional Pandas fallback
uv pip install '.[validation]'
uv pip install '.[pandas]'

# Everything
uv pip install '.[all]'
```

Or in a synced dev environment:
```
uv sync --group dev
```

Note: This project uses uv exclusively; pip/poetry are not used in CI or docs.

### Pre-commit

Install and run local hooks:

```
uv run pre-commit install
uv run pre-commit run --all-files
```

### CI

GitHub Actions (Python 3.14) runs: ruff, black --check, ty, and pytest with `PYTHONPATH=src`.

---

## Makefile

For convenience, common tasks are available via `make`:

```
# One-time: install dev deps (uses a PyO3 3.14 compatibility env var)
make sync

# Quality
make lint
make fix
make fmt
make fmt-check
make type

# Tests (targets src/ by default)
make test

# Pre-commit hooks
make pre-commit
```

Note: On Python 3.14, pydantic-core builds via PyO3 may need the
`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` environment variable. The Makefile and CI
set this automatically. If you still encounter build issues on 3.14, consider
using Python 3.13 temporarily until upstream packages fully support 3.14.

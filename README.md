# `politipo` Library Documentation

Welcome to the official documentation for the `politipo` library! This guide provides a comprehensive overview of the library's functionality, usage, and best practices. Whether you're a beginner looking to get started or an advanced user seeking detailed API references, this documentation has you covered.

Current version: 0.1.3 - Featuring automatic library detection for simplified type mapping.

---

## Introduction

`politipo` is a versatile Python library designed to simplify type conversions between various data structures and models. It supports seamless conversions between built-in types (e.g., `dict`), Pydantic models, SQLModel instances, SQLAlchemy models, and dataframes from Pandas and Polars. The library's primary goal is to streamline data transformations, making it easier for developers to work with diverse data formats in their applications.

The library consists of two main components:
- **TypeConverter**: Handles data conversions between different types (e.g., converting a dictionary to a Pydantic model)
- **TypeMapper**: Maps type definitions between different libraries using a canonical type system (e.g., mapping Python's `int` to SQLAlchemy's `Integer`)

---

## Installation

To install `politipo`, use the following command in your terminal or command prompt:

```bash
pip install politipo
```

### Requirements
- Python 3.11 or higher
- Additional dependencies (e.g., `pydantic`, `sqlmodel`, `sqlalchemy`, `pandas`, `polars`) may be required depending on the conversions you perform. These will be dynamically imported as needed.

---

## Quick Start

### TypeConverter Example

Here's a simple example to get you up and running with `TypeConverter`:

```python
from politipo import TypeConverter
from pydantic import BaseModel

# Define a Pydantic model
class User(BaseModel):
    id: int
    name: str

# Create a converter and perform a conversion
converter = TypeConverter(from_type=dict, to_type=User)
user = converter.convert({"id": 1, "name": "Alice"})
print(user)  # Output: User(id=1, name='Alice')
```

This example converts a dictionary to a Pydantic model, demonstrating the library's core functionality.

### TypeMapper Example

Here's a simple example showing how to use `TypeMapper` to map types between different libraries:

```python
from politipo import TypeMapper
import polars as pl
from sqlalchemy import Integer

# Create a mapper
mapper = TypeMapper()

# Map SQLAlchemy Integer to Polars Int64
polars_type = mapper.map_type(Integer, to_library='polars')  # Auto-detects SQLAlchemy
print(polars_type)  # Output: Int64

# Map Python int to Pandas Int64Dtype
pandas_type = mapper.map_type(int, to_library='pandas')  # Auto-detects Python
print(pandas_type)  # Output: Int64Dtype()
```

This example shows how to map type definitions between SQLAlchemy, Polars, and Pandas libraries.

---

## Features

`politipo` offers a rich set of features to handle type conversions effectively:

- **Bidirectional Conversions**: Convert between supported types in both directions.
- **Type Safety**: Leverages Python type hints for runtime validation.
- **Collection Handling**: Supports conversions for single items and collections (e.g., lists, dataframes).
- **Dynamic Imports**: Only requires libraries necessary for the specific conversion, reducing dependency overhead.
- **Error Handling**: Provides clear error messages for unsupported types or missing dependencies.
- **Extensibility**: Allows advanced users to customize conversion logic.
- **Type Mapping**: Map type definitions between different libraries through a canonical type system.
- **Minimal Dependencies**: Each component only requires the libraries necessary for the specific operation.

---

## Usage

This section provides in-depth explanations and examples of how to use `politipo` for various conversions and type mappings.

### TypeConverter Usage

#### Basic Conversions
To perform a conversion, create a `TypeConverter` instance with the source (`from_type`) and target (`to_type`) types, then call the `convert` method:

```python
converter = TypeConverter(from_type=SourceType, to_type=TargetType)
result = converter.convert(data)
```

#### Converting to `dict`
Convert various types (e.g., Pydantic models, SQLModel instances) to a dictionary:

```python
converter = TypeConverter(from_type=User, to_type=dict)
user_dict = converter.convert(user_instance)  # user_instance is a User object
```

#### Converting to Pydantic Models
Convert from `dict`, other Pydantic models, SQLModel instances, SQLAlchemy models, or dataframes to a Pydantic model:

```python
converter = TypeConverter(from_type=dict, to_type=User)
user = converter.convert({"id": 1, "name": "Alice"})
```

#### Converting to SQLModel Instances
Convert data to SQLModel instances (requires the `sqlmodel` library):

```python
converter = TypeConverter(from_type=dict, to_type=UserSQLModel)
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
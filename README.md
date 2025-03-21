# `politipo` Library Documentation

Welcome to the official documentation for the `politipo` library! This guide provides a comprehensive overview of the library's functionality, usage, and best practices. Whether you're a beginner looking to get started or an advanced user seeking detailed API references, this documentation has you covered.

---

## Introduction

`politipo` is a versatile Python library designed to simplify type conversions between various data structures and models. It supports seamless conversions between built-in types (e.g., `dict`), Pydantic models, SQLModel instances, SQLAlchemy models, and dataframes from Pandas and Polars. The library's primary goal is to streamline data transformations, making it easier for developers to work with diverse data formats in their applications.

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

Here’s a simple example to get you up and running with `politipo`:

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

---

## Features

`politipo` offers a rich set of features to handle type conversions effectively:

- **Bidirectional Conversions**: Convert between supported types in both directions.
- **Type Safety**: Leverages Python type hints for runtime validation.
- **Collection Handling**: Supports conversions for single items and collections (e.g., lists, dataframes).
- **Dynamic Imports**: Only requires libraries necessary for the specific conversion, reducing dependency overhead.
- **Error Handling**: Provides clear error messages for unsupported types or missing dependencies.
- **Extensibility**: Allows advanced users to customize conversion logic.

---

## Usage

This section provides in-depth explanations and examples of how to use `politipo` for various conversions.

### Basic Conversions
To perform a conversion, create a `TypeConverter` instance with the source (`from_type`) and target (`to_type`) types, then call the `convert` method:

```python
converter = TypeConverter(from_type=SourceType, to_type=TargetType)
result = converter.convert(data)
```

### Converting to `dict`
Convert various types (e.g., Pydantic models, SQLModel instances) to a dictionary:

```python
converter = TypeConverter(from_type=User, to_type=dict)
user_dict = converter.convert(user_instance)  # user_instance is a User object
```

### Converting to Pydantic Models
Convert from `dict`, other Pydantic models, SQLModel instances, SQLAlchemy models, or dataframes to a Pydantic model:

```python
converter = TypeConverter(from_type=dict, to_type=User)
user = converter.convert({"id": 1, "name": "Alice"})
```

### Converting to SQLModel Instances
Convert data to SQLModel instances (requires the `sqlmodel` library):

```python
converter = TypeConverter(from_type=dict, to_type=UserSQLModel)
user_sqlmodel = converter.convert({"id": 1, "name": "Alice"})
```

### Converting to SQLAlchemy Models
Convert data to SQLAlchemy models (requires the `sqlalchemy` library):

```python
converter = TypeConverter(from_type=dict, to_type=UserSQLAlchemy)
user_sqlalchemy = converter.convert({"id": 1, "name": "Alice"})
```

### Converting to DataFrames
Convert lists of `dict`, Pydantic models, or SQLModel instances to Pandas or Polars DataFrames:

```python
import pandas as pd
converter = TypeConverter(from_type=User, to_type=pd.DataFrame)
df = converter.convert([user1, user2])  # user1, user2 are User instances
```

### Collection Conversions
Convert DataFrames to lists of models:

```python
converter = TypeConverter(from_type=pd.DataFrame, to_type=User)
users = converter.convert(df)  # df is a Pandas DataFrame
```

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

---

## Best Practices

- **Use Type Hints**: Always specify `from_type` and `to_type` with type hints to ensure clarity and leverage runtime type checking.
- **Handle Errors**: Wrap conversions in try-except blocks to catch `ValueError` (unsupported conversions) or `ImportError` (missing dependencies):
  ```python
  try:
      result = converter.convert(data)
  except ValueError as e:
      print(f"Conversion error: {e}")
  except ImportError as e:
      print(f"Missing dependency: {e}")
  ```
- **Optimize Performance**: For large datasets (e.g., dataframes), test conversion performance and consider batching if necessary.

---

## Troubleshooting

- **Missing Library Errors**: If you encounter an `ImportError`, install the required library (e.g., `pip install pydantic` for Pydantic models).
- **Unsupported Type Errors**: Ensure both `from_type` and `to_type` are supported by `politipo`. Check the Features section for supported types.
- **Data Validation Errors**: Verify that the input data matches the structure expected by the target type (e.g., required fields are present).

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
| `list[dict]`          | `pl.DataFrame`        | Creates a Polars DataFrame from a list of dictionaries                                       | ✅        | `[ {"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"} ]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[Pydantic]`      | `pl.DataFrame`        | Creates a Polars DataFrame from a list of Pydantic models                                    | ✅        | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `list[SQLModel]`      | `pl.DataFrame`        | Creates a Polars DataFrame from a list of SQLModel instances                                 | ✅        | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` |
| `pd.DataFrame`        | `list[Pydantic]`      | Converts a Pandas DataFrame to a list of Pydantic models                                     | ✅        | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` |
| `pd.DataFrame`        | `list[SQLModel]`      | Converts a Pandas DataFrame to a list of SQLModel instances                                  | ✅        | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` |
| `pl.DataFrame`        | `list[Pydantic]`      | Converts a Polars DataFrame to a list of Pydantic models                                     | ✅        | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]` |
| `pl.DataFrame`        | `list[SQLModel]`      | Converts a Polars DataFrame to a list of SQLModel instances                                  | ✅        | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | `[UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]` |
| `str`                 | `dict`                | Not supported; raises ValueError                                                             | ❌        | `"invalid"`                                      | N/A                                              |
| `dict`                | `list`                | Not supported; raises ValueError                                                             | ❌        | `{"id": 1, "name": "Alice"}`                     | N/A                                              |
| `list[SQLAlchemy]`    | `pd.DataFrame`        | Not explicitly supported; requires manual list-to-dict conversion                            | ❌        | `[UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]` | N/A                                              |
| `list[SQLAlchemy]`    | `pl.DataFrame`        | Not explicitly supported; requires manual list-to-dict conversion                            | ❌        | `[UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]` | N/A                                              |
| `pd.DataFrame`        | `list[SQLAlchemy]`    | Not explicitly supported; returns list of SQLAlchemy instances but not tested                | ❌        | `pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | N/A                                              |
| `pl.DataFrame`        | `list[SQLAlchemy]`    | Not explicitly supported; returns list of SQLAlchemy instances but not tested                | ❌        | `pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])` | N/A                                              |


---

### Notes on the Table

1. **Type Definitions**:
   - `Pydantic`: Instances of `UserPydantic` (inherits from `pydantic.BaseModel`).
   - `SQLModel`: Instances of `UserSQLModel` (inherits from `sqlmodel.SQLModel`).
   - `SQLAlchemy`: Instances of `UserSQLAlchemy` (inherits from `sqlalchemy.orm.declarative_base()`).
   - `pd.DataFrame`: Pandas DataFrame (`pandas.DataFrame`).
   - `pl.DataFrame`: Polars DataFrame (`polars.DataFrame`).

2. **Examples**:
   - **From**: The "Example From" column uses the exact sample data from `test_politipo.py` where applicable (e.g., `sample_dict`, `sample_pydantic`).
   - **To**: The "Example To" column shows the expected output, simplified for readability. For DataFrames, the output is shown as the constructor call that matches the content, though actual instances would have additional metadata (e.g., column types).
   - **N/A**: For unsupported conversions, no example output is provided as they raise exceptions.

3. **Supported Combinations**:
   - The ✅ entries are fully implemented and tested, with examples drawn from the test suite.
   - For DataFrame-related conversions, the `from_type` involves lists or DataFrames, and the `to_type` may result in lists, reflecting the library’s handling of collections.

4. **Unsupported Combinations**:
   - The ❌ entries show examples of inputs that would fail, with "N/A" for outputs since they aren’t produced.

This table provides a comprehensive view of `politipo`’s conversion capabilities, enriched with practical examples to illustrate each supported combination.
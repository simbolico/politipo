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
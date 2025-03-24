"""
Example demonstrating Pandera integration with politipo library.

This example shows how to:
1. Map types between Pandera and other libraries
2. Validate DataFrames with Pandera schemas
3. Convert between different representations

Prerequisites:
    pip install politipo pandas pandera polars
"""

import pandas as pd
import polars as pl
import pandera as pa
from sqlalchemy import Integer, String
import sys
import os

# Add the parent directory to the Python path to import politipo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Imported sys.path:", sys.path)

from politipo import TypeMapper, TypeConverter
print("Successfully imported politipo modules")

# Part 1: Type Mapping with Pandera
print("======= Type Mapping with Pandera =======")

mapper = TypeMapper()
print("Created TypeMapper instance")

# Map Pandera types to other libraries
print("\nMapping Pandera types to other libraries:")
sqlalchemy_int = mapper.map_type(pa.Int, to_library='sqlalchemy')
pandas_string = mapper.map_type(pa.String, to_library='pandas')
polars_float = mapper.map_type(pa.Float, to_library='polars')
python_bool = mapper.map_type(pa.Bool, to_library='python')

print(f"pa.Int -> SQLAlchemy: {sqlalchemy_int}")
print(f"pa.String -> Pandas: {pandas_string}")
print(f"pa.Float -> Polars: {polars_float}")
print(f"pa.Bool -> Python: {python_bool}")

# Map other libraries' types to Pandera
print("\nMapping other libraries' types to Pandera:")
pandera_int = mapper.map_type(Integer, to_library='pandera')
pandera_string = mapper.map_type(str, to_library='pandera', from_library='python')
pandera_float = mapper.map_type(pl.Float64, to_library='pandera')

print(f"SQLAlchemy Integer -> Pandera: {pandera_int}")
print(f"Python str -> Pandera: {pandera_string}")
print(f"Polars Float64 -> Pandera: {pandera_float}")

# Part 2: DataFrame Validation with Pandera Schemas
print("\n======= DataFrame Validation with Pandera Schemas =======")

# Define a simpler Pandera schema with just id and name
user_schema = pa.DataFrameSchema({
    "id": pa.Column(pa.Int, required=True),
    "name": pa.Column(pa.String, required=True)
})
print("Created Pandera schema")

# Create sample data
user_data = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Charlie"}
]

# Convert to Pandas DataFrame
pandas_df = pd.DataFrame(user_data)
print("\nPandas DataFrame:")
print(pandas_df)

# Validate with Pandera schema using TypeConverter
converter_pandas = TypeConverter(from_type=pd.DataFrame, to_type=user_schema)
print("Created TypeConverter for Pandas validation")
validated_pandas_df = converter_pandas.convert(pandas_df)
print("\nValidated Pandas DataFrame:")
print(validated_pandas_df)

# Convert to Polars DataFrame
polars_df = pl.DataFrame(user_data)
print("\nPolars DataFrame:")
print(polars_df)

# Validate with Pandera schema using TypeConverter
print("\nValidating Polars DataFrame (converts to Pandas behind the scenes):")
try:
    converter_polars = TypeConverter(from_type=pl.DataFrame, to_type=user_schema)
    print("Created TypeConverter for Polars validation")
    validated_polars_df = converter_polars.convert(polars_df)
    print("Validation successful - Polars DataFrame:")
    print(validated_polars_df)
except Exception as e:
    print(f"Note: Polars validation requires Pandas as an intermediary: {e}")

# Part 3: Handling validation errors
print("\n======= Handling Validation Errors =======")

# Create invalid data (wrong type for id)
invalid_data = [
    {"id": "four", "name": "David"},  # Invalid id (string instead of int)
    {"id": 5, "name": "Eve"}
]

invalid_df = pd.DataFrame(invalid_data)
print("\nInvalid DataFrame (with string id):")
print(invalid_df)

try:
    # This will raise a SchemaError because id must be an integer
    print("Attempting to validate invalid data (should fail)...")
    validated_invalid_df = converter_pandas.convert(invalid_df)
    print("This should not be printed - validation should fail")
except pa.errors.SchemaError as e:
    print("\nValidation Error (as expected):")
    print(f"  {str(e).split('Schema')[0]}")
    
print("\n======= End of Example =======") 
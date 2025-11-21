from datetime import datetime

import pandas as pd
import pandera as pa

# Create a simple schema
schema = pa.DataFrameSchema(
    {
        "name": pa.Column(dtype="string"),
        "age": pa.Column(dtype="int64"),
        "email": pa.Column(dtype="string", nullable=True),
        "created_at": pa.Column(dtype="datetime64"),
    }
)

print("Successfully created schema")
print(f"Schema type: {type(schema)}")
print(f"Columns: {list(schema.columns.keys())}")

# Create test data
test_df = pd.DataFrame(
    {
        "name": ["Alice Smith"],
        "age": [30],
        "email": ["alice@example.com"],
        "created_at": [datetime.now()],
    }
)

print("\nValidating test DataFrame...")
validated_df = schema.validate(test_df)
print("Validation successful!")

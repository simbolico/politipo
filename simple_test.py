#!/usr/bin/env python3
"""
Simple test for Politipo with Pandera integration
"""
import sys
from datetime import datetime
from typing import Optional
import pandas as pd
import pandera as pa

# Set up file for output
original_stdout = sys.stdout
with open('simple_test_results.txt', 'w') as f:
    sys.stdout = f
    
    try:
        from pydantic import BaseModel, Field
        print("Successfully imported Pydantic")
    except ImportError as e:
        print(f"Error importing Pydantic: {e}")
        sys.stdout = original_stdout
        sys.exit(1)

    try:
        from politipo import pydantic_to_pandera_schema, pandera_to_pydantic_model
        print("Successfully imported Politipo")
    except ImportError as e:
        print(f"Error importing Politipo: {e}")
        sys.stdout = original_stdout
        sys.exit(1)

    # Define minimal Pydantic model
    class User(BaseModel):
        name: str = Field(min_length=1, max_length=50)
        age: int = Field(ge=0, le=150)
        email: Optional[str] = None

    print("\nDefined User Pydantic model")

    try:
        # Test creating a Pandera schema manually
        print("\nTrying to create Pandera schema manually...")
        
        schema = pa.DataFrameSchema({
            "name": pa.Column(dtype="object", checks=[pa.Check.str_length(min_value=1, max_value=50)]),
            "age": pa.Column(dtype="int64", checks=[pa.Check.in_range(min_value=0, max_value=150)]),
            "email": pa.Column(dtype="object", nullable=True)
        })
        
        print("Successfully created Pandera schema manually")
        print(f"Schema type: {type(schema)}")
        print(f"Columns: {list(schema.columns.keys())}")
        
        # Test data
        df = pd.DataFrame({
            "name": ["Alice Smith"],
            "age": [30],
            "email": ["alice@example.com"]
        })
        
        print("\nValidating DataFrame...")
        validated_df = schema.validate(df)
        print("Validation successful!")
        
        # Now try with Politipo
        print("\nTrying Politipo conversion...")
        try:
            pandera_schema = pydantic_to_pandera_schema(User, verbose=True)
            print("\nSuccessfully created schema via Politipo!")
            print(f"Schema type: {type(pandera_schema)}")
            print(f"Columns: {list(pandera_schema.columns.keys())}")
            
            # Force dtype match
            print("Creating test DataFrame with correct types...")
            df2 = pd.DataFrame({
                "name": ["Alice Smith"],
                "age": [30],
                "email": ["alice@example.com"]
            })
            
            # Validate with the Politipo-generated schema
            print("Validating with Politipo schema (schema should have coerce=True built-in)...")
            validated_df = pandera_schema.validate(df2)  # Remove coerce parameter
            print("Validation with Politipo schema successful!")
        except Exception as e:
            print(f"Error in Politipo conversion: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in Pandera test: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
# Restore stdout
sys.stdout = original_stdout
print("Test completed. See simple_test_results.txt for results.") 
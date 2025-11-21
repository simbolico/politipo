#!/usr/bin/env python3
"""
Debug script for constraint conversion in Politipo's Pandera integration
"""
import inspect
import sys

from pydantic import BaseModel, Field

# Write output to file
with open("debug_constraints.txt", "w") as f:
    sys.stdout = f

    try:
        print("Importing required modules...")
        import pandas as pd

        from politipo import pydantic_to_pandera_schema
        from politipo.core.types.constraints import MaxLength, MaxValue, MinLength, MinValue
        from politipo.plugins.pandera.types import PanderaTypeSystem

        print("Imports successful")

        # Define test model
        class TestModel(BaseModel):
            name: str = Field(min_length=2, max_length=50)
            age: int = Field(ge=18, le=120)

        print("\n=== Model field info ===")
        for name, field in TestModel.model_fields.items():
            print(f"{name} field constraints:")
            if hasattr(field, "extra"):
                for constraint_name, constraint in field.extra.items():
                    print(f"  {constraint_name}: {constraint} (type: {type(constraint)})")
            elif hasattr(field, "constraints"):
                for constraint_name, constraint in field.constraints.items():
                    print(f"  {constraint_name}: {constraint} (type: {type(constraint)})")
            else:
                print(f"  Field has no constraints attribute: {field}")
                print(f"  Field attributes: {dir(field)}")

        # Debug the constraint objects
        print("\n=== Creating dummy constraints for testing ===")
        min_length = MinLength(value=2)
        max_length = MaxLength(value=50)
        min_value = MinValue(value=18)
        max_value = MaxValue(value=120)

        print(f"min_length: {min_length} ({type(min_length)})")
        print(f"min_length.value: {getattr(min_length, 'value', 'Not available')}")
        print(f"min_length has 'get' method: {hasattr(min_length, 'get')}")
        print(f"min_length.__dict__: {min_length.__dict__}")

        print(f"max_length: {max_length} ({type(max_length)})")
        print(f"max_value: {max_value} ({type(max_value)})")
        print(f"min_value: {min_value} ({type(min_value)})")

        # Debug type system conversion
        print("\n=== Creating Pandera schema via Politipo ===")
        type_system = PanderaTypeSystem()

        # Debug the specific method
        print("\n=== Examining _canonical_to_series method ===")
        print(inspect.getsource(type_system._canonical_to_series))

        # Try conversion
        print("\n=== Converting model to schema ===")
        try:
            schema = pydantic_to_pandera_schema(TestModel, verbose=True)
            print(f"Schema created: {type(schema)}")
            print(f"Columns: {list(schema.columns.keys())}")

            # Test with data
            df = pd.DataFrame({"name": ["John Doe"], "age": [30]})

            validated = schema.validate(df)
            print("Validation successful")

        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"Setup error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

# Restore stdout
sys.stdout = sys.__stdout__
print("Debug complete. See debug_constraints.txt for results.")

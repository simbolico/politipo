from datetime import datetime
from typing import List, Optional
from politipo.core.conversion import ConversionEngine
from politipo.plugins.pandas import PandasTypeSystem
from politipo.plugins.pydantic import PydanticTypeSystem
from politipo.plugins.pandera import PanderaTypeSystem
from politipo.core.conversion.strategies.pandas_to_pydantic import DataFrameToModelStrategy
from politipo.core.errors import PolitipoError, ConversionError
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from politipo import pydantic_to_pandera_schema, pandera_to_pydantic_model
import pandera as pa


class Address(BaseModel):
    """Nested model for demonstration."""
    street: str = Field(..., min_length=1, max_length=100)
    city: str = Field(..., min_length=1)
    country: str = Field(..., min_length=2, max_length=2)  # ISO country code


class User(BaseModel):
    """User model with nested fields and constraints."""
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    created_at: datetime
    addresses: List[Address]


def test_schema_tools():
    """Test the schema translation functions."""
    print("\n=== Testing Schema Translation Tools ===\n")

    # First, let's create a simple schema directly as a fallback
    print("Creating a simple schema directly:")
    simple_schema = pa.DataFrameSchema({
        "name": pa.Column(dtype="string", checks=[pa.Check.str_length(min_value=1, max_value=50)]),
        "age": pa.Column(dtype="int64", checks=[pa.Check.in_range(min_value=0, max_value=150)]),
        "email": pa.Column(dtype="string", nullable=True),
        "created_at": pa.Column(dtype="datetime64[ns]")
    })
    print("Successfully created simple schema directly!")
    print(f"Columns: {list(simple_schema.columns.keys())}\n")

    # Create a flattened version of User model for Pandera testing
    class FlatUser(BaseModel):
        """Flattened user model for Pandera compatibility."""
        name: str = Field(..., min_length=1, max_length=50)
        age: int = Field(..., ge=0, le=150)
        email: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        created_at: datetime

    print("Test 1: Converting flattened Pydantic User model to Pandera schema...")
    try:
        pandera_schema = pydantic_to_pandera_schema(FlatUser, verbose=True)
        print("\nSuccessfully created Pandera schema:")
        print(f"Schema type: {type(pandera_schema)}")
        print("\nColumns:")
        for col_name, col in pandera_schema.columns.items():
            print(f"  {col_name}:")
            print(f"    - dtype: {col.dtype}")
            print(f"    - nullable: {col.nullable}")
            if col.checks:
                print(f"    - checks: {[check.name for check in col.checks]}")

        # Create a test DataFrame that should pass validation
        test_df = pd.DataFrame({
            "name": ["Alice Smith"],
            "age": [30],
            "email": ["alice@example.com"],
            "created_at": [datetime.now()],
        })
        
        # Ensure we have the correct dtypes in our DataFrame
        test_df = test_df.astype({
            "name": "string",
            "age": "int64",
            "email": "string",
            "created_at": "datetime64[ns]"
        })

        print("\nValidating test DataFrame with generated schema...")
        validated_df = pandera_schema.validate(test_df)
        print("✓ Validation successful!")

        # Test 2: Convert the Pandera schema back to a Pydantic model
        print("\nTest 2: Converting Pandera schema back to Pydantic model...")
        regenerated_model = pandera_to_pydantic_model(pandera_schema)
        print("\nSuccessfully created Pydantic model:")
        print(f"Model name: {regenerated_model.__name__}")
        print("Fields:")

        # Handle both Pydantic v1 and v2 models, prioritizing v2 API pattern
        if hasattr(regenerated_model, 'model_fields'):
            # Pydantic v2
            for field_name, field in regenerated_model.model_fields.items():
                print(f"  {field_name}:")
                print(f"    - type: {field.annotation}")
                print(f"    - required: {field.is_required()}")
                if field.description:
                    print(f"    - description: {field.description}")
        elif hasattr(regenerated_model, '__fields__'):
            # Pydantic v1 (legacy support)
            for field_name, field in regenerated_model.__fields__.items():
                print(f"  {field_name}:")
                try:
                    print(f"    - type: {field.type_}")
                except AttributeError:
                    print(f"    - type: {field.annotation if hasattr(field, 'annotation') else 'unknown'}")
                try:
                    print(f"    - required: {field.required}")
                except AttributeError:
                    print(f"    - required: {not field.allow_none if hasattr(field, 'allow_none') else 'unknown'}")
                if hasattr(field, 'field_info') and hasattr(field.field_info, 'description') and field.field_info.description:
                    print(f"    - description: {field.field_info.description}")
        else:
            print("  Unable to inspect model fields")

        # Test the regenerated model with the same data
        print("\nValidating data with regenerated Pydantic model...")
        try:
            # Get the first row of data as a dict
            test_data = test_df.iloc[0].to_dict()
            
            # Try to create an instance with the data
            instance = regenerated_model(**test_data)
            print("✓ Validation successful!")
            print(f"Created instance: {instance}")
        except Exception as validation_error:
            print(f"Validation error: {type(validation_error).__name__}: {validation_error}")
            print("Unable to create instance of regenerated model.")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Use the simple schema as a fallback
        print("\nUsing simple schema as fallback")
        try:
            # Create a test DataFrame
            test_df = pd.DataFrame({
                "name": ["Alice Smith"],
                "age": [30],
                "email": ["alice@example.com"],
                "created_at": [datetime.now()],
            })
            
            # Ensure we have the correct dtypes in our DataFrame
            test_df = test_df.astype({
                "name": "string",
                "age": "int64",
                "email": "string",
                "created_at": "datetime64[ns]"
            })

            # Validate with simple schema
            print("Validating with simple schema...")
            validated_df = simple_schema.validate(test_df)
            print("✓ Simple schema validation successful!")
        except Exception as e2:
            print(f"Error with fallback: {e2}")


def main():
    try:
        # Initialize engine
        engine = ConversionEngine()
        pandas_system = PandasTypeSystem()
        pydantic_system = PydanticTypeSystem()
        
        # Register type systems
        engine.register_type_system(pandas_system)
        engine.register_type_system(pydantic_system)

        # Register conversion strategy
        engine.register_strategy(
            "df_to_model",
            DataFrameToModelStrategy()
        )

        # Create a DataFrame with test data including nested structures
        df = pd.DataFrame({
            "name": ["Alice Smith", "Bob Jones", ""],  # Empty name to test constraint
            "age": [30, 25, 200],  # Invalid age to test constraint
            "email": ["alice@example.com", None, "invalid-email"],  # Test optional and invalid
            "created_at": pd.date_range("2024-01-01", periods=3),
            "addresses": [
                [{"street": "123 Main St", "city": "New York", "country": "US"}],
                [{"street": "456 High St", "city": "London", "country": "UK"}],
                [{"street": "", "city": "Paris", "country": "FRA"}]  # Invalid data
            ]
        })

        print("Converting DataFrame to User models...")
        print("\nInput DataFrame:")
        print(df)
        print("\nAttempting conversion...")

        # First convert the DataFrame to a list of dictionaries
        df_dicts = df.to_dict('records')

        # Create and validate User models
        users = []
        for record in df_dicts:
            try:
                # Handle nested Address models
                if "addresses" in record:
                    record["addresses"] = [
                        Address(**addr) for addr in record["addresses"]
                    ]
                
                user = User(**record)
                users.append(user)
                print(f"\nSuccessfully converted user:")
                print(f"  Name: {user.name}")
                print(f"  Age: {user.age}")
                print(f"  Email: {user.email}")
                print(f"  Created: {user.created_at}")
                print("  Addresses:")
                for addr in user.addresses:
                    print(f"    - {addr.street}, {addr.city}, {addr.country}")
                    
            except PolitipoError as e:
                print(f"\nConversion error: {e}")
            except Exception as e:
                print(f"\nValidation error: {e}")

        print(f"\nSuccessfully converted {len(users)} out of {len(df)} records")

    except PolitipoError as e:
        print(f"Politipo error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    try:
        # Set up logging to file
        import sys
        import io
        
        # Redirect stdout to capture output
        original_stdout = sys.stdout
        output_buffer = io.StringIO()
        sys.stdout = output_buffer
        
        # Run the tests
        main()
        print("\n" + "="*50)
        test_schema_tools()
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Write the output to a file
        with open("test_results.txt", "w") as f:
            f.write(output_buffer.getvalue())
            
        print("Tests completed. Results written to test_results.txt")
        
    except Exception as e:
        # Make sure to restore stdout even if an exception occurs
        sys.stdout = original_stdout
        print(f"Error running tests: {e}")
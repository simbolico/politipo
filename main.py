from datetime import datetime
from typing import List, Optional
from politipo.core.conversion import ConversionEngine
from politipo.plugins.pandas import PandasTypeSystem
from politipo.plugins.pydantic import PydanticTypeSystem
from politipo.core.conversion.strategies.pandas_to_pydantic import DataFrameToModelStrategy
from politipo.core.errors import PolitipoError
from pydantic import BaseModel, Field
import pandas as pd


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
    main()
import pytest
from politipo import TypeConverter
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
import pandas as pd
import polars as pl
from unittest.mock import patch
import sys

# Check if Pandera is installed
try:
    import pandera as pa
    has_pandera = True
except ImportError:
    has_pandera = False

# Setup for SQLAlchemy
Base = declarative_base()

# Define test models
class UserPydantic(BaseModel):
    id: int
    name: str

class UserSQLModel(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str

class UserSQLAlchemy(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)

# Sample data for testing
sample_dict = {"id": 1, "name": "Alice"}
sample_pydantic = UserPydantic(id=1, name="Alice")
sample_sqlmodel = UserSQLModel(id=1, name="Alice")
sample_sqlalchemy = UserSQLAlchemy(id=1, name="Alice")
sample_list_dicts = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
sample_list_pydantic = [UserPydantic(id=1, name="Alice"), UserPydantic(id=2, name="Bob")]
sample_list_sqlmodel = [UserSQLModel(id=1, name="Alice"), UserSQLModel(id=2, name="Bob")]
sample_list_sqlalchemy = [UserSQLAlchemy(id=1, name="Alice"), UserSQLAlchemy(id=2, name="Bob")]

# DataFrames
sample_df_pandas = pd.DataFrame(sample_list_dicts)
sample_df_polars = pl.DataFrame(sample_list_dicts)

# ### Conversion to `dict`
def test_to_dict_from_dict():
    converter = TypeConverter(from_type=dict, to_type=dict)
    result = converter.convert(sample_dict)
    assert result == sample_dict

def test_to_dict_from_pydantic():
    converter = TypeConverter(from_type=UserPydantic, to_type=dict)
    result = converter.convert(sample_pydantic)
    assert result == sample_dict

def test_to_dict_from_sqlmodel():
    converter = TypeConverter(from_type=UserSQLModel, to_type=dict)
    result = converter.convert(sample_sqlmodel)
    assert result == sample_dict

def test_to_dict_from_sqlalchemy():
    converter = TypeConverter(from_type=UserSQLAlchemy, to_type=dict)
    result = converter.convert(sample_sqlalchemy)
    assert result == sample_dict

# ### Conversion to Pydantic Models
def test_to_pydantic_from_dict():
    converter = TypeConverter(from_type=dict, to_type=UserPydantic)
    result = converter.convert(sample_dict)
    assert isinstance(result, UserPydantic)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_pydantic_from_pydantic():
    converter = TypeConverter(from_type=UserPydantic, to_type=UserPydantic)
    result = converter.convert(sample_pydantic)
    assert result == sample_pydantic

def test_to_pydantic_from_sqlmodel():
    converter = TypeConverter(from_type=UserSQLModel, to_type=UserPydantic)
    result = converter.convert(sample_sqlmodel)
    assert isinstance(result, UserPydantic)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_pydantic_from_sqlalchemy():
    converter = TypeConverter(from_type=UserSQLAlchemy, to_type=UserPydantic)
    result = converter.convert(sample_sqlalchemy)
    assert isinstance(result, UserPydantic)
    assert result.id == 1
    assert result.name == "Alice"

# ### Conversion to SQLModel Instances
def test_to_sqlmodel_from_dict():
    converter = TypeConverter(from_type=dict, to_type=UserSQLModel)
    result = converter.convert(sample_dict)
    assert isinstance(result, UserSQLModel)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_sqlmodel_from_pydantic():
    converter = TypeConverter(from_type=UserPydantic, to_type=UserSQLModel)
    result = converter.convert(sample_pydantic)
    assert isinstance(result, UserSQLModel)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_sqlmodel_from_sqlmodel():
    converter = TypeConverter(from_type=UserSQLModel, to_type=UserSQLModel)
    result = converter.convert(sample_sqlmodel)
    assert result == sample_sqlmodel

def test_to_sqlmodel_from_sqlalchemy():
    converter = TypeConverter(from_type=UserSQLAlchemy, to_type=UserSQLModel)
    result = converter.convert(sample_sqlalchemy)
    assert isinstance(result, UserSQLModel)
    assert result.id == 1
    assert result.name == "Alice"

# ### Conversion to SQLAlchemy Models
def test_to_sqlalchemy_from_dict():
    converter = TypeConverter(from_type=dict, to_type=UserSQLAlchemy)
    result = converter.convert(sample_dict)
    assert isinstance(result, UserSQLAlchemy)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_sqlalchemy_from_pydantic():
    converter = TypeConverter(from_type=UserPydantic, to_type=UserSQLAlchemy)
    result = converter.convert(sample_pydantic)
    assert isinstance(result, UserSQLAlchemy)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_sqlalchemy_from_sqlmodel():
    converter = TypeConverter(from_type=UserSQLModel, to_type=UserSQLAlchemy)
    result = converter.convert(sample_sqlmodel)
    assert isinstance(result, UserSQLAlchemy)
    assert result.id == 1
    assert result.name == "Alice"

def test_to_sqlalchemy_from_sqlalchemy():
    converter = TypeConverter(from_type=UserSQLAlchemy, to_type=UserSQLAlchemy)
    result = converter.convert(sample_sqlalchemy)
    assert result == sample_sqlalchemy

# ### Conversion to Pandas DataFrame
def test_to_pandas_from_list_dicts():
    converter = TypeConverter(from_type=dict, to_type=pd.DataFrame)
    with pytest.raises(ValueError):  # Single dict should fail for DataFrame
        converter.convert(sample_dict)
    converter = TypeConverter(from_type=dict, to_type=pd.DataFrame)
    result = converter.convert(sample_list_dicts)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_df_pandas)

def test_to_pandas_from_list_pydantic():
    converter = TypeConverter(from_type=UserPydantic, to_type=pd.DataFrame)
    result = converter.convert(sample_list_pydantic)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_df_pandas)

def test_to_pandas_from_list_sqlmodel():
    converter = TypeConverter(from_type=UserSQLModel, to_type=pd.DataFrame)
    result = converter.convert(sample_list_sqlmodel)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_df_pandas)

# ### Conversion to Polars DataFrame
def test_to_polars_from_list_dicts():
    converter = TypeConverter(from_type=dict, to_type=pl.DataFrame)
    with pytest.raises(ValueError):  # Single dict should fail for DataFrame
        converter.convert(sample_dict)
    converter = TypeConverter(from_type=dict, to_type=pl.DataFrame)
    result = converter.convert(sample_list_dicts)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(sample_df_polars)

def test_to_polars_from_list_pydantic():
    converter = TypeConverter(from_type=UserPydantic, to_type=pl.DataFrame)
    result = converter.convert(sample_list_pydantic)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(sample_df_polars)

def test_to_polars_from_list_sqlmodel():
    converter = TypeConverter(from_type=UserSQLModel, to_type=pl.DataFrame)
    result = converter.convert(sample_list_sqlmodel)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(sample_df_polars)

# ### Error Handling Tests
def test_unsupported_from_type():
    converter = TypeConverter(from_type=str, to_type=dict)
    with pytest.raises(ValueError, match="Unsupported from_type"):
        converter.convert("invalid")

def test_unsupported_to_type():
    converter = TypeConverter(from_type=dict, to_type=list)
    with pytest.raises(ValueError, match="Unsupported to_type"):
        converter.convert(sample_dict)

def test_missing_library():
    with patch.dict(sys.modules, {'pydantic': None}):
        converter = TypeConverter(from_type=UserPydantic, to_type=dict)
        with pytest.raises(ImportError, match="Pydantic is required for this conversion"):
            converter.convert(sample_dict)

# ### Collection Handling Tests
def test_collection_to_pydantic_from_pandas():
    converter = TypeConverter(from_type=pd.DataFrame, to_type=UserPydantic)
    result = converter.convert(sample_df_pandas)
    assert isinstance(result, list)
    assert all(isinstance(item, UserPydantic) for item in result)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[0].name == "Alice"
    assert result[1].id == 2
    assert result[1].name == "Bob"

def test_collection_to_sqlmodel_from_polars():
    converter = TypeConverter(from_type=pl.DataFrame, to_type=UserSQLModel)
    result = converter.convert(sample_df_polars)
    assert isinstance(result, list)
    assert all(isinstance(item, UserSQLModel) for item in result)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[0].name == "Alice"
    assert result[1].id == 2
    assert result[1].name == "Bob"

# New tests for DataFrame to SQLAlchemy conversions
def test_collection_to_sqlalchemy_from_pandas():
    converter = TypeConverter(from_type=pd.DataFrame, to_type=UserSQLAlchemy)
    result = converter.convert(sample_df_pandas)
    assert isinstance(result, list)
    assert all(isinstance(item, UserSQLAlchemy) for item in result)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[0].name == "Alice"
    assert result[1].id == 2
    assert result[1].name == "Bob"

def test_collection_to_sqlalchemy_from_polars():
    converter = TypeConverter(from_type=pl.DataFrame, to_type=UserSQLAlchemy)
    result = converter.convert(sample_df_polars)
    assert isinstance(result, list)
    assert all(isinstance(item, UserSQLAlchemy) for item in result)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[0].name == "Alice"
    assert result[1].id == 2
    assert result[1].name == "Bob"

# New tests for SQLAlchemy to DataFrame conversions
def test_to_pandas_from_list_sqlalchemy():
    converter = TypeConverter(from_type=UserSQLAlchemy, to_type=pd.DataFrame)
    result = converter.convert(sample_list_sqlalchemy)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_df_pandas)

def test_to_polars_from_list_sqlalchemy():
    converter = TypeConverter(from_type=UserSQLAlchemy, to_type=pl.DataFrame)
    result = converter.convert(sample_list_sqlalchemy)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(sample_df_polars)

# Pandera Tests
@pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
def test_pandera_schema_validation():
    """Test validating DataFrames with Pandera schemas using TypeConverter."""
    import pandera as pa
    
    # Define a Pandera schema
    schema = pa.DataFrameSchema({
        "id": pa.Column(pa.Int, required=True),
        "name": pa.Column(pa.String, required=True)
    })
    
    # Create a valid DataFrame
    valid_df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    # Create an invalid DataFrame (wrong type)
    invalid_df = pd.DataFrame({
        "id": ["1", "2", "3"],  # Strings instead of integers
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    # Test successful validation
    converter = TypeConverter(from_type=pd.DataFrame, to_type=schema)
    result_df = converter.convert(valid_df)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 3
    
    # Test failed validation
    with pytest.raises(Exception):  # Should raise a SchemaError, but we'll catch any exception for compatibility
        converter.convert(invalid_df)

@pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
@pytest.mark.skip(reason="Pandera does not directly support Polars DataFrames yet")
def test_polars_with_pandera():
    """Test validating Polars DataFrames with Pandera schemas."""
    import pandera as pa
    
    # Define a Pandera schema
    schema = pa.DataFrameSchema({
        "id": pa.Column(pa.Int, required=True),
        "name": pa.Column(pa.String, required=True)
    })
    
    # Create a valid Polars DataFrame
    valid_df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    
    # Test validation with Polars DataFrame
    converter = TypeConverter(from_type=pl.DataFrame, to_type=schema)
    result_df = converter.convert(valid_df)
    assert isinstance(result_df, pl.DataFrame)
    assert len(result_df) == 3

@pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
def test_convert_methods_with_pandera():
    """Test that convert_single and convert_collection work with Pandera schemas."""
    import pandera as pa
    
    with patch('pandera.DataFrameSchema.validate') as mock_validate:
        # Setup mock return value
        mock_validate.return_value = pd.DataFrame({"id": [1], "name": ["Alice"]})
        
        # Define a Pandera schema
        schema = pa.DataFrameSchema({
            "id": pa.Column(pa.Int, required=True),
            "name": pa.Column(pa.String, required=True)
        })
        
        # Create a test DataFrame
        df = pd.DataFrame({"id": [1], "name": ["Alice"]})
        
        # Test convert_single
        converter = TypeConverter(from_type=pd.DataFrame, to_type=schema)
        result_single = converter.convert_single(df)
        assert len(result_single) == 1
        mock_validate.assert_called_once_with(df)
        
        # Reset mock
        mock_validate.reset_mock()
        
        # Test convert_collection
        result_collection = converter.convert_collection(df)
        assert len(result_collection) == 1
        mock_validate.assert_called_once_with(df)

@pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
def test_pandera_converter_missing_pandera_error():
    """Test that appropriate error is raised when Pandera is not available."""
    with patch.dict(sys.modules, {'pandera': None}):
        # For TypeConverter, we need to create a dummy schema-like object
        class FakeSchema:
            def __init__(self):
                self.validate = lambda x: x
                self.columns = {}
        
        converter = TypeConverter(from_type=dict, to_type=FakeSchema())
        
        with pytest.raises(ImportError):
            converter.convert({"id": 1, "name": "Alice"}) 

import pytest
from politipo import TypeConverter
import sys
from typing import List, Optional, Dict, Any
from unittest.mock import patch

# Check if Pydantic is installed
try:
    import pydantic
    from pydantic import BaseModel, Field, EmailStr
    has_pydantic = True
    
    # Check Pydantic version
    is_v2 = hasattr(BaseModel, "model_dump")
    if is_v2:
        from typing import Annotated
except ImportError:
    has_pydantic = False
    is_v2 = False


@pytest.mark.skipif(not has_pydantic, reason="Pydantic not installed")
class TestAdvancedPydanticTypeConverter:
    """Test suite for advanced Pydantic functionality in TypeConverter."""
    
    def setup_method(self):
        """Set up test models and sample data."""
        # Define nested Pydantic models
        class Address(BaseModel):
            street: str
            city: str
            zip_code: str
            
        class User(BaseModel):
            id: int
            name: str
            email: Optional[str] = None
            address: Address
            tags: List[str] = []
            
        self.Address = Address
        self.User = User
        
        # Sample data
        self.address_dict = {
            "street": "123 Main St",
            "city": "Anytown",
            "zip_code": "12345"
        }
        
        self.user_dict = {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "address": self.address_dict,
            "tags": ["customer", "premium"]
        }
        
        # Complex model with array of models
        class Comment(BaseModel):
            id: int
            text: str
            
        class BlogPost(BaseModel):
            id: int
            title: str
            content: str
            comments: List[Comment] = []
            
        self.Comment = Comment
        self.BlogPost = BlogPost
        
        self.blog_dict = {
            "id": 1,
            "title": "Hello World",
            "content": "This is my first post",
            "comments": [
                {"id": 1, "text": "Great post!"},
                {"id": 2, "text": "Thanks for sharing"}
            ]
        }
    
    def test_nested_pydantic_conversion(self):
        """Test conversion with nested Pydantic models."""
        converter = TypeConverter(from_type=dict, to_type=self.User)
        user = converter.convert(self.user_dict)
        
        # Check user properties
        assert user.id == 1
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        
        # Check nested address
        assert isinstance(user.address, self.Address)
        assert user.address.street == "123 Main St"
        assert user.address.city == "Anytown"
        assert user.address.zip_code == "12345"
        
        # Check array property
        assert len(user.tags) == 2
        assert user.tags[0] == "customer"
        
    def test_list_of_nested_models(self):
        """Test conversion with a list of nested Pydantic models."""
        converter = TypeConverter(from_type=dict, to_type=self.BlogPost)
        blog = converter.convert(self.blog_dict)
        
        # Check blog properties
        assert blog.id == 1
        assert blog.title == "Hello World"
        
        # Check comments array
        assert len(blog.comments) == 2
        assert all(isinstance(comment, self.Comment) for comment in blog.comments)
        assert blog.comments[0].id == 1
        assert blog.comments[0].text == "Great post!"
        
    def test_round_trip_conversion(self):
        """Test round-trip conversion with nested models."""
        # Dict to model
        to_model_converter = TypeConverter(from_type=dict, to_type=self.User)
        user = to_model_converter.convert(self.user_dict)
        
        # Model to dict
        to_dict_converter = TypeConverter(from_type=self.User, to_type=dict)
        result_dict = to_dict_converter.convert(user)
        
        # Compare dictionaries (key by key to avoid serialization differences)
        assert result_dict["id"] == self.user_dict["id"]
        assert result_dict["name"] == self.user_dict["name"]
        assert result_dict["email"] == self.user_dict["email"]
        assert isinstance(result_dict["address"], dict)
        assert result_dict["address"]["street"] == self.user_dict["address"]["street"]
        
    @pytest.mark.skipif(not is_v2, reason="Requires Pydantic v2")
    def test_pydantic_v2_model_validation(self):
        """Test Pydantic v2 model validation features."""
        class ProductV2(BaseModel):
            id: int
            name: str
            price: Annotated[float, Field(gt=0, description="Product price")]
            
        # Valid data
        valid_data = {"id": 1, "name": "Test Product", "price": 29.99}
        converter = TypeConverter(from_type=dict, to_type=ProductV2)
        product = converter.convert(valid_data)
        assert product.price == 29.99
        
        # Invalid data - should raise validation error with coerce=True
        invalid_data = {"id": 1, "name": "Test Product", "price": -10.0}
        converter = TypeConverter(from_type=dict, to_type=ProductV2)
        with pytest.raises(Exception):
            converter.convert(invalid_data, coerce=True)
            
    @pytest.mark.skipif(is_v2, reason="Requires Pydantic v1")
    def test_pydantic_v1_model_validation(self):
        """Test Pydantic v1 model validation features."""
        from pydantic import validator
        
        class ProductV1(BaseModel):
            id: int
            name: str
            price: float
            
            @validator('price')
            def price_must_be_positive(cls, v):
                if v <= 0:
                    raise ValueError('Price must be positive')
                return v
                
        # Valid data
        valid_data = {"id": 1, "name": "Test Product", "price": 29.99}
        converter = TypeConverter(from_type=dict, to_type=ProductV1)
        product = converter.convert(valid_data)
        assert product.price == 29.99
        
        # Invalid data - should raise validation error with coerce=True
        invalid_data = {"id": 1, "name": "Test Product", "price": -10.0}
        converter = TypeConverter(from_type=dict, to_type=ProductV1)
        with pytest.raises(Exception):
            converter.convert(invalid_data, coerce=True)
            
    def test_optional_nested_fields(self):
        """Test handling of optional nested fields."""
        class OptionalAddress(BaseModel):
            street: Optional[str] = None
            city: Optional[str] = None
            
        class OptionalUser(BaseModel):
            id: int
            name: str
            address: Optional[OptionalAddress] = None
            
        # Test with address
        data_with_address = {
            "id": 1, 
            "name": "Alice",
            "address": {"street": "123 Main St", "city": "Anytown"}
        }
        
        converter = TypeConverter(from_type=dict, to_type=OptionalUser)
        user_with_address = converter.convert(data_with_address)
        assert user_with_address.address is not None
        assert user_with_address.address.street == "123 Main St"
        
        # Test without address
        data_without_address = {
            "id": 2,
            "name": "Bob"
        }
        
        user_without_address = converter.convert(data_without_address)
        assert user_without_address.address is None
        
    def test_pydantic_version_detection(self):
        """Test Pydantic version detection in TypeConverter."""
        converter = TypeConverter(from_type=dict, to_type=self.User)
        version = converter._get_pydantic_version()
        
        if is_v2:
            assert version == 2
        else:
            assert version == 1
            
    def test_missing_pydantic(self):
        """Test behavior when Pydantic is not available."""
        with patch.dict(sys.modules, {'pydantic': None}):
            converter = TypeConverter(from_type=dict, to_type=dict)
            with pytest.raises(ImportError):
                converter._get_pydantic_version()


@pytest.mark.skipif(not has_pydantic, reason="Pydantic not installed")
class TestPydanticDataFrameConversions:
    """Test Pydantic conversion with DataFrames."""
    
    def setup_method(self):
        """Set up test models and sample data."""
        try:
            import pandas as pd
            import polars as pl
            self.has_dataframes = True
        except ImportError:
            self.has_dataframes = False
            return
            
        # Define model with various field types
        class Product(BaseModel):
            id: int
            name: str
            price: float
            in_stock: bool
            tags: List[str] = []
            
        self.Product = Product
        
        # Sample data
        self.products = [
            {"id": 1, "name": "Product A", "price": 19.99, "in_stock": True, "tags": ["new", "featured"]},
            {"id": 2, "name": "Product B", "price": 29.99, "in_stock": False, "tags": ["clearance"]},
            {"id": 3, "name": "Product C", "price": 39.99, "in_stock": True, "tags": []}
        ]
        
    @pytest.mark.skipif(not has_pydantic, reason="Pydantic not installed")
    def test_pydantic_to_pandas(self):
        """Test conversion from list of Pydantic models to Pandas DataFrame."""
        if not self.has_dataframes:
            pytest.skip("Pandas not installed")
            
        import pandas as pd
        
        # Create Pydantic models
        products = [self.Product(**p) for p in self.products]
        
        # Convert to DataFrame
        converter = TypeConverter(from_type=self.Product, to_type=pd.DataFrame)
        df = converter.convert(products)
        
        # Verify DataFrame
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "price", "in_stock", "tags"]
        assert df.iloc[0]["id"] == 1
        assert df.iloc[1]["name"] == "Product B"
        assert df.iloc[2]["price"] == 39.99
        
    @pytest.mark.skipif(not has_pydantic, reason="Pydantic not installed")
    def test_pandas_to_pydantic(self):
        """Test conversion from Pandas DataFrame to list of Pydantic models."""
        if not self.has_dataframes:
            pytest.skip("Pandas not installed")
            
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(self.products)
        
        # Convert to Pydantic models
        converter = TypeConverter(from_type=pd.DataFrame, to_type=self.Product)
        products = converter.convert(df)
        
        # Verify models
        assert len(products) == 3
        assert all(isinstance(p, self.Product) for p in products)
        assert products[0].id == 1
        assert products[1].name == "Product B"
        assert products[2].price == 39.99
        assert not products[1].in_stock
        assert len(products[0].tags) == 2 
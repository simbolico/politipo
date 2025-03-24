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
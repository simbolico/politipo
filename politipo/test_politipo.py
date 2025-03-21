import pytest
from politipo import TypeConverter
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base
import pandas as pd
import polars as pl

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
    import sys
    original_pydantic = sys.modules.get('pydantic')
    sys.modules['pydantic'] = None  # Simulate missing Pydantic
    converter = TypeConverter(from_type=UserPydantic, to_type=dict)
    with pytest.raises(ImportError, match="Pydantic is required"):
        converter.convert(sample_pydantic)
    sys.modules['pydantic'] = original_pydantic  # Restore Pydantic

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
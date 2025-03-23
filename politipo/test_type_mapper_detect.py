import pytest
from politipo import TypeMapper
import datetime
import decimal

def test_detect_library_python_types():
    mapper = TypeMapper()
    
    # Test Python built-in types
    assert mapper.detect_library(int) == 'python'
    assert mapper.detect_library(str) == 'python'
    assert mapper.detect_library(float) == 'python'
    assert mapper.detect_library(bool) == 'python'
    assert mapper.detect_library(list) == 'python'
    assert mapper.detect_library(dict) == 'python'
    assert mapper.detect_library(datetime.date) == 'python'
    assert mapper.detect_library(datetime.datetime) == 'python'
    assert mapper.detect_library(decimal.Decimal) == 'python'
    
    # Test typing annotations
    from typing import List, Dict
    assert mapper.detect_library(List[int]) == 'python'
    assert mapper.detect_library(Dict[str, int]) == 'python'

def test_detect_library_sqlalchemy_types():
    mapper = TypeMapper()
    
    try:
        from sqlalchemy import Integer, String, Float, Boolean
        
        assert mapper.detect_library(Integer) == 'sqlalchemy'
        assert mapper.detect_library(String) == 'sqlalchemy'
        assert mapper.detect_library(Float) == 'sqlalchemy'
        assert mapper.detect_library(Boolean) == 'sqlalchemy'
    except ImportError:
        pytest.skip("SQLAlchemy not installed")

def test_detect_library_pandas_types():
    mapper = TypeMapper()
    
    try:
        import pandas as pd
        
        assert mapper.detect_library(pd.Int64Dtype()) == 'pandas'
        assert mapper.detect_library(pd.StringDtype()) == 'pandas'
        assert mapper.detect_library('float64') == 'pandas'
        assert mapper.detect_library('bool') == 'pandas'
    except ImportError:
        pytest.skip("Pandas not installed")

def test_detect_library_polars_types():
    mapper = TypeMapper()
    
    try:
        import polars as pl
        
        assert mapper.detect_library(pl.Int64) == 'polars'
        assert mapper.detect_library(pl.Utf8) == 'polars'
        assert mapper.detect_library(pl.Float64) == 'polars'
        assert mapper.detect_library(pl.Boolean) == 'polars'
    except ImportError:
        pytest.skip("Polars not installed")

def test_auto_detection_in_map_type():
    mapper = TypeMapper()
    
    # Test Python to SQLAlchemy (auto-detection)
    try:
        from sqlalchemy import Integer
        result = mapper.map_type(int, 'sqlalchemy')
        assert result is Integer
    except ImportError:
        pytest.skip("SQLAlchemy not installed")
    
    # Test SQLAlchemy to Polars (auto-detection)
    try:
        from sqlalchemy import Integer
        import polars as pl
        result = mapper.map_type(Integer, 'polars')
        assert result is pl.Int64
    except ImportError:
        pytest.skip("SQLAlchemy or Polars not installed")

def test_invalid_type_detection():
    mapper = TypeMapper()
    
    # Create a custom class that doesn't match any library
    class CustomType:
        pass
        
    with pytest.raises(ValueError, match="Could not automatically determine library"):
        mapper.detect_library(CustomType)
        
    with pytest.raises(ValueError, match="Could not automatically determine library"):
        mapper.map_type(CustomType, 'python') 
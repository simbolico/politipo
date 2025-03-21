import pytest
from politipo.type_mapper import TypeMapper
import sys

# Mock imports for testing
try:
    import sqlalchemy
    has_sqlalchemy = True
except ImportError:
    has_sqlalchemy = False

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import polars as pl
    has_polars = True
except ImportError:
    has_polars = False


class TestTypeMapper:
    """Test suite for TypeMapper class."""
    
    def setup_method(self):
        """Set up TypeMapper instance before each test."""
        self.mapper = TypeMapper()
    
    # Python type mapping tests
    def test_python_to_canonical(self):
        """Test conversion from Python types to canonical types."""
        assert self.mapper.get_canonical_type(int, 'python') == 'integer'
        assert self.mapper.get_canonical_type(str, 'python') == 'string'
        assert self.mapper.get_canonical_type(float, 'python') == 'float'
        assert self.mapper.get_canonical_type(bool, 'python') == 'boolean'
    
    def test_canonical_to_python(self):
        """Test conversion from canonical types to Python types."""
        assert self.mapper.get_library_type('integer', 'python') is int
        assert self.mapper.get_library_type('string', 'python') is str
        assert self.mapper.get_library_type('float', 'python') is float
        assert self.mapper.get_library_type('boolean', 'python') is bool
    
    def test_python_nonetype_raises_error(self):
        """Test that None type raises an error."""
        with pytest.raises(ValueError, match="NoneType is not mapped"):
            self.mapper.get_canonical_type(type(None), 'python')
    
    def test_invalid_canonical_type(self):
        """Test handling of invalid canonical types."""
        with pytest.raises(ValueError, match="No Python type for canonical"):
            self.mapper.get_library_type('invalid_type', 'python')
    
    def test_invalid_python_type(self):
        """Test handling of invalid Python types."""
        with pytest.raises(ValueError, match="No canonical mapping for Python type"):
            self.mapper.get_canonical_type(list, 'python')
    
    # Library support tests
    def test_unsupported_library(self):
        """Test handling of unsupported libraries."""
        with pytest.raises(ValueError, match="Unsupported library"):
            self.mapper.get_canonical_type(int, 'unsupported_lib')
        
        with pytest.raises(ValueError, match="Unsupported library"):
            self.mapper.get_library_type('integer', 'unsupported_lib')
    
    # Cross-library mapping tests
    def test_map_type_python_to_python(self):
        """Test mapping between Python types."""
        # Map int to int (same type)
        assert self.mapper.map_type(int, 'python', 'python') is int
        # Map str to str (same type)
        assert self.mapper.map_type(str, 'python', 'python') is str
    
    # SQLAlchemy tests (conditional on availability)
    @pytest.mark.skipif(not has_sqlalchemy, reason="SQLAlchemy not installed")
    def test_sqlalchemy_mappings(self):
        """Test conversions between SQLAlchemy and canonical types."""
        from sqlalchemy import Integer, String, Float, Boolean
        
        # SQLAlchemy to canonical
        assert self.mapper.get_canonical_type(Integer, 'sqlalchemy') == 'integer'
        assert self.mapper.get_canonical_type(String, 'sqlalchemy') == 'string'
        assert self.mapper.get_canonical_type(Float, 'sqlalchemy') == 'float'
        assert self.mapper.get_canonical_type(Boolean, 'sqlalchemy') == 'boolean'
        
        # Canonical to SQLAlchemy
        assert self.mapper.get_library_type('integer', 'sqlalchemy') is Integer
        assert self.mapper.get_library_type('string', 'sqlalchemy') is String
        assert self.mapper.get_library_type('float', 'sqlalchemy') is Float
        assert self.mapper.get_library_type('boolean', 'sqlalchemy') is Boolean
    
    @pytest.mark.skipif(not has_sqlalchemy, reason="SQLAlchemy not installed")
    def test_cross_mapping_python_sqlalchemy(self):
        """Test mapping between Python and SQLAlchemy types."""
        from sqlalchemy import Integer, String, Float, Boolean
        
        # Python to SQLAlchemy
        assert self.mapper.map_type(int, 'python', 'sqlalchemy') is Integer
        assert self.mapper.map_type(str, 'python', 'sqlalchemy') is String
        assert self.mapper.map_type(float, 'python', 'sqlalchemy') is Float
        assert self.mapper.map_type(bool, 'python', 'sqlalchemy') is Boolean
        
        # SQLAlchemy to Python
        assert self.mapper.map_type(Integer, 'sqlalchemy', 'python') is int
        assert self.mapper.map_type(String, 'sqlalchemy', 'python') is str
        assert self.mapper.map_type(Float, 'sqlalchemy', 'python') is float
        assert self.mapper.map_type(Boolean, 'sqlalchemy', 'python') is bool
    
    # Pandas tests (conditional on availability)
    @pytest.mark.skipif(not has_pandas, reason="Pandas not installed")
    def test_pandas_mappings(self):
        """Test conversions between Pandas and canonical types."""
        import pandas as pd
        
        # Test pandas type mapping
        assert self.mapper.get_canonical_type(pd.Int64Dtype(), 'pandas') == 'integer'
        assert self.mapper.get_canonical_type(pd.StringDtype(), 'pandas') == 'string'
        assert self.mapper.get_canonical_type('float64', 'pandas') == 'float'
        assert self.mapper.get_canonical_type('bool', 'pandas') == 'boolean'
        
        # Canonical to Pandas
        pd_int_type = self.mapper.get_library_type('integer', 'pandas')
        assert isinstance(pd_int_type, pd.api.extensions.ExtensionDtype)
        assert str(pd_int_type) == 'Int64'
        
        assert isinstance(self.mapper.get_library_type('string', 'pandas'), pd.api.extensions.ExtensionDtype)
        assert self.mapper.get_library_type('float', 'pandas') == 'float64'
        assert self.mapper.get_library_type('boolean', 'pandas') == 'bool'
    
    # Polars tests (conditional on availability)
    @pytest.mark.skipif(not has_polars, reason="Polars not installed")
    def test_polars_mappings(self):
        """Test conversions between Polars and canonical types."""
        import polars as pl
        
        # Polars to canonical
        assert self.mapper.get_canonical_type(pl.Int64, 'polars') == 'integer'
        assert self.mapper.get_canonical_type(pl.Utf8, 'polars') == 'string'
        assert self.mapper.get_canonical_type(pl.Float64, 'polars') == 'float'
        assert self.mapper.get_canonical_type(pl.Boolean, 'polars') == 'boolean'
        
        # Canonical to Polars
        assert self.mapper.get_library_type('integer', 'polars') is pl.Int64
        assert self.mapper.get_library_type('string', 'polars') is pl.Utf8
        assert self.mapper.get_library_type('float', 'polars') is pl.Float64
        assert self.mapper.get_library_type('boolean', 'polars') is pl.Boolean
    
    # Cross-library mapping between all supported libraries
    @pytest.mark.skipif(not (has_sqlalchemy and has_pandas and has_polars), 
                        reason="One or more required libraries not installed")
    def test_cross_library_mapping(self):
        """Test mapping between all supported libraries when available."""
        from sqlalchemy import Integer
        import pandas as pd
        import polars as pl
        
        # SQLAlchemy -> Pandas
        pd_type = self.mapper.map_type(Integer, 'sqlalchemy', 'pandas')
        assert isinstance(pd_type, pd.api.extensions.ExtensionDtype)
        assert str(pd_type) == 'Int64'
        
        # SQLAlchemy -> Polars
        assert self.mapper.map_type(Integer, 'sqlalchemy', 'polars') is pl.Int64
        
        # Pandas -> Polars
        assert self.mapper.map_type(pd.Int64Dtype(), 'pandas', 'polars') is pl.Int64
        
        # Polars -> Pandas
        pd_type = self.mapper.map_type(pl.Int64, 'polars', 'pandas')
        assert isinstance(pd_type, pd.api.extensions.ExtensionDtype)
        assert str(pd_type) == 'Int64' 
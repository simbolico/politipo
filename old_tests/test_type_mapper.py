import pytest
from politipo import TypeMapper
import sys
import datetime
import decimal
from unittest.mock import patch

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

try:
    import pandera as pa
    has_pandera = True
except ImportError:
    has_pandera = False


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
        """Test mapping types from Python to Python."""
        assert self.mapper.map_type(int, 'python', 'python') is int
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
        assert self.mapper.map_type(int, 'sqlalchemy', 'python') is Integer
        assert self.mapper.map_type(str, 'sqlalchemy', 'python') is String
        assert self.mapper.map_type(float, 'sqlalchemy', 'python') is Float
        assert self.mapper.map_type(bool, 'sqlalchemy', 'python') is Boolean
        
        # SQLAlchemy to Python
        assert self.mapper.map_type(Integer, 'python', 'sqlalchemy') is int
        assert self.mapper.map_type(String, 'python', 'sqlalchemy') is str
        assert self.mapper.map_type(Float, 'python', 'sqlalchemy') is float
        assert self.mapper.map_type(Boolean, 'python', 'sqlalchemy') is bool
    
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
        pd_type = self.mapper.map_type(Integer, 'pandas', 'sqlalchemy')
        assert isinstance(pd_type, pd.api.extensions.ExtensionDtype)
        assert str(pd_type) == 'Int64'
        
        # SQLAlchemy -> Polars
        assert self.mapper.map_type(Integer, 'polars', 'sqlalchemy') is pl.Int64
        
        # Pandas -> Polars
        assert self.mapper.map_type(pd.Int64Dtype(), 'polars', 'pandas') is pl.Int64
        
        # Polars -> Pandas
        pd_type = self.mapper.map_type(pl.Int64, 'pandas', 'polars')
        assert isinstance(pd_type, pd.api.extensions.ExtensionDtype)
        assert str(pd_type) == 'Int64'

    # Tests from test_type_mapper_detect.py
    def test_detect_library_python_types(self):
        """Test auto-detection of Python types."""
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

    @pytest.mark.skipif(not has_sqlalchemy, reason="SQLAlchemy not installed")
    def test_detect_library_sqlalchemy_types(self):
        """Test auto-detection of SQLAlchemy types."""
        mapper = TypeMapper()
        from sqlalchemy import Integer, String, Float, Boolean
        
        assert mapper.detect_library(Integer) == 'sqlalchemy'
        assert mapper.detect_library(String) == 'sqlalchemy'
        assert mapper.detect_library(Float) == 'sqlalchemy'
        assert mapper.detect_library(Boolean) == 'sqlalchemy'

    @pytest.mark.skipif(not has_pandas, reason="Pandas not installed")
    def test_detect_library_pandas_types(self):
        """Test auto-detection of Pandas types."""
        mapper = TypeMapper()
        import pandas as pd
        
        assert mapper.detect_library(pd.Int64Dtype()) == 'pandas'
        assert mapper.detect_library(pd.StringDtype()) == 'pandas'
        assert mapper.detect_library('float64') == 'pandas'
        assert mapper.detect_library('bool') == 'pandas'

    @pytest.mark.skipif(not has_polars, reason="Polars not installed")
    def test_detect_library_polars_types(self):
        """Test auto-detection of Polars types."""
        mapper = TypeMapper()
        import polars as pl
        
        assert mapper.detect_library(pl.Int64) == 'polars'
        assert mapper.detect_library(pl.Utf8) == 'polars'
        assert mapper.detect_library(pl.Float64) == 'polars'
        assert mapper.detect_library(pl.Boolean) == 'polars'

    def test_auto_detection_in_map_type(self):
        """Test auto-detection feature in map_type function."""
        mapper = TypeMapper()
        
        # Test Python to SQLAlchemy (auto-detection)
        if has_sqlalchemy:
            from sqlalchemy import Integer
            result = mapper.map_type(int, 'sqlalchemy')
            assert result is Integer
        
        # Test SQLAlchemy to Polars (auto-detection)
        if has_sqlalchemy and has_polars:
            from sqlalchemy import Integer
            import polars as pl
            result = mapper.map_type(Integer, 'polars')
            assert result is pl.Int64

    def test_invalid_type_detection(self):
        """Test handling of types that cannot be automatically detected."""
        mapper = TypeMapper()
        
        # Create a custom class that doesn't match any library
        class CustomType:
            pass
            
        with pytest.raises(ValueError, match="Could not automatically determine library"):
            mapper.detect_library(CustomType)
            
        with pytest.raises(ValueError, match="Could not automatically determine library"):
            mapper.map_type(CustomType, 'python')

    # Pandera integration tests
    @pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
    def test_type_mapper_pandera_to_canonical(self):
        """Test mapping Pandera types to canonical types."""
        import pandera as pa
        mapper = TypeMapper()
        
        # Test mapping Pandera types to canonical types
        assert mapper.get_canonical_type(pa.Int, 'pandera') == 'integer'
        assert mapper.get_canonical_type(pa.String, 'pandera') == 'string'
        assert mapper.get_canonical_type(pa.Float, 'pandera') == 'float'
        assert mapper.get_canonical_type(pa.Bool, 'pandera') == 'boolean'
        assert mapper.get_canonical_type(pa.Date, 'pandera') == 'date'
        assert mapper.get_canonical_type(pa.DateTime, 'pandera') == 'datetime'
        assert mapper.get_canonical_type(pa.Decimal, 'pandera') == 'decimal'
        
        # Test with pa.Int() instance
        int_type = pa.Int()
        assert mapper.get_canonical_type(int_type, 'pandera') == 'integer'

    @pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
    def test_type_mapper_canonical_to_pandera(self):
        """Test mapping canonical types to Pandera types."""
        import pandera as pa
        mapper = TypeMapper()
        
        # Test mapping canonical types to Pandera types
        assert mapper.get_library_type('integer', 'pandera') is pa.Int
        assert mapper.get_library_type('string', 'pandera') is pa.String
        assert mapper.get_library_type('float', 'pandera') is pa.Float
        assert mapper.get_library_type('boolean', 'pandera') is pa.Bool
        assert mapper.get_library_type('date', 'pandera') is pa.Date
        assert mapper.get_library_type('datetime', 'pandera') is pa.DateTime
        assert mapper.get_library_type('decimal', 'pandera') is pa.Decimal

    @pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
    def test_type_mapper_detect_pandera_types(self):
        """Test auto-detecting Pandera types."""
        import pandera as pa
        mapper = TypeMapper()
        
        # Test auto-detection with class types
        assert mapper.detect_library(pa.Int) == 'pandera'
        assert mapper.detect_library(pa.String) == 'pandera'
        
        # Test auto-detection with instance types
        assert mapper.detect_library(pa.Int()) == 'pandera'
        assert mapper.detect_library(pa.String()) == 'pandera'

    @pytest.mark.skipif(not (has_pandera and has_sqlalchemy and has_pandas and has_polars), 
                        reason="One or more required libraries not installed")
    def test_map_type_between_libraries_with_pandera(self):
        """Test mapping types between different libraries including Pandera."""
        import pandera as pa
        from sqlalchemy import Integer, String
        import pandas as pd
        import polars as pl
        
        mapper = TypeMapper()
        
        # Pandera to SQLAlchemy
        sqlalchemy_int = mapper.map_type(pa.Int, to_library='sqlalchemy')
        assert sqlalchemy_int is Integer
        
        # SQLAlchemy to Pandera
        pandera_string = mapper.map_type(String, to_library='pandera')
        assert pandera_string is pa.String
        
        # Pandera to Pandas
        pandas_int = mapper.map_type(pa.Int, to_library='pandas')
        assert isinstance(pandas_int, pd.api.extensions.ExtensionDtype)
        assert str(pandas_int) == 'Int64'
        
        # Pandera to Polars
        polars_string = mapper.map_type(pa.String, to_library='polars')
        assert polars_string is pl.Utf8

    @pytest.mark.skipif(True, reason="Moved to test_type_converter.py")
    def test_type_converter_pandera_schema_validation(self):
        pass

    @pytest.mark.skipif(not has_pandera, reason="Pandera not installed")
    def test_missing_pandera_error(self):
        """Test that appropriate error is raised when Pandera is not available."""
        with patch.dict(sys.modules, {'pandera': None}):
            mapper = TypeMapper()
            
            # Should raise ImportError when trying to use Pandera functionality
            with pytest.raises(ImportError):
                mapper.get_canonical_type(object(), 'pandera')

            with pytest.raises(ImportError):
                mapper.get_library_type('integer', 'pandera') 

import pytest
from politipo import TypeMapper
import sys
import datetime
import decimal
from typing import List, Optional, Dict, Tuple, Any
from unittest.mock import patch

# Check if Pydantic is installed
try:
    import pydantic
    from pydantic import BaseModel, Field, EmailStr, SecretStr
    has_pydantic = True
    
    # Check Pydantic version
    is_v2 = hasattr(BaseModel, "model_dump")
    if is_v2:
        from typing import Annotated
except ImportError:
    has_pydantic = False
    is_v2 = False


def find_in_tuple_dict(tuple_dict: Tuple, key: str) -> bool:
    """Helper function to find a key in a tuple-formatted dictionary."""
    if not isinstance(tuple_dict, tuple):
        return False
    return any(item[0] == key for item in tuple_dict)

def get_from_tuple_dict(tuple_dict: Tuple, key: str) -> Any:
    """Helper function to get a value from a tuple-formatted dictionary."""
    if not isinstance(tuple_dict, tuple):
        return None
    for item in tuple_dict:
        if item[0] == key:
            return item[1]
    return None


@pytest.mark.skipif(not has_pydantic, reason="Pydantic not installed")
class TestPydanticTypeMapper:
    """Test suite for Pydantic-specific TypeMapper functionality."""
    
    def setup_method(self):
        """Set up TypeMapper instance before each test."""
        self.mapper = TypeMapper()
    
    def test_detect_library_pydantic_model(self):
        """Test auto-detection of Pydantic model classes."""
        class UserModel(BaseModel):
            id: int
            name: str
        
        assert self.mapper.detect_library(UserModel) == 'pydantic'

    def test_basic_type_mapping(self):
        """Test basic type mapping between Pydantic and canonical types."""
        # Python → Canonical
        assert self.mapper.get_canonical_type(int, 'pydantic') == 'integer'
        assert self.mapper.get_canonical_type(str, 'pydantic') == 'string'
        assert self.mapper.get_canonical_type(float, 'pydantic') == 'float'
        assert self.mapper.get_canonical_type(bool, 'pydantic') == 'boolean'
        assert self.mapper.get_canonical_type(datetime.date, 'pydantic') == 'date'
        
        # Canonical → Pydantic
        assert self.mapper.get_library_type('integer', 'pydantic') is int
        assert self.mapper.get_library_type('string', 'pydantic') is str
        assert self.mapper.get_library_type('float', 'pydantic') is float
        assert self.mapper.get_library_type('boolean', 'pydantic') is bool
        assert self.mapper.get_library_type('date', 'pydantic') is datetime.date

    def test_list_type_mapping(self):
        """Test list type mapping for Pydantic."""
        from typing import List
        
        # Python List → Canonical
        list_type = List[int]
        canonical = self.mapper.get_canonical_type(list_type, 'pydantic')
        assert canonical == ('list', 'integer')
        
        # Canonical → Pydantic List
        mapped_type = self.mapper.get_library_type(canonical, 'pydantic')
        from typing import get_origin, get_args
        assert get_origin(mapped_type) is list
        assert get_args(mapped_type)[0] is int

    def test_pydantic_model_detection(self):
        """Test detection of Pydantic model."""
        class User(BaseModel):
            id: int
            name: str
            
        assert self.mapper.detect_library(User) == 'pydantic'
    
    @pytest.mark.skipif(not is_v2, reason="Requires Pydantic v2")
    def test_pydantic_v2_special_types(self):
        """Test Pydantic v2 special types."""
        # Import the specific types
        from pydantic import EmailStr, SecretStr
        
        # EmailStr mapping
        assert self.mapper.get_canonical_type(EmailStr, 'pydantic') == 'email'
        assert self.mapper.get_library_type('email', 'pydantic') is EmailStr
        
        # SecretStr mapping
        assert self.mapper.get_canonical_type(SecretStr, 'pydantic') == 'secret'
        assert self.mapper.get_library_type('secret', 'pydantic') is SecretStr
    
    @pytest.mark.skipif(is_v2, reason="Requires Pydantic v1")
    def test_pydantic_v1_constrained_types(self):
        """Test Pydantic v1 constrained types."""
        # We need to import these specifically for v1
        from pydantic import conint, confloat, constr, PositiveInt, NegativeInt
        
        # Test positive int (result is a tuple with constraints)
        canonical = self.mapper.get_canonical_type(PositiveInt, 'pydantic')
        assert canonical[0] == 'constrained'
        assert canonical[1] == 'integer'
        assert isinstance(canonical[2], tuple)
        
        # Test constrained int (result is a tuple with constraints)
        constrained_int = conint(ge=1, le=100)
        canonical = self.mapper.get_canonical_type(constrained_int, 'pydantic')
        assert canonical[0] == 'constrained'
        assert canonical[1] == 'integer'
        assert isinstance(canonical[2], tuple)
        
        # Convert back to Pydantic
        mapped_type = self.mapper.get_library_type(canonical, 'pydantic')
        assert mapped_type.__origin__ == conint

    def test_pydantic_model_schema_extraction(self):
        """Test extracting schema from Pydantic model."""
        class UserProfile(BaseModel):
            id: int
            name: str
            email: Optional[str] = None
            is_active: bool = True
            
        # Test conversion to canonical type
        canonical = self.mapper.get_canonical_type(UserProfile, 'pydantic')
        assert canonical[0] == 'pydantic_model'
        assert canonical[1] == 'UserProfile'
        assert isinstance(canonical[2], tuple)
        
        # Verify schema contents using helper functions for tuple-formatted dicts
        schema_tuple = canonical[2]
        assert find_in_tuple_dict(schema_tuple, 'properties')
        properties = get_from_tuple_dict(schema_tuple, 'properties')
        assert find_in_tuple_dict(properties, 'id')
        assert find_in_tuple_dict(properties, 'name')
        assert find_in_tuple_dict(properties, 'email')
        
    @pytest.mark.skip(reason="Model reconstruction needs more work")
    def test_pydantic_model_reconstruction(self):
        """Test reconstructing Pydantic model from schema."""
        # Define original model
        class User(BaseModel):
            id: int
            name: str
            email: Optional[str] = None
            
        # Convert to canonical and back
        canonical = self.mapper.get_canonical_type(User, 'pydantic')
        reconstructed_model = self.mapper.get_library_type(canonical, 'pydantic')
        
        # Check the reconstructed model
        assert reconstructed_model.__name__ == 'User'
        
        # Create instances to verify
        original = User(id=1, name="Alice")
        
        # We can't directly compare classes, so we create an instance and check fields
        reconstructed = reconstructed_model(id=1, name="Alice")
        assert reconstructed.id == 1
        assert reconstructed.name == "Alice"
        
        # Test with email
        reconstructed_with_email = reconstructed_model(id=2, name="Bob", email="bob@example.com")
        assert reconstructed_with_email.email == "bob@example.com"

    def test_version_detection(self):
        """Test Pydantic version detection."""
        version = self.mapper._get_pydantic_version()
        if is_v2:
            assert version == 2
        else:
            assert version == 1
            
    def test_missing_pydantic(self):
        """Test behavior when Pydantic is not available."""
        with patch.dict(sys.modules, {'pydantic': None}):
            mapper = TypeMapper()
            with pytest.raises(ImportError):
                mapper._get_pydantic_version()
                
    def test_cross_library_mapping(self):
        """Test mapping Pydantic types to other libraries."""
        # Pydantic to SQLAlchemy
        mapped = self.mapper.map_type(int, 'sqlalchemy', 'pydantic')
        from sqlalchemy import Integer
        assert mapped is Integer
        
        # Pydantic to Pandas
        mapped = self.mapper.map_type(int, 'pandas', 'pydantic')
        import pandas as pd
        assert isinstance(mapped, pd.api.extensions.ExtensionDtype)


@pytest.mark.skipif(not has_pydantic, reason="Pydantic not installed")
class TestPydanticComplexMapping:
    """Test more complex Pydantic mapping scenarios."""
    
    def setup_method(self):
        """Set up TypeMapper instance before each test."""
        self.mapper = TypeMapper()
    
    def test_nested_model_mapping(self):
        """Test mapping with nested Pydantic models."""
        class Address(BaseModel):
            street: str
            city: str
            
        class User(BaseModel):
            id: int
            name: str
            address: Address
            
        # Map the model to canonical
        canonical = self.mapper.get_canonical_type(User, 'pydantic')
        assert canonical[0] == 'pydantic_model'
        
        # Check schema contains nested model info using helper functions
        schema_tuple = canonical[2]
        assert find_in_tuple_dict(schema_tuple, 'properties')
        properties = get_from_tuple_dict(schema_tuple, 'properties')
        assert find_in_tuple_dict(properties, 'address')
        
        # Skip model reconstruction test since it's complex
    
    @pytest.mark.skip(reason="Advanced constraint handling needs work")
    def test_annotated_field_with_constraints(self):
        """Test mapping Annotated fields with constraints in Pydantic v2."""
        from typing import Annotated
        
        class Product(BaseModel):
            id: int
            name: str
            price: Annotated[float, Field(gt=0, description="Product price")]
            
        # Map to canonical
        canonical = self.mapper.get_canonical_type(Product, 'pydantic')
        
        # The canonical representation should be a tuple with constraints
        assert canonical[0] == 'pydantic_model'
        assert canonical[1] == 'Product'
        
        # Check schema contains price field with helper functions
        schema_tuple = canonical[2]
        assert find_in_tuple_dict(schema_tuple, 'properties')
        properties = get_from_tuple_dict(schema_tuple, 'properties')
        assert find_in_tuple_dict(properties, 'price') 
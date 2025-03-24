import unittest
from unittest.mock import patch
import sys
import pytest

# Import the Politipo components
from politipo import TypeMapper, TypeConverter


class TestPanderaIntegration(unittest.TestCase):

    def setUp(self):
        # Skip the tests if Pandera is not installed
        try:
            import pandera as pa
            import pandas as pd
            import polars as pl
        except ImportError:
            pytest.skip("Pandera, Pandas, or Polars is not installed")

    def test_type_mapper_pandera_to_canonical(self):
        """Test mapping Pandera types to canonical types."""
        import pandera as pa
        mapper = TypeMapper()
        
        # Test mapping Pandera types to canonical types
        self.assertEqual(mapper.get_canonical_type(pa.Int, 'pandera'), 'integer')
        self.assertEqual(mapper.get_canonical_type(pa.String, 'pandera'), 'string')
        self.assertEqual(mapper.get_canonical_type(pa.Float, 'pandera'), 'float')
        self.assertEqual(mapper.get_canonical_type(pa.Bool, 'pandera'), 'boolean')
        self.assertEqual(mapper.get_canonical_type(pa.Date, 'pandera'), 'date')
        self.assertEqual(mapper.get_canonical_type(pa.DateTime, 'pandera'), 'datetime')
        self.assertEqual(mapper.get_canonical_type(pa.Decimal, 'pandera'), 'decimal')
        
        # Test with pa.Int() instance
        int_type = pa.Int()
        self.assertEqual(mapper.get_canonical_type(int_type, 'pandera'), 'integer')

    def test_type_mapper_canonical_to_pandera(self):
        """Test mapping canonical types to Pandera types."""
        import pandera as pa
        mapper = TypeMapper()
        
        # Test mapping canonical types to Pandera types
        self.assertIs(mapper.get_library_type('integer', 'pandera'), pa.Int)
        self.assertIs(mapper.get_library_type('string', 'pandera'), pa.String)
        self.assertIs(mapper.get_library_type('float', 'pandera'), pa.Float)
        self.assertIs(mapper.get_library_type('boolean', 'pandera'), pa.Bool)
        self.assertIs(mapper.get_library_type('date', 'pandera'), pa.Date)
        self.assertIs(mapper.get_library_type('datetime', 'pandera'), pa.DateTime)
        self.assertIs(mapper.get_library_type('decimal', 'pandera'), pa.Decimal)

    def test_type_mapper_detect_pandera_types(self):
        """Test auto-detecting Pandera types."""
        import pandera as pa
        mapper = TypeMapper()
        
        # Test auto-detection with class types
        self.assertEqual(mapper.detect_library(pa.Int), 'pandera')
        self.assertEqual(mapper.detect_library(pa.String), 'pandera')
        
        # Test auto-detection with instance types
        self.assertEqual(mapper.detect_library(pa.Int()), 'pandera')
        self.assertEqual(mapper.detect_library(pa.String()), 'pandera')

    def test_map_type_between_libraries(self):
        """Test mapping types between different libraries."""
        import pandera as pa
        from sqlalchemy import Integer, String
        import pandas as pd
        import polars as pl
        
        mapper = TypeMapper()
        
        # Pandera to SQLAlchemy
        sqlalchemy_int = mapper.map_type(pa.Int, to_library='sqlalchemy')
        self.assertIs(sqlalchemy_int, Integer)
        
        # SQLAlchemy to Pandera
        pandera_string = mapper.map_type(String, to_library='pandera')
        self.assertIs(pandera_string, pa.String)
        
        # Pandera to Pandas
        pandas_int = mapper.map_type(pa.Int, to_library='pandas')
        self.assertEqual(pandas_int, pd.Int64Dtype())
        
        # Pandera to Polars
        polars_string = mapper.map_type(pa.String, to_library='polars')
        self.assertIs(polars_string, pl.Utf8)

    def test_type_converter_pandera_schema_validation(self):
        """Test validating DataFrames with Pandera schemas using TypeConverter."""
        import pandera as pa
        import pandas as pd
        
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
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 3)
        
        # Test failed validation
        with self.assertRaises(Exception):  # Should raise a SchemaError, but we'll catch any exception for compatibility
            converter.convert(invalid_df)

    @pytest.mark.skip(reason="Pandera does not directly support Polars DataFrames yet")
    def test_type_converter_polars_with_pandera(self):
        """Test validating Polars DataFrames with Pandera schemas."""
        import pandera as pa
        import polars as pl
        
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
        self.assertIsInstance(result_df, pl.DataFrame)
        self.assertEqual(len(result_df), 3)

    @patch('pandera.DataFrameSchema.validate')
    def test_convert_methods_with_pandera(self, mock_validate):
        """Test that convert_single and convert_collection work with Pandera schemas."""
        import pandera as pa
        import pandas as pd
        
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
        self.assertEqual(len(result_single), 1)
        mock_validate.assert_called_once_with(df)
        
        # Reset mock
        mock_validate.reset_mock()
        
        # Test convert_collection
        result_collection = converter.convert_collection(df)
        self.assertEqual(len(result_collection), 1)
        mock_validate.assert_called_once_with(df)

    def test_missing_pandera_error(self):
        """Test that appropriate error is raised when Pandera is not available."""
        with patch.dict(sys.modules, {'pandera': None}):
            mapper = TypeMapper()
            
            # Should raise ImportError when trying to use Pandera functionality
            with self.assertRaises(ImportError):
                mapper.get_canonical_type(object(), 'pandera')

            with self.assertRaises(ImportError):
                mapper.get_library_type('integer', 'pandera')
            
            # For TypeConverter, we need to create a dummy schema-like object
            class FakeSchema:
                def __init__(self):
                    self.validate = lambda x: x
                    self.columns = {}
            
            converter = TypeConverter(from_type=dict, to_type=FakeSchema())
            
            with self.assertRaises(ImportError):
                converter.convert({"id": 1, "name": "Alice"})


if __name__ == '__main__':
    unittest.main() 
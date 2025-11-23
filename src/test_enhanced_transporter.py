# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "pydantic>=2.5",
#     "sqlalchemy>=2.0",
#     "pyarrow>=15.0",
# ]
# ///

import datetime
import json
import uuid
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

import politipo as pt


# Test models for enhanced features
class EnhancedModel(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime.datetime
    created_date: datetime.date
    created_time: datetime.time
    binary_data: bytes
    metadata: dict
    tags: list[str]
    embedding_float32: Annotated[list[float], "vector", 3]
    embedding_int32: Annotated[list[int], "vector", 2, "int32"]


class SimpleModel(BaseModel):
    name: str
    value: int


# --- Test New TypeSpec Classes ---


def test_binary_spec():
    """Test BinarySpec functionality"""
    spec = pt.BinarySpec()

    # Test SQL types
    assert spec.sql == pt.sa.LargeBinary if pt.SQL_AVAILABLE else "BLOB"

    # Test Arrow type
    if pt.ARROW_AVAILABLE:
        assert spec.arrow is not None

    # Test serialization
    test_bytes = b"test data"
    assert spec.serialize(test_bytes) == test_bytes
    assert spec.serialize("string") == b"string"


def test_time_spec():
    """Test TimeSpec functionality"""
    spec = pt.TimeSpec()

    # Test SQL types
    assert spec.sql == pt.sa.Time if pt.SQL_AVAILABLE else "TIME"
    assert spec.kuzu == "TIME"
    assert spec.duckdb == "TIME"

    # Test Arrow type
    if pt.ARROW_AVAILABLE:
        assert spec.arrow is not None


def test_date_only_spec():
    """Test DateOnlySpec functionality"""
    spec = pt.DateOnlySpec()

    # Test SQL types
    assert spec.sql == pt.sa.Date if pt.SQL_AVAILABLE else "DATE"
    assert spec.kuzu == "DATE"
    assert spec.duckdb == "DATE"

    # Test serialization
    test_date = datetime.date(2023, 1, 1)
    assert spec.serialize(test_date) == test_date


def test_json_spec():
    """Test JSONSpec functionality"""
    spec = pt.JSONSpec()

    # Test SQL types
    assert spec.sql == pt.sa.JSON if pt.SQL_AVAILABLE else "JSON"

    # Test dialect-specific types
    if pt.SQL_AVAILABLE:
        assert isinstance(spec.to_sqlalchemy_type("postgresql"), pt.sa.JSON)
        assert isinstance(spec.to_sqlalchemy_type("sqlite"), pt.sa.JSON)

    # Test serialization
    test_dict = {"key": "value", "number": 42}
    serialized = spec.serialize(test_dict)
    assert json.loads(serialized) == test_dict


def test_enhanced_vector_spec():
    """Test enhanced VectorSpec with different element types"""
    # Test float32 vector (default)
    float_spec = pt.VectorSpec(4, "float32")
    assert float_spec.element_type == "float32"

    if pt.ARROW_AVAILABLE:
        arrow_type = float_spec.arrow
        assert arrow_type is not None

    # Test int32 vector
    int_spec = pt.VectorSpec(4, "int32")
    assert int_spec.element_type == "int32"

    # Test serialization with type conversion
    float_vec = int_spec.serialize([1.5, 2.7, 3.1, 4.9])
    assert float_vec == [1, 2, 3, 4]  # Should be converted to ints

    int_vec = int_spec.serialize([1, 2, 3, 4])
    assert int_vec == [1, 2, 3, 4]


# --- Test Enhanced PolyTransporter ---


def test_polytransporter_from_model():
    """Test the from_model class method"""
    transporter = pt.PolyTransporter.from_model(SimpleModel)
    assert transporter.model_cls == SimpleModel
    assert isinstance(transporter.spec, pt.StructSpec)


def test_polytransporter_encode():
    """Test the enhanced encode method"""
    transporter = pt.PolyTransporter.from_model(EnhancedModel)

    instance = EnhancedModel(
        id=uuid.uuid4(),
        name="test",
        created_at=datetime.datetime(2023, 1, 1, 12, 0, 0),
        created_date=datetime.date(2023, 1, 1),
        created_time=datetime.time(12, 0, 0),
        binary_data=b"test",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
        embedding_float32=[0.1, 0.2, 0.3],
        embedding_int32=[1, 2],
    )

    encoded = transporter.encode(instance)
    assert encoded["name"] == "test"
    assert isinstance(encoded["id"], bytes)  # UUID should be serialized to bytes
    assert encoded["binary_data"] == b"test"
    assert isinstance(encoded["metadata"], str)  # JSON should be serialized to string


def test_polytransporter_encode_invalid_model():
    """Test encode with invalid model type"""
    transporter = pt.PolyTransporter.from_model(SimpleModel)

    with pytest.raises(ValueError, match="Instance must be of type SimpleModel"):
        transporter.encode(
            EnhancedModel(
                id=uuid.uuid4(),
                name="test",
                created_at=datetime.datetime.now(),
                created_date=datetime.date.today(),
                created_time=datetime.time(12, 0, 0),
                binary_data=b"test",
                metadata={},
                tags=[],
                embedding_float32=[0.1, 0.2, 0.3],
                embedding_int32=[1, 2],
            )
        )


def test_polytransporter_decode():
    """Test the enhanced decode method"""
    transporter = pt.PolyTransporter.from_model(SimpleModel)

    data = {"name": "test", "value": 42}
    decoded = transporter.decode(data)

    assert isinstance(decoded, SimpleModel)
    assert decoded.name == "test"
    assert decoded.value == 42


def test_polytransporter_decode_invalid_data():
    """Test decode with invalid data"""
    transporter = pt.PolyTransporter.from_model(SimpleModel)

    with pytest.raises(ValueError, match="Data must be a mapping/dict"):
        transporter.decode("not a dict")

    # Test with missing required field
    with pytest.raises(ValueError):
        transporter.decode({"name": "test"})  # Missing 'value'


def test_enhanced_arrow_schema():
    """Test enhanced arrow schema with metadata"""
    if not pt.ARROW_AVAILABLE:
        pytest.skip("PyArrow not available")

    transporter = pt.PolyTransporter.from_model(EnhancedModel)
    schema = transporter.arrow_schema

    assert schema is not None

    # Check that fields have metadata
    for field_name in schema.names:
        field = schema.field(field_name)
        metadata = field.metadata or {}

        # Should have politipo:type metadata (metadata keys are bytes in Arrow)
        assert b"politipo:type" in metadata

        # Check specific metadata for vector fields
        if field_name == "embedding_float32":
            assert metadata[b"politipo:vector_dim"] == b"3"
            assert metadata[b"politipo:vector_element_type"] == b"float32"

        if field_name == "embedding_int32":
            assert metadata[b"politipo:vector_dim"] == b"2"
            assert metadata[b"politipo:vector_element_type"] == b"int32"


def test_to_sql_ddl_method():
    """Test the new to_sql_ddl method"""
    transporter = pt.PolyTransporter.from_model(EnhancedModel)

    ddl = transporter.to_sql_ddl("test_table", "duckdb")
    assert "CREATE TABLE" in ddl
    assert "test_table" in ddl

    # Should be equivalent to generate_ddl
    ddl2 = transporter.generate_ddl("duckdb", "test_table")
    assert ddl == ddl2


# --- Test Type Resolution for New Types ---


def test_resolve_date_type():
    """Test datetime.date type resolution"""
    spec = pt._RESOLVER.resolve(datetime.date)
    assert isinstance(spec, pt.DateOnlySpec)


def test_resolve_time_type():
    """Test datetime.time type resolution"""
    spec = pt._RESOLVER.resolve(datetime.time)
    assert isinstance(spec, pt.TimeSpec)


def test_resolve_bytes_type():
    """Test bytes type resolution"""
    spec = pt._RESOLVER.resolve(bytes)
    assert isinstance(spec, pt.BinarySpec)


def test_resolve_dict_type():
    """Test dict type resolution"""
    spec = pt._RESOLVER.resolve(dict)
    assert isinstance(spec, pt.JSONSpec)


# --- Test DDL Generation with New Types ---


def test_ddl_generation_with_new_types():
    """Test DDL generation works with all new types"""
    transporter = pt.PolyTransporter.from_model(EnhancedModel)

    # Test different dialects
    dialects = ["duckdb", "sql+sqlite", "sql+postgresql"]

    for dialect in dialects:
        if "sql" in dialect and not pt.SQL_AVAILABLE:
            continue

        ddl = transporter.generate_ddl(dialect, "enhanced_table")
        assert "CREATE TABLE" in ddl
        assert "enhanced_table" in ddl

        # Should contain type definitions for all fields
        if dialect != "kuzu":  # Kùzu has different syntax
            assert "id" in ddl
            assert "name" in ddl
            assert "created_at" in ddl


# Test models with constraints
class ConstrainedModel(BaseModel):
    id: int
    name: Annotated[str, Field(min_length=2, max_length=50)]
    age: Annotated[int, Field(ge=0, le=120)]
    email: Annotated[str, Field(pattern=r".+@.+\.com")]
    score: Annotated[float, Field(gt=0.0)]
    tags: list[str]


def test_ddl_with_pydantic_constraints():
    """Test DDL generation includes constraints from Pydantic validators."""
    transporter = pt.PolyTransporter.from_model(ConstrainedModel)

    # Test DuckDB DDL (simpler to check)
    ddl_duckdb = transporter.generate_ddl("duckdb", "constrained_table")

    assert "CREATE TABLE IF NOT EXISTS constrained_table" in ddl_duckdb
    assert "CHECK (LENGTH(name) >= 2)" in ddl_duckdb
    assert "CHECK (LENGTH(name) <= 50)" in ddl_duckdb
    assert "CHECK (age >= 0)" in ddl_duckdb
    assert "CHECK (age <= 120)" in ddl_duckdb
    assert "CHECK (email ~ '.+@.+\\.com')" in ddl_duckdb
    assert "CHECK (score > 0.0)" in ddl_duckdb

    # Test PostgreSQL DDL with SQLAlchemy if available
    if pt.SQL_AVAILABLE:
        ddl_postgres = transporter.generate_ddl("sql+postgresql", "constrained_table")
        assert "CREATE TABLE" in ddl_postgres
        # SQLAlchemy will generate CHECK constraints as well
        assert "constrained_table" in ddl_postgres


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

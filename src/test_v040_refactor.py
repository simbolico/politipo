# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "pydantic>=2.5",
#     "sqlalchemy>=2.0",
# ]
# ///

"""
Tests for the v0.4.0 refactor focusing on:
1. SQL dialect-specific type handling
2. Vector type mapping (ARRAY in PostgreSQL, JSON in SQLite)
3. SQLAlchemy compiler integration
4. Dependency warnings
"""

import warnings
from typing import Annotated

import pytest
from pydantic import BaseModel

import politipo as pt


class VectorTestModel(BaseModel):
    """Test model with various types including Vector."""

    id: int
    name: str
    embedding: Annotated[list[float], "vector", 4] | None
    score: float


def test_vector_spec_dialect_aware():
    """Test that VectorSpec returns correct types per dialect."""
    vector_spec = pt.VectorSpec(dim=4)

    # PostgreSQL should use ARRAY
    if pt.SQL_AVAILABLE:
        pg_type = vector_spec.to_sqlalchemy_type("postgresql")
        assert isinstance(pg_type, pt.sa.ARRAY)

        # SQLite should use JSON (fallback)
        sqlite_type = vector_spec.to_sqlalchemy_type("sqlite")
        assert isinstance(sqlite_type, pt.sa.JSON)

    # Test without SQLAlchemy
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(pt, "SQL_AVAILABLE", False)
        fallback_type = vector_spec.to_sqlalchemy_type("sqlite")
        assert fallback_type == "FLOAT32[]"


def test_generate_ddl_postgresql_vector():
    """Test DDL generation for PostgreSQL with Vector type."""
    if not pt.SQL_AVAILABLE:
        pytest.skip("SQLAlchemy not available")

    transporter = pt.PolyTransporter(VectorTestModel)
    ddl = transporter.generate_ddl("sql+postgresql", "test_table")

    # Verify the DDL contains CREATE TABLE
    assert "CREATE TABLE" in ddl
    assert "test_table" in ddl

    # For PostgreSQL, Vector should be compiled to ARRAY type
    assert "ARRAY" in ddl or "FLOAT[]" in ddl


def test_generate_ddl_sqlite_vector():
    """Test DDL generation for SQLite with Vector type."""
    if not pt.SQL_AVAILABLE:
        pytest.skip("SQLAlchemy not available")

    transporter = pt.PolyTransporter(VectorTestModel)
    ddl = transporter.generate_ddl("sql+sqlite", "test_table")

    # Verify the DDL contains CREATE TABLE
    assert "CREATE TABLE" in ddl
    assert "test_table" in ddl

    # For SQLite, Vector should be compiled to JSON
    assert "JSON" in ddl


def test_dependency_warnings():
    """Test that missing dependencies trigger warnings."""
    with pytest.MonkeyPatch().context() as mp:
        # Simulate missing Arrow
        mp.setattr(pt, "ARROW_AVAILABLE", False)

        # Test with a model that has a vector field
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pt.PolyTransporter(VectorTestModel)

            # Should have issued a warning
            assert len(w) >= 1
            assert "high-performance types" in str(w[0].message)
            assert "pyarrow" in str(w[0].message)


def test_dependency_warnings_with_uuid():
    """Test warnings for UUID fields when Arrow is missing."""

    class UUIDModel(BaseModel):
        id: int
        uuid_value: pt.DataType.UUID

    with pytest.MonkeyPatch().context() as mp:
        # Simulate missing Arrow
        mp.setattr(pt, "ARROW_AVAILABLE", False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pt.PolyTransporter(UUIDModel)

            # Should have issued a warning
            assert len(w) >= 1
            assert "UUID" in str(w[0].message)


def test_typespec_default_to_sqlalchemy_type():
    """Test default implementation of to_sqlalchemy_type."""
    spec = pt.StringSpec()

    # Should use the sql property by default
    type_result = spec.to_sqlalchemy_type("sqlite")

    if pt.SQL_AVAILABLE:
        # Should be a callable that returns a SQLAlchemy type
        assert callable(type_result) or hasattr(type_result, "compile")
    else:
        # Should be a string fallback
        assert isinstance(type_result, str)


def test_no_dependency_warnings_for_simple_types():
    """Test that models with only simple types don't trigger warnings."""

    class SimpleModel(BaseModel):
        id: int
        name: str
        active: bool

    with pytest.MonkeyPatch().context() as mp:
        # Simulate missing Arrow
        mp.setattr(pt, "ARROW_AVAILABLE", False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pt.PolyTransporter(SimpleModel)

            # Should NOT have issued a warning
            assert len(w) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

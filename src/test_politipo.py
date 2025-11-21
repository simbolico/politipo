# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "pytest-cov",
#     "pydantic>=2.5",
#     "sqlalchemy>=2.0",
#     "pyarrow>=15.0",
#     "polars>=0.20.0",
#     "duckdb>=0.10.0",
#     "pandera>=0.18.0",
# ]
# ///

import datetime
import decimal
import enum
import sys
import uuid
from typing import Annotated
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

# Import core logic
import politipo as pt

# --- Fixtures & Setup ---


class UserType(enum.Enum):
    ADMIN = "admin"
    GUEST = "guest"


class ComplexModel(BaseModel):
    id: Annotated[uuid.UUID, FieldInfo(primary_key=True)]
    name: str
    age: int
    score: float
    is_active: bool
    # Financial Precision
    balance: Annotated[decimal.Decimal, pt.Precision(18, 4)]
    # Vector Embedding
    embedding: pt.Vector[4] | None
    # Enum
    role: UserType
    # Nested
    tags: list[str]
    # Metadata for Pandera
    email: Annotated[str, Field(pattern=r".+@.+\.com")]
    level: Annotated[int, Field(gt=0, le=100)]
    created_at: datetime.datetime


@pytest.fixture
def sample_data():
    return [
        ComplexModel(
            id=uuid.uuid4(),
            name="Alice",
            age=30,
            score=9.5,
            is_active=True,
            balance=decimal.Decimal("100.5000"),
            embedding=[0.1, 0.2, 0.3, 0.4],
            role=UserType.ADMIN,
            tags=["admin", "staff"],
            email="alice@corp.com",
            level=10,
            created_at=datetime.datetime(2023, 1, 1, 12, 0, 0),
        ),
        ComplexModel(
            id=uuid.uuid4(),
            name="Bob",
            age=25,
            score=8.0,
            is_active=False,
            balance=decimal.Decimal("50.0000"),
            embedding=None,
            role=UserType.GUEST,
            tags=[],
            email="bob@corp.com",
            level=5,
            created_at=datetime.datetime(2023, 1, 2, 12, 0, 0),
        ),
    ]


# --- 1. Resolver & Specs Tests ---


def test_resolve_primitives():
    spec = pt._RESOLVER.resolve(int)
    assert isinstance(spec, pt.IntegerSpec)
    assert spec.duckdb == "BIGINT"

    spec = pt._RESOLVER.resolve(str)
    assert isinstance(spec, pt.StringSpec)
    assert spec.kuzu == "STRING"

    spec = pt._RESOLVER.resolve(bool)
    assert isinstance(spec, pt.BooleanSpec)

    spec = pt._RESOLVER.resolve(float)
    assert isinstance(spec, pt.FloatSpec)


def test_resolve_uuid_optimization():
    """Testa se UUID usa otimização binária no Arrow e nativa no DuckDB"""
    spec = pt._RESOLVER.resolve(uuid.UUID)
    assert isinstance(spec, pt.UUIDSpec)
    assert spec.duckdb == "UUID"
    # Arrow must use binary(16) for memory efficiency
    if pt.ARROW_AVAILABLE:
        import pyarrow as pa

        assert spec.arrow == pa.binary(16)

    # Serialization check
    uid = uuid.uuid4()
    assert spec.serialize(uid) == uid.bytes
    assert spec.serialize(None) is None
    assert spec.serialize("not-uuid") == "not-uuid"  # Fallback behavior


def test_resolve_decimal_precision():
    """Testa se Precision metadata é respeitado"""
    MyDec = Annotated[decimal.Decimal, pt.Precision(38, 18)]
    spec = pt._RESOLVER.resolve(MyDec)
    assert isinstance(spec, pt.DecimalSpec)
    assert spec.p == 38
    assert spec.s == 18
    assert spec.duckdb == "DECIMAL(38,18)"


def test_resolve_vector():
    """Testa se Vector[N] vira Fixed List"""
    MyVec = pt.Vector[1536]
    spec = pt._RESOLVER.resolve(MyVec)
    assert isinstance(spec, pt.VectorSpec)
    assert spec.dim == 1536
    assert spec.kuzu == "FIXED_LIST(FLOAT, 1536)"
    assert spec.duckdb == "FLOAT[1536]"


def test_resolve_enum_optimization():
    """Testa se Enum vira Dictionary Encoding no Arrow"""
    spec = pt._RESOLVER.resolve(UserType)
    assert isinstance(spec, pt.EnumSpec)
    assert spec.duckdb == "VARCHAR"  # Simples para SQL

    if pt.ARROW_AVAILABLE:
        import pyarrow as pa

        assert isinstance(spec.arrow, pa.DictionaryType)

    # Serialization check
    assert spec.serialize(UserType.ADMIN) == "admin"
    assert spec.serialize(None) is None


def test_resolve_containers():
    # List
    spec = pt._RESOLVER.resolve(list[int])
    assert isinstance(spec, pt.ListSpec)
    assert isinstance(spec.child, pt.IntegerSpec)
    assert spec.duckdb == "BIGINT[]"

    # Serialization recursive check
    assert spec.serialize([1, 2]) == [1, 2]
    assert spec.serialize(None) is None


def test_resolve_struct_nested():
    class Nested(BaseModel):
        x: int

    spec = pt._RESOLVER.resolve(Nested)
    assert isinstance(spec, pt.StructSpec)
    assert "x" in spec.fields
    assert spec.duckdb == "STRUCT(x BIGINT)"

    # Serialization check
    obj = Nested(x=10)
    assert spec.serialize(obj) == {"x": 10}
    assert spec.serialize(None) is None


def test_resolve_nullable_optional():
    spec = pt._RESOLVER.resolve(int | None)
    assert spec.nullable is True

    # Union complex fallback
    spec = pt._RESOLVER.resolve(int | str)
    assert isinstance(spec, pt.StringSpec)  # Fallback defined in logic
    assert spec.nullable is True


def test_field_info_metadata():
    """Testa detecção de Primary Key e Nullability via FieldInfo"""

    class MetaModel(BaseModel):
        pk: Annotated[int, FieldInfo(primary_key=True)]
        opt: Annotated[int, FieldInfo()]  # default required

    spec = pt._RESOLVER.resolve(MetaModel)
    assert spec.fields["pk"].is_pk is True
    assert spec.fields["opt"].is_pk is False


# --- 2. DDL Generation Tests ---


def test_generate_ddl_dialects():
    ddl_duck = pt.generate_ddl(ComplexModel, "duckdb", "users")
    assert "CREATE TABLE IF NOT EXISTS users" in ddl_duck
    assert "id UUID PRIMARY KEY" in ddl_duck
    assert "embedding FLOAT[4]" in ddl_duck

    ddl_kuzu = pt.generate_ddl(ComplexModel, "kuzu", "Users")
    assert "CREATE NODE TABLE Users" in ddl_kuzu
    assert "embedding FIXED_LIST(FLOAT, 4)" in ddl_kuzu
    assert "PRIMARY KEY (id)" in ddl_kuzu

    ddl_sql = pt.generate_ddl(ComplexModel, "sql", "users")
    assert "CREATE TABLE users" in ddl_sql
    # Check SQLAlchemy fallback string conversion
    assert "VARCHAR" in ddl_sql


# --- 3. Engine / Transporter Tests ---


@pytest.mark.skipif(not pt.ARROW_AVAILABLE, reason="Arrow not installed")
def test_to_arrow_conversion(sample_data):
    tbl = pt.to_arrow(sample_data)
    import pyarrow as pa

    assert isinstance(tbl, pa.Table)
    assert len(tbl) == 2

    # Check Binary Optimization for UUID
    id_field = tbl.schema.field("id")
    assert pa.types.is_binary(id_field.type) or pa.types.is_fixed_size_binary(id_field.type)

    # Check Enum Dictionary Encoding
    role_field = tbl.schema.field("role")
    assert isinstance(role_field.type, pa.DictionaryType)

    # Check Vector
    emb_field = tbl.schema.field("embedding")
    assert pa.types.is_list(emb_field.type) or pa.types.is_fixed_size_list(emb_field.type)

    # Check Empty Data
    empty_tbl = pt.to_arrow([])
    assert empty_tbl is None


@pytest.mark.skipif(not pt.POLARS_AVAILABLE, reason="Polars not installed")
def test_to_polars_conversion(sample_data):
    df = pt.to_polars(sample_data, validate=False)
    import polars as pl

    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert "balance" in df.columns
    # Polars doesn't support Dictionary type from Arrow directly mostly, converts to Utf8 or Cat
    # We just check data integrity
    assert df["name"][0] == "Alice"


@pytest.mark.skipif(not pt.DUCKDB_AVAILABLE, reason="DuckDB not installed")
def test_ingest_duckdb(sample_data):
    import duckdb

    con = duckdb.connect(":memory:")
    transporter = pt.PolyTransporter(ComplexModel)

    cnt = transporter.ingest_duckdb(con, "test_table", sample_data)
    assert cnt == 2

    # Verify data inside DuckDB
    res = con.sql("SELECT name, balance FROM test_table WHERE name='Alice'").fetchone()
    assert res[0] == "Alice"
    # Check if decimal precision was maintained
    assert float(res[1]) == 100.5


# --- 4. Quality Gate (Pandera) Tests ---


@pytest.mark.skipif(not pt.PANDERA_AVAILABLE, reason="Pandera not installed")
def test_pandera_schema_generation():
    schema = pt.QualityGate.generate_schema(ComplexModel, backend="pandas")
    import pandera as pa

    assert isinstance(schema, pa.DataFrameSchema)
    assert "level" in schema.columns

    # Check constraints extracted from Pydantic
    checks = schema.columns["level"].checks
    # Should have gt=0 and le=100
    assert len(checks) >= 2

    email_checks = schema.columns["email"].checks
    assert any("str_matches" in str(c) for c in email_checks)


@pytest.mark.skipif(not pt.PANDERA_AVAILABLE or not pt.POLARS_AVAILABLE, reason="Deps missing")
def test_pandera_validation_execution(sample_data):
    # Valid data
    df = pt.to_polars(sample_data, validate=True)
    assert df is not None

    # Invalid Data: bypass Pydantic validation so Pandera catches it
    bad = ComplexModel.model_construct(
        id=uuid.uuid4(),
        name="Bad",
        age=0,
        score=0,
        is_active=True,
        balance=0,
        embedding=None,
        role=UserType.GUEST,
        tags=[],
        created_at=datetime.datetime.now(),
        email="not-an-email",  # Fails pattern
        level=200,  # Fails le=100
    )
    bad_data = [bad]

    # Pandera raises SchemaError for invalid data
    import pandera as pa

    with pytest.raises(pa.errors.SchemaError):
        pt.to_polars(bad_data, validate=True)


# --- 5. Edge Cases & Missing Dependencies Simulation ---


def test_missing_arrow_mock():
    """Simulates environment without PyArrow"""
    with patch("politipo.ARROW_AVAILABLE", False):
        spec = pt.StringSpec()
        assert spec.arrow is None
        assert spec.get_arrow_field("x") is None

        # Spec that depends on arrow availability
        uuid_spec = pt.UUIDSpec()
        assert uuid_spec.arrow is None

        # Transporter should raise
        t = pt.PolyTransporter(ComplexModel)
        assert t.arrow_schema is None
        with pytest.raises(ImportError):
            t.to_arrow([])


def test_missing_polars_mock(sample_data):
    """Simulates environment without Polars"""
    with patch("politipo.POLARS_AVAILABLE", False):
        with pytest.raises(ImportError):
            pt.to_polars(sample_data)


def test_missing_duckdb_mock(sample_data):
    """Simulates environment without DuckDB"""
    with patch("politipo.DUCKDB_AVAILABLE", False):
        t = pt.PolyTransporter(ComplexModel)
        with pytest.raises(ImportError):
            t.ingest_duckdb(None, "t", sample_data)


def test_missing_pandera_mock():
    """Simulates environment without Pandera"""
    with patch("politipo.PANDERA_AVAILABLE", False):
        with pytest.raises(ImportError):
            pt.QualityGate.generate_schema(ComplexModel)


def test_missing_sql_mock():
    """Simulates environment without SQLAlchemy"""
    with patch("politipo.SQL_AVAILABLE", False):
        spec = pt.StringSpec()
        assert spec.sql == "VARCHAR"  # Fallback string

        spec = pt.ListSpec(pt.IntegerSpec())
        assert spec.sql == "JSON"  # Fallback for array


def test_invalid_model_resolution():
    """Ensures transporter fails on non-model"""
    with pytest.raises(ValueError):
        pt.PolyTransporter(int)  # Not a BaseModel


def test_resolve_unsupported_type():
    """Ensures resolver handles fallback"""

    class WeirdType:
        pass

    spec = pt._RESOLVER.resolve(WeirdType)
    assert isinstance(spec, pt.StringSpec)  # Fallback
    assert spec.nullable is True


def test_struct_spec_arrow_missing():
    """Simulate Arrow missing specifically during Struct Arrow creation"""
    with patch("politipo.ARROW_AVAILABLE", False):
        spec = pt.StructSpec({})
        assert spec.arrow is None


def test_vector_spec_arrow_missing():
    with patch("politipo.ARROW_AVAILABLE", False):
        spec = pt.VectorSpec(4)
        assert spec.arrow is None


def test_enum_spec_arrow_missing():
    with patch("politipo.ARROW_AVAILABLE", False):
        spec = pt.EnumSpec(pt.StringSpec())
        assert spec.arrow is None


def test_quality_gate_extract_empty():
    """Test extraction robust to empty metadata"""
    mock_field = MagicMock()
    mock_field.metadata = []
    c = pt.QualityGate._extract_constraints(mock_field)
    assert c == {}


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))

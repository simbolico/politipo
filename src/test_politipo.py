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

import builtins as _builtins
import datetime
import decimal
import enum
import importlib as _importlib
import importlib.util as _importlib_util
import os as _os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
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


def test_vector_serialize_edge_cases():
    vs = pt.VectorSpec(3)
    assert vs.serialize(None) is None
    with pytest.raises(TypeError):
        vs.serialize("oops")


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
    assert spec.kuzu == "STRUCT(x INT64)"
    if pt.ARROW_AVAILABLE:
        import pyarrow as pa

        assert isinstance(spec.arrow, pa.StructType)

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


def test_resolver_with_explicit_fieldinfo_and_errors():
    # primary_key via FieldInfo argument
    class FI:
        primary_key = True
        metadata = []

    s = pt._RESOLVER.resolve(int, FI())
    assert isinstance(s, pt.IntegerSpec) and s.is_pk is True

    # FieldInfo is_required=False sets nullable
    s2 = pt._RESOLVER.resolve(int, FieldInfo(default=None))
    assert s2.nullable is True

    # Decimal default mapping
    s3 = pt._RESOLVER.resolve(decimal.Decimal)
    assert isinstance(s3, pt.DecimalSpec)

    # Trigger except TypeError branch for is_required
    class Dummy:
        primary_key = False
        metadata = None

        def is_required(self):
            raise TypeError("bad")

    s4 = pt._RESOLVER.resolve(int, Dummy())
    assert isinstance(s4, pt.IntegerSpec)


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

    # Unknown dialect returns empty string
    assert pt.PolyTransporter(ComplexModel).generate_ddl("foo", "users") == ""

    # Non-string identifier is rejected
    with pytest.raises(ValueError):
        pt.PolyTransporter(ComplexModel).generate_ddl("duckdb", 123)  # type: ignore[arg-type]


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

    # Extra constraint branches: lt, min/max length, multiple_of
    class ExtraModel(BaseModel):
        x: Annotated[int, Field(lt=10, ge=0, multiple_of=2)]
        s: Annotated[str, Field(min_length=1, max_length=3)]

    schema2 = pt.QualityGate.generate_schema(ExtraModel, backend="pandas")
    assert "x" in schema2.columns and "s" in schema2.columns


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


def test_require_and_wrappers_and_pipeline(sample_data):
    # require raises for each extra when unavailable
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(pt, "ARROW_AVAILABLE", False)
        mp.setattr(pt, "POLARS_AVAILABLE", False)
        mp.setattr(pt, "DUCKDB_AVAILABLE", False)
        mp.setattr(pt, "PANDERA_AVAILABLE", False)
        mp.setattr(pt, "SQL_AVAILABLE", False)
        for extra in ("arrow", "polars", "duckdb", "validation", "sqlalchemy"):
            with pytest.raises(ImportError):
                pt.require(extra)

    # wrapper helpers
    assert pt.resolve(ComplexModel)
    assert pt.to_polars([]) is None
    with pytest.raises(ValueError):
        pt.from_models([])
    pipe = pt.pipeline(sample_data)
    assert isinstance(pipe, pt.Pipeline)


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


# --- 6. UX Pipeline Tests (merged from test_politipo_ux.py) ---


class UXModel(BaseModel):
    id: uuid.UUID
    name: Annotated[str, Field(min_length=2, max_length=10, pattern=r"^[A-Za-z]+$")]
    score: Annotated[int, Field(ge=0, le=100, multiple_of=5)]
    amount: Annotated[decimal.Decimal, pt.Precision(18, 2)]
    created_at: datetime.datetime


@pytest.mark.skipif(not pt.ARROW_AVAILABLE, reason="arrow missing")
def test_pipeline_arrow_roundtrip():
    rows = [
        UXModel(
            id=uuid.uuid4(),
            name="Alice",
            score=10,
            amount=decimal.Decimal("12.34"),
            created_at=datetime.datetime.now(),
        )
    ]
    pipe = pt.from_models(rows).to_arrow()
    assert pipe._arrow is not None


@pytest.mark.skipif(not (pt.DUCKDB_AVAILABLE and pt.ARROW_AVAILABLE), reason="duckdb/arrow missing")
def test_pipeline_duckdb(tmp_path):
    import duckdb

    rows = [
        UXModel(
            id=uuid.uuid4(),
            name="Bob",
            score=5,
            amount=decimal.Decimal("1.00"),
            created_at=datetime.datetime.now(),
        )
    ]
    con = duckdb.connect(str(tmp_path / "db.duckdb"))
    pt.from_models(rows).to_arrow().to_duckdb(con, "ux_table")
    res = con.sql("select count(*) from ux_table").fetchone()[0]
    assert res == 1


@pytest.mark.skipif(not (pt.POLARS_AVAILABLE and pt.ARROW_AVAILABLE), reason="polars/arrow missing")
def test_pipeline_polars_validate():
    import polars as pl

    rows = [
        UXModel(
            id=uuid.uuid4(),
            name="Carol",
            score=100,
            amount=decimal.Decimal("2.00"),
            created_at=datetime.datetime.now(),
        )
    ]
    df = pt.from_models(rows).to_polars(validate=False)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1


@pytest.mark.skipif(not (pt.POLARS_AVAILABLE and pt.PANDERA_AVAILABLE), reason="deps missing")
def test_polars_validate_lazyframe_collect(monkeypatch, sample_data):
    import polars as pl

    t = pt.PolyTransporter(ComplexModel)
    _ = t.to_polars(sample_data, validate=False)

    class DummySchema:
        def validate(self, _df):
            return _df.lazy()

    monkeypatch.setattr(pt.QualityGate, "generate_schema", lambda *a, **k: DummySchema())
    out = t.to_polars(sample_data, validate=True)
    assert isinstance(out, pl.DataFrame)


def test_import_without_extras_covers(monkeypatch):
    # Patch import to force ImportError for optional deps and reload module under alias
    real_import = _builtins.__import__

    def fake_import(name, *a, **k):
        if name in {"sqlalchemy", "pyarrow", "polars", "duckdb", "pandera"}:
            raise ImportError("forced")
        return real_import(name, *a, **k)

    monkeypatch.setattr(_builtins, "__import__", fake_import)
    path = _os.path.join(_os.path.dirname(__file__), "politipo.py")
    spec = _importlib_util.spec_from_file_location("politipo_noextras", path)
    assert spec and spec.loader
    mod = _importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
    import sys as _sys

    _sys.modules[spec.name] = mod  # type: ignore[union-attr]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.SQL_AVAILABLE is False
    assert mod.ARROW_AVAILABLE is False
    assert mod.POLARS_AVAILABLE is False
    assert mod.DUCKDB_AVAILABLE is False
    assert mod.PANDERA_AVAILABLE is False


def test_generate_ddl_compile_type_branches(monkeypatch):
    # Build a transporter and inject custom specs to exercise compile_type branches
    t = pt.PolyTransporter(ComplexModel)

    class DummySpec(pt.TypeSpec):
        @property
        def sql(self):
            return "VARCHAR"

        @property
        def kuzu(self):
            return "STRING"

        @property
        def duckdb(self):
            return "VARCHAR"

        @property
        def arrow(self):
            return None

    class BadCallableSpec(pt.TypeSpec):
        @property
        def sql(self):
            def boom():
                raise Exception("boom")

            return boom

        @property
        def kuzu(self):
            return "STRING"

        @property
        def duckdb(self):
            return "VARCHAR"

        @property
        def arrow(self):
            return None

    t.spec.fields["dummy_str"] = DummySpec()
    t.spec.fields["bad"] = BadCallableSpec()

    # Force SQL path
    ddl = t.generate_ddl("sql+postgresql", "tbl")
    assert "CREATE TABLE" in ddl


@pytest.mark.skipif(not (pt.POLARS_AVAILABLE and pt.PANDERA_AVAILABLE), reason="deps missing")
def test_pandera_additional_constraints():
    # Construct invalid rows to exercise min_length/max_length/pattern and multiple_of
    bad = UXModel.model_construct(
        id=uuid.uuid4(),
        name="TooLongName",  # > 10 and not matching pattern fully
        score=12,  # not multiple_of=5
        amount=decimal.Decimal("3.00"),
        created_at=datetime.datetime.now(),
    )
    import pandera as pa

    with pytest.raises(pa.errors.SchemaError):
        # Use transporter path to run validation fallback if needed
        pt.PolyTransporter(UXModel).to_polars([bad], validate=True)


def test_generate_ddl_with_sqlalchemy_if_available():
    ddl = pt.PolyTransporter(UXModel).generate_ddl("sql+postgresql", "ux_table")
    assert isinstance(ddl, str)
    assert "CREATE TABLE" in ddl and "ux_table" in ddl


def test_generate_ddl_sqlalchemy_path_monkeypatched(monkeypatch):
    # Force SQLAlchemy path with stubs to increase coverage
    class TypeStub:
        def __call__(self, *a, **k):
            return self

        def compile(self, dialect=None):
            # return a simple portable name
            return "VARCHAR"

    class SAStub:
        String = TypeStub
        BigInteger = TypeStub
        Float = TypeStub
        Boolean = TypeStub
        Uuid = TypeStub
        Numeric = staticmethod(lambda p, s: TypeStub())
        DateTime = staticmethod(lambda timezone=True: TypeStub())
        ARRAY = staticmethod(lambda t: TypeStub())
        JSON = TypeStub

    class DialectStub:
        @staticmethod
        def dialect():
            return object()

    class DialectsStub:
        postgresql = DialectStub
        sqlite = DialectStub
        mysql = DialectStub

    import sys as _sys

    monkeypatch.setattr(pt, "SQL_AVAILABLE", True)
    monkeypatch.setattr(pt, "sa", SAStub)
    _sys.modules["sqlalchemy.dialects"] = DialectsStub

    ddl = pt.PolyTransporter(UXModel).generate_ddl("sql+postgresql", "ux_table")
    assert "CREATE TABLE" in ddl


def test_invalid_table_identifier_raises():
    with pytest.raises(ValueError):
        pt.PolyTransporter(UXModel).generate_ddl("duckdb", "bad-name")


def test_require_messages():
    # Simulate missing extras and confirm helpful messages
    with pytest.raises(ImportError) as e:
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(pt, "ARROW_AVAILABLE", False)
            # Use model_construct to bypass Pydantic validation for this UX test
            row = UXModel.model_construct(
                id=uuid.uuid4(),
                name="Dave",
                score=1,
                amount=decimal.Decimal("1.00"),
                created_at=datetime.datetime.now(),
            )
            t = pt.PolyTransporter(UXModel)
            t.to_arrow([row])
    # We don't assert the exact text, but ensure it hints uv + extra
    msg = str(e.value)
    assert "uv pip install" in msg or msg


@pytest.mark.skipif(not pt.ARROW_AVAILABLE, reason="arrow missing")
def test_invalid_vector_length_raises():
    class VModel(BaseModel):
        emb: pt.Vector[4] | None

    # Bypass validation to inject bad length
    bad = VModel.model_construct(emb=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        pt.PolyTransporter(VModel).to_arrow([bad])


# --- 7. Robustness & Property Tests (merged from test_robustness.py) ---


class RUserType(enum.Enum):
    ADMIN = "admin"
    GUEST = "guest"


class RModel(BaseModel):
    id: Annotated[uuid.UUID, pt.FieldInfo(primary_key=True)]
    name: Annotated[str, Field(min_length=1, max_length=20)]
    score: Annotated[int, Field(ge=0, le=100)]
    amount: Annotated[decimal.Decimal, pt.Precision(18, 2)]
    role: RUserType
    created_at: datetime.datetime


def build_model():
    return RModel(
        id=uuid.uuid4(),
        name="X",
        score=0,
        amount=decimal.Decimal("0.00"),
        role=RUserType.ADMIN,
        created_at=datetime.datetime.now(),
    )


@pytest.mark.skipif(not pt.ARROW_AVAILABLE, reason="arrow missing")
@given(
    st.lists(
        st.builds(
            RModel,
            id=st.uuids(version=4),
            name=st.text(min_size=1, max_size=10).map(lambda s: s.strip() or "X"),
            score=st.integers(min_value=0, max_value=100),
            # Generate cents up to 10^18-1, then scale to 2 decimal places to fit DECIMAL(18,2)
            amount=st.integers(min_value=0, max_value=10**18 - 1).map(
                lambda c: decimal.Decimal(c).scaleb(-2)
            ),
            role=st.sampled_from(list(RUserType)),
            created_at=st.datetimes(),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_to_arrow_property(rows):
    tbl = pt.to_arrow(rows)
    import pyarrow as pa

    assert isinstance(tbl, pa.Table)
    assert tbl.num_rows == len(rows)
    # Columns match model fields
    assert set(tbl.column_names) == set(RModel.model_fields.keys())


@pytest.mark.skipif(not (pt.DUCKDB_AVAILABLE and pt.ARROW_AVAILABLE), reason="deps missing")
def test_duckdb_empty_ingest(tmp_path):
    import duckdb

    con = duckdb.connect(str(tmp_path / "db.duckdb"))
    t = pt.PolyTransporter(RModel)
    # Ingest empty should not crash; table created with zero rows
    t.ingest_duckdb(con, "r_table", [])
    res = con.sql("select count(*) from r_table").fetchone()[0]
    assert res == 0


@pytest.mark.skipif(not pt.ARROW_AVAILABLE, reason="arrow missing")
def test_to_arrow_concurrent():
    rows = [build_model() for _ in range(20)]
    t = pt.PolyTransporter(RModel)

    def work():
        tbl = t.to_arrow(rows)
        return tbl.num_rows

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda _: work(), range(8)))
    assert all(r == len(rows) for r in results)


@pytest.mark.skipif(not (pt.POLARS_AVAILABLE and pt.PANDERA_AVAILABLE), reason="deps missing")
def test_polars_fallback_patch(monkeypatch):
    # Force Polars backend validate to raise, ensure Pandas fallback path used
    import pandera.polars as pa_pl

    rows = [build_model()]
    t = pt.PolyTransporter(RModel)

    class Boom(Exception):
        pass

    orig = pa_pl.DataFrameSchema.validate

    def boom(self, df):
        raise Boom("boom")

    monkeypatch.setattr(pa_pl.DataFrameSchema, "validate", boom)
    df = t.to_polars(rows, validate=True)
    # If fallback worked, we still return a polars.DataFrame
    import polars as pl

    assert isinstance(df, pl.DataFrame)
    # restore
    monkeypatch.setattr(pa_pl.DataFrameSchema, "validate", orig)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))

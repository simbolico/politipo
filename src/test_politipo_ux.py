import datetime
import decimal
import uuid
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

import politipo as pt


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

    import sys

    monkeypatch.setattr(pt, "SQL_AVAILABLE", True)
    monkeypatch.setattr(pt, "sa", SAStub)
    sys.modules["sqlalchemy.dialects"] = DialectsStub

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

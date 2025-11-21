import datetime
import decimal
import enum
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, Field

import politipo as pt


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

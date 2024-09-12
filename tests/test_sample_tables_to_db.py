import pytest
from pathlib import Path
from pca_analysis.code import bin_pumps_to_db as pb
import pandas as pd
import duckdb as db
from pandas.testing import assert_frame_equal


@pytest.fixture
def sample_dirpath():
    return "./tests/test_samples"


@pytest.fixture
def sample_45(sample_dirpath: str):
    return str(Path(sample_dirpath) / "45.D")


@pytest.fixture
def sample_100(sample_dirpath: str):
    return str(Path(sample_dirpath) / "100.D")


def test_sample_dirpath(sample_dirpath):
    assert Path(sample_dirpath).exists()


@pytest.fixture
def solvcomp_45_df_path(sample_dirpath: str) -> str:
    return str(Path(sample_dirpath) / "solvcomp_45.csv")


@pytest.fixture
def timetable_45_df_path(sample_dirpath: str) -> str:
    return str(Path(sample_dirpath) / "timetable_45.csv")


def test_get_bin_pump_tables(
    sample_45: str,
    solvcomp_45_df_path: str,
    timetable_45_df_path: str,
) -> None:
    solvcomp: pd.DataFrame
    timetable: pd.DataFrame
    solvcomp, timetable = pb.get_bin_pump_tables(dpath=sample_45)

    assert isinstance(solvcomp, pd.DataFrame)
    assert isinstance(timetable, pd.DataFrame)
    assert not solvcomp.empty
    assert not timetable.empty

    outpath = Path(sample_45).parent

    solvcomp.to_csv(solvcomp_45_df_path, index=False)
    timetable.to_csv(timetable_45_df_path, index=False)


@pytest.fixture
def sample_45(
    solvcomp_45_df_path: str,
    timetable_45_df_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The samples as the result from `get_bin_pump_tables`
    """

    return (pd.read_csv(solvcomp_45_df_path), pd.read_csv(timetable_45_df_path))


@pytest.fixture
def con() -> db.DuckDBPyConnection:
    return db.connect()


@pytest.fixture
def sample_45_id() -> str:
    return "b4bfe605-8e98-4c5e-b8f0-27229e3ebd7e"



def test_sample_tables_to_db(
    sample_45: tuple[pd.DataFrame, pd.DataFrame],
    con: db.DuckDBPyConnection,
    sample_45_id: str,
) -> None:
    new_tbl_num = pb.sample_tables_to_db(sample_45, con)

    schema_name = "bin_pump"
    solvcomp_tbl = "solvcomps"
    timetabletbl = "timetables"
    id_tbl = "id"

    # check if schema exists
    assert con.execute(
        """--sql
    SELECT
        schema_name
    FROM
        duckdb_schemas
    WHERE
        schema_name = ?
    """,
        parameters=[schema_name],
    ).fetchone()

    # check if tables exist
    assert con.execute(
        """--sql
    SELECT
        table_name
    FROM
        duckdb_tables
    WHERE
        schema_name = ?
    AND
        table_name = ?
    """,
        parameters=[schema_name, solvcomp_tbl],
    )

    # check if tables exist
    assert con.execute(
        """--sql
    SELECT
        table_name
    FROM
        duckdb_tables
    WHERE
        schema_name = ?
    AND
        table_name = ?
    """,
        parameters=[schema_name, timetabletbl],
    ).fetchone()

    # check if id table exists
    assert con.execute(
        """--sql
    SELECT
        table_name
    FROM
        duckdb_tables
    WHERE
        schema_name = ?
    AND
        table_name = ?
    """,
        parameters=[schema_name, id_tbl],
    ).fetchone()

    # check that the newly created tbl_num matchces that assocaited with the id in the db
    fetched_tbl_num = con.execute(
        """--sql
    SELECT
        tbl_num
    FROM
        bin_pump.id
    WHERE
        id = $sample_id
    """,
        parameters={
            "sample_id": sample_45_id,
        },
    ).fetchone()

    assert fetched_tbl_num
    assert fetched_tbl_num[0] == new_tbl_num

    # check that the reverse is als otrue
    fetched_id = con.execute(
        """--sql
    SELECT
        id
    FROM
        bin_pump.id
    WHERE
        tbl_num = ?
    """,
        parameters=[new_tbl_num],
    ).fetchone()

    assert fetched_id
    assert fetched_id[0] == sample_45_id

    # now test that the tables are entered correctly..
    db_solvcomp = con.execute(
        """--sql
    SELECT
        tbl_num,
        idx,
        channel,
        ch1_solv,
        name_2,
        selected,
        used,
        percent
    FROM
        bin_pump.solvcomps as t
    JOIN
        bin_pump.id as idtblc
    USING
        (tbl_num)
    WHERE
        id = ?
    """,
        parameters=[sample_45_id],
    ).df()

    df1 = db_solvcomp
    df2 = sample_45[0]

    # replicate changes made in function
    column_order = df1.columns
    df2 = df2.drop("id", axis=1)
    df2["tbl_num"] = new_tbl_num
    df2 = df2[column_order]

    assert_frame_equal(df1, df2, check_dtype=False)

    # now test adding a second table.. it should increment the table number by one, add
    # the new table number and id to the bin_pump.id, then to the two other tables.

    # ..eh. Im sure it works..

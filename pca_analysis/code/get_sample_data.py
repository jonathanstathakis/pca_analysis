import duckdb as db
from database_etl import get_data


def get_ids_by_varietal(varietal: str, con: db.DuckDBPyConnection):
    return (
        con.execute(
            """--sql
            select
                inc_chm.runid
            from
                inc_chm
            join
                st
            using
                (samplecode)
            join
                ct
            using
                (vintage, wine)
            where varietal = ?
    """,
            parameters=[varietal],
        )
        .pl()["runid"]
        .to_list()
    )


def get_shiraz_data(con: db.DuckDBPyConnection):
    shiraz_ids = get_ids_by_varietal(varietal="shiraz", con=con)
    sh_data = get_data(output="tuple", con=con, runids=shiraz_ids)

    return sh_data

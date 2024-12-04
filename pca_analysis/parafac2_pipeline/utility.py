"""
misc functions
"""

import duckdb as db
import polars as pl
import plotly.express as px


def plot_imgs(
    imgs: pl.DataFrame, nm_col: str, abs_col: str, time_col: str, runid_col: str
):
    return imgs.pipe(
        lambda x: px.line_3d(
            data_frame=x,
            x=nm_col,
            z=abs_col,
            y=time_col,
            color=runid_col,
            line_group=nm_col,
            height=750,
            width=1500,
        )
    ).update_layout(margin=dict(t=50, l=50, b=50, r=50))


def write_to_db(
    signal_frame: pl.DataFrame,
    table_name: str,
    conn: db.DuckDBPyConnection,
    overwrite: bool = False,
) -> None:
    if overwrite:
        create_clause = "create or replace table"
    else:
        create_clause = "create table if not exists"
    query = f"""--sql
        {create_clause} {table_name} (
        exec_id varchar not null,
        sample int not null,
        signal varchar not null,
        wavelength int not null,
        idx int not null,
        abs float not null,
        primary key (exec_id, sample, signal, wavelength, idx)
        );

        insert or replace into {table_name}
            select
                exec_id,
                sample,
                signal,
                wavelength,
                idx,
                abs
            from
                signal_frame
        """

    conn.execute(query)

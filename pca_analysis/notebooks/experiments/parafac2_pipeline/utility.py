"""
misc functions
"""

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

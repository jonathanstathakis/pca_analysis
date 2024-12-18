from .bcorrdb import BCRCols
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


class BCorrResults:
    def viz_3d_line_plots_by_sample(self, cols=2, title=None) -> go.Figure:
        """
        TODO4: fix plot so no *underlying* lines, add input(?)
        """
        sample_slices = self.df.partition_by(
            BCRCols.SAMPLE_IDX, include_key=False, as_dict=True
        )
        num_samples = len(sample_slices)

        rows = num_samples // cols

        flat_specs = [{"type": "scatter3d"}] * rows * cols

        reshaped_specs = np.asarray(flat_specs).reshape(-1, cols)
        reshaped_specs_list = [list(x) for x in reshaped_specs]

        titles = [f"{BCRCols.SAMPLE_IDX} = {str(x[0])}" for x in sample_slices.keys()]

        # see <https://plotly.com/python/subplots/>,
        # API: <https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html>
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=reshaped_specs_list,
            horizontal_spacing=0.0001,
            vertical_spacing=0.05,
            subplot_titles=titles,
        )

        coords = gen_row_col_coords(rows, cols)

        for idx, sample in enumerate(sample_slices.values()):
            # margin = dict(b=1, t=1, l=1, r=1)
            traces = px.line_3d(
                data_frame=sample,
                x=BCRCols.TIME_IDX,
                y=BCRCols.WAVELENGTH_IDX,
                z=BCRCols.ABS,
                line_group=BCRCols.WAVELENGTH_IDX,
            ).data

            # got to write each wavelength individually to each subplot
            for trace in traces:
                fig.add_trace(trace, row=coords[idx][0], col=coords[idx][1])

        # camera <https://plot.ly/python/3d-camera-controls/>
        # camera in subplots <https://community.plotly.com/t/make-subplots-with-3d-surfaces-how-to-set-camera-scene/15137/4>
        default_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            # eye=dict(x=1.25 * eye_mult, y=1.25 * eye_mult, z=1.25 * eye_mult),
        )

        fig.update_layout(
            height=500 * rows,
            template="plotly_dark",
            #   margin=margin
            title=title,
        )

        fig.update_scenes(camera=default_camera)

        return fig

    def viz_compare_signals(self, wavelength: int) -> go.Figure:
        """return a 2d plot at a given wavelength for each sample, corrected, baseline
        and original"""

        wavelength_vals = self.df.get_column(BCRCols.WAVELENGTH_IDX).unique(
            maintain_order=True
        )

        if wavelength not in wavelength_vals:
            raise ValueError(f"{wavelength} not in {wavelength_vals}")

        filtered_df = self.df.filter(pl.col(BCRCols.WAVELENGTH_IDX) == wavelength)

        fig = px.line(
            data_frame=filtered_df,
            x=BCRCols.TIME_IDX,
            y=BCRCols.ABS,
            color=BCRCols.SIGNAL,
            facet_col=BCRCols.SAMPLE_IDX,
            facet_col_wrap=wavelength_vals.len() // 4,
            title=f"corrected and fitted baseline @ wavelength = {wavelength}",
        )

        return fig


def gen_row_col_coords(rows: int, cols: int) -> list[tuple[int, int]]:
    """generate a list of tuples where every list element is the row, containing the row and column position."""
    indexes = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            indexes.append((row, col))

    return indexes


def prepare_scatter_3d_from_df(df, x_col, y_col, z_col):
    # scatter3d API: https://plotly.com/python/reference/scatter3d/#scatter3d-line

    x = df.select(x_col).to_numpy().flatten()
    y = df.select(y_col).to_numpy().flatten()
    z = df.select(z_col).to_numpy().flatten()

    return go.Scatter3d(x=x, y=y, z=z, mode="lines")

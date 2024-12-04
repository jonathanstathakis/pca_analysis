import plotly.graph_objects as go
import plotly.express as px
import polars as pl


def _overlay_components_input_signal(signals, wavelength, runid):
    from polars import Schema, UInt32, String, Int32, Int64, Float64

    assert signals.schema == Schema(
        {
            "sample_idx": UInt32,
            "runid": String,
            "signal": String,
            "component": Int32,
            "wavelength_idx": UInt32,
            "wavelength": Int64,
            "elution_point": UInt32,
            "abs": Float64,
        }
    )

    filtered_signals = signals.filter(
        pl.col("wavelength") == wavelength, pl.col("runid") == runid
    )

    fig = go.Figure()

    input_trace = px.line(
        filtered_signals,
        x="elution_point",
        y="abs",
        color="signal",
    ).data
    fig.add_traces(input_trace).update_traces(
        line_dash="dot", selector=dict(name="input")
    ).update_layout(title="overlay components and input signal")

    return fig


class Parafac2Viz:
    def overlay_components_input_signal(self, rectified_df, wavelength, runid):
        return _overlay_components_input_signal(
            signals=rectified_df,
            wavelength=wavelength,
            runid=runid,
        )

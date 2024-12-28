from scipy import signal
import plotly.graph_objects as go
import numpy as np
from pca_analysis import xr_plotly


class FindPeaks:
    def find_peaks(self, sample, height_ratio=0.001):
        self.height = (sample.max() * height_ratio).item()
        self.peaks, self.properties = signal.find_peaks(sample, height=self.height)

        self.peaks_x = sample["time"][self.peaks].to_numpy()
        self.peaks_y = sample[self.peaks].to_numpy()

        return self

    def plot_peaks(self, sample):
        line = go.Scatter(x=sample["time"], y=sample.data.squeeze(), name=sample.name)
        labels = np.arange(1, len(self.peaks_x) + 1)
        peaks = go.Scatter(
            x=self.peaks_x,
            y=self.peaks_y,
            text=labels,
            marker=dict(color="red"),
            mode="markers+text",
            name="peaks",
            textposition="top center",
        )

        return go.Figure().add_traces([line, peaks])

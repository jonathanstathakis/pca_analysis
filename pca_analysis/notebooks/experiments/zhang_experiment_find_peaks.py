from scipy import signal
import matplotlib.pyplot as plt


class FindPeaks:
    def find_peaks(self, sample, height_ratio=0.001):
        self.height = (sample.max() * height_ratio).item()
        self.peaks, self.properties = signal.find_peaks(sample, height=self.height)

        self.peaks_x = sample["time"][self.peaks].to_numpy()
        self.peaks_y = sample[self.peaks].to_numpy()

        return self

    def plot_peaks(self, sample):
        sample.plot.line()
        plt.scatter(self.peaks_x, self.peaks_y)
        plt.xlim(0, 25)

        return self

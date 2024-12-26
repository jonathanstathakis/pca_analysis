import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from copy import deepcopy


class MyPCA:
    def __init__(self):
        self.pca = PCA()

    def run_pca(self, data):
        obj = deepcopy(self)
        obj.pca = PCA()

        obj.pca.fit_transform(data)

        return obj

    def scree_plot(self, display_percentage_cutoff=1e-3, exp_var_change_prop=0.01):
        if not hasattr(self, "pca"):
            raise RuntimeError("call `run_pca` first")

        pca_scree_plot(
            explained_variance=self.pca.explained_variance_,
            display_percentage_cutoff=display_percentage_cutoff,
            exp_var_change_prop=exp_var_change_prop,
        )


def pca_scree_plot(
    explained_variance, display_percentage_cutoff=1e-3, exp_var_change_prop=0.01
):
    """
    draw a bar scree plot of the explained variance vs number of components.
    Additionally, adds a line indicating the number of components at which the change
    in explained variance does not exceed a user-defined proportion of change, given
    as `exp_var_change_prop`.
    """
    # implementation taken from <https://www.jcchouinard.com/pca-scree-plot/>

    # scale the cusmsum to between 0 and 1 to make calculations easier
    cumsum_exp_var = np.cumsum(explained_variance)
    scaler = MinMaxScaler()
    norm_cumsum_exp_var = scaler.fit_transform(cumsum_exp_var.reshape(-1, 1))

    diff = np.diff(norm_cumsum_exp_var, axis=0)

    # crude definition of significant component cutoff as where the rate of change
    # is greater than 2% of the maximum.

    sig_component_idxs = np.where(
        diff > norm_cumsum_exp_var.max() * exp_var_change_prop
    )
    exp_var_change_prop = sig_component_idxs[0][-1]

    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance,
    )

    plt.plot(
        range(1, len(explained_variance) + 1),
        cumsum_exp_var,
        c="red",
        label="Cumulative Explained Variance",
    )

    plt.vlines(
        exp_var_change_prop,
        plt.ylim()[0],
        plt.ylim()[1],
        linestyles="dashed",
        colors="black",
        label="signif. comp. cutoff",
    )

    plt.xlabel("num. components")
    plt.ylabel("Explained Variance (eigenvalues)")
    plt.title("scree plot")

    plt.legend(loc="center right", fontsize=8)
    plt.tight_layout()

    np.diff(explained_variance)

    # limit the plot to a range of points from 0 to 0.001% of the maximum y (explained variance)

    cutoff = explained_variance.max() * display_percentage_cutoff
    plot_x_lim = np.where(explained_variance > cutoff)[0]

    xlim_start, xlim_end = plot_x_lim[0], plot_x_lim[-1]

    plt.xlim(xlim_start, xlim_end)

    print(
        f"the number of significant components determined to be {exp_var_change_prop}"
    )
    return exp_var_change_prop

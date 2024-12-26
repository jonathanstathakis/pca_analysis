import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def pca_scree_plot(
    explained_variance, display_percentage_cutoff=1e-3, sig_component_cutoff=0.01
):
    # implementation taken from <https://www.jcchouinard.com/pca-scree-plot/>

    # scale the cusmsum to between 0 and 1 to make calculations easier
    cumsum_exp_var = np.cumsum(explained_variance)
    scaler = MinMaxScaler()
    norm_cumsum_exp_var = scaler.fit_transform(cumsum_exp_var.reshape(-1, 1))

    diff = np.diff(norm_cumsum_exp_var, axis=0)

    # crude definition of significant component cutoff as where the rate of change
    # is greater than 2% of the maximum.

    sig_component_idxs = np.where(
        diff > norm_cumsum_exp_var.max() * sig_component_cutoff
    )
    sig_component_cutoff = sig_component_idxs[0][-1]

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
        sig_component_cutoff,
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
        f"the number of significant components determined to be {sig_component_cutoff}"
    )
    return sig_component_cutoff

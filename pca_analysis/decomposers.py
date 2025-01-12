import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


def pca_scree_plot(explained_variance, exp_var_change_prop=0.01):
    """
    draw a bar scree plot of the explained variance vs number of components.
    Additionally, adds a line indicating the number of components at which the change
    in explained variance does not exceed a user-defined proportion of change, given
    as `exp_var_change_prop`.

    TODO change the annotation to a vertical line.
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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(explained_variance) + 1)),
            y=explained_variance,
            line=dict(color="red"),
            name="Cumulative Explained Variance",
        )
    )

    # fig.add_annotation(
    #     x=,
    #     y=explained_variance[exp_var_change_prop],
    #     text=str(exp_var_change_prop),
    # )
    fig.add_vline(
        exp_var_change_prop, line_dash="dash", line_width=3, name="signif. comp. cutoff"
    )

    fig.update_layout(
        xaxis=dict(title="num. components", range=(0, 2 * exp_var_change_prop)),
        yaxis=dict(title="Explained Variance (eigenvalues)"),
        title="scree test for significant components",
    )
    # limit the plot to a range of points from 0 to 0.001% of the maximum y (explained variance)

    print(
        f"the number of significant components determined to be {exp_var_change_prop}"
    )
    fig.show()
    return exp_var_change_prop


class PCA(skPCA):
    def scree_plot(self, exp_var_change_prop=0.01):
        if not hasattr(self, "explained_variance_"):
            raise RuntimeError("call `fit_transform` first")

        pca_scree_plot(
            explained_variance=self.explained_variance_,
            exp_var_change_prop=exp_var_change_prop,
        )

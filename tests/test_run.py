import pytest
import duckdb as db
from pca_analysis.agilette.run import Run
import altair as alt


@pytest.fixture(scope="module")
def test_single_runid():
    return "89"


@pytest.fixture(scope="module")
def run(test_single_runid: str, testcon: db.DuckDBPyConnection) -> Run:
    return Run(con=testcon, runid=test_single_runid)


def test_run_init(run: Run) -> None:
    assert run


def test_get_img(run: Run):
    img = run.get_img()

    assert not img.is_empty()


from IPython.display import display


def test_plot_chromatogram(run: Run):
    x: alt.Chart = run.plot_chromatogram()
    # x.show()
    assert x


def test_plot_3d_line(run: Run):
    x = run.plot_line_3d()
    # x.show()
    assert x

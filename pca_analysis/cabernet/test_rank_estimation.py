import pytest
from pca_analysis.cabernet.cabernet import Cabernet
from pca_analysis.cabernet.shiraz.shiraz import Shiraz


@pytest.fixture
def test_rank_est_data(cab: Cabernet):
    return cab.isel(sample=[0, 1]).sel(mins=slice(0, 5), wavelength=slice(220, 260))


def test_corcondia(test_rank_est_data: Cabernet):
    results = test_rank_est_data.rank_estimation.corcondia(
        "input_data", rank_range=(1, 3)
    )

    print(results.diagnostic_table)
    results.diagnostic_over_rank.show()

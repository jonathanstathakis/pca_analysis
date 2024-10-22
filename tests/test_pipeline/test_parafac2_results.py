from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Parafac2Results,
)
import pytest

# TODO: fix
pytest.skip(
    allow_module_level=True, reason="have broken the input, will need to be fixed"
)


def test_pfac2results(pfac2results: Parafac2Results):
    assert pfac2results


def test_viz_recon_3d(pfac2results: Parafac2Results):
    assert pfac2results.viz_recon_3d()


def test_show_tables(pfac2results: Parafac2Results):
    assert not pfac2results._show_tables().is_empty()


def test_viz_overlay_components(pfac2results: Parafac2Results):
    assert pfac2results._viz_overlay_components_sample_wavelength(0, 0)


def test_viz_input_img_curve(pfac2results: Parafac2Results):
    assert pfac2results._viz_input_img_curve_sample_wavelength(0, 0)


def test_viz_overlay_curve_components(pfac2results: Parafac2Results):
    assert pfac2results.viz_overlay_curve_components(0, 0)


def test_viz_recon_input_overlay(pfac2results: Parafac2Results):
    assert pfac2results.viz_recon_input_overlay(0, 0)


def test_viz_recon_input_overlay_facet(pfac2results: Parafac2Results):
    assert pfac2results.viz_recon_input_overlay_facet(0, 0, "sample")


def test_computation_matches_tly(pfac2results: Parafac2Results):
    """assert that any variation between my implementation and tensorlys is insiginficant. Variation arises from calculations performed by different libraries."""
    assert pfac2results._check_computations_match_tly()

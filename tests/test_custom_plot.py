import pytest
from kalman_reconstruction.custom_plot import adjust_lightness, handler_map_alpha, ncols_nrows_from_N
from matplotlib.collections import PathCollection
from matplotlib.pyplot import Line2D

@pytest.mark.parametrize(
    "color, amount, expected",
    [
        ("red", 0.5, "#ff7f7f"),  # Increase lightness of red by 0.5
        ("#00ff00", -0.2, "#00cc00"),  # Decrease lightness of lime by 0.2
        ("invalid", 0.1, "invalid"),  # Invalid color returns the original value
    ],
)
def test_adjust_lightness(color, amount, expected):
    """
    Test case for the `adjust_lightness` function.
    """
    assert adjust_lightness(color, amount) == expected


def test_handler_map_alpha():
    """
    Test case for the `handler_map_alpha` function.
    """
    handler_map = handler_map_alpha()
    assert PathCollection in handler_map
    assert Line2D in handler_map


def test_ncols_nrows_from_N():
    assert ncols_nrows_from_N(12) == {'ncols': 4, 'nrows': 3}
    assert ncols_nrows_from_N(8) == {'ncols': 3, 'nrows': 3}
    assert ncols_nrows_from_N(1) == {'ncols': 1, 'nrows': 1}
    assert ncols_nrows_from_N(0) == {'ncols': 0, 'nrows': 0}
    assert ncols_nrows_from_N(100) == {'ncols': 10, 'nrows': 10}

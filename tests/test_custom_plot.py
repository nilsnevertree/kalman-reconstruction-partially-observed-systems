from warnings import warn

import numpy as np
import pytest

from matplotlib.collections import PathCollection
from matplotlib.pyplot import Line2D
from numpy.testing import assert_equal

from kalman_reconstruction.custom_plot import (
    adjust_lightness,
    handler_map_alpha,
    ncols_nrows_from_N,
)


def test_handler_map_alpha():
    """Test case for the `handler_map_alpha` function."""
    handler_map = handler_map_alpha()
    assert PathCollection in handler_map
    assert Line2D in handler_map


@pytest.mark.parametrize(
    "N, expected",
    [
        (12, {"ncols": 4, "nrows": 3}),
        (8, {"ncols": 3, "nrows": 3}),
        (1, {"ncols": 1, "nrows": 1}),
        (0, pytest.raises(ValueError)),
        (100, {"ncols": 10, "nrows": 10}),
        (2.5, {"ncols": 2, "nrows": 2}),
        ("25", {"ncols": 5, "nrows": 5}),
        ("wrong", pytest.raises(ValueError)),
    ],
)
def test_ncols_nrows_from_N(N, expected):
    if isinstance(expected, dict):
        assert_equal(ncols_nrows_from_N(N), expected)
    else:
        with expected:
            ncols_nrows_from_N(N)

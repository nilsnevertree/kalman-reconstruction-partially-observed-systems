import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_almost_equal

from kalman_reconstruction.pipeline import (
    expand_and_assign_coords,
    multiple_runs_of_func,
    run_function_on_multiple_subdatasets,
)


def simple_function_01(ds, var="var", var_new="var_new"):
    """Multiply the variable by 2."""
    ds[var_new] = ds[var] * 2
    return ds


def simple_function_02(
    ds: xr.Dataset, var: str = "var", var_new: str = "var_new", value: int = 10
) -> xr.Dataset:
    """
    Perform a simple function on a dataset.

    This function creates a new dataset and performs a simple operation on a variable in the input dataset.
    It adds a new variable to the resulting dataset and assigns the sum of the input variable and a specified value to it.

    Parameters:
        ds (xr.Dataset): The input dataset.
        var (str): The name of the variable to operate on. Default is "var1".
        var_new (str): The name of the new variable to be added. Default is "var_new".
        value (int): The value to be added to the input variable. Default is 10.

    Returns:
        xr.Dataset: The resulting dataset with the modified variable.
    """
    # Create a new dataset with the same coordinates as the input dataset
    result = xr.Dataset(coords=ds.coords)

    # Assign additional coordinates to the resulting dataset
    result.assign_coords(dict(test=[0, 1]))

    # Copy the variable from the input dataset to the resulting dataset
    result[var] = ds[var]

    # Create a new DataArray with the specified value and coordinates from the resulting dataset
    result[var_new] = xr.DataArray(value, coords=result.coords)

    # Add the input variable and the new DataArray element-wise and assign the result to the new variable
    result[var_new] = result[var_new] + ds[var]

    return result


@pytest.fixture
def example_dataset_01():
    x = np.array([2, 3])
    y = np.array([0, 1, 2, 3])
    data_01 = np.array(
        [
            [0, 2, 3, np.nan],
            [1, 1, 1, 1],
        ]
    )
    return xr.Dataset(
        data_vars=dict(
            var=(("x", "y"), data_01),
        ),
        coords=dict(
            x=x,
            y=y,
        ),
    )


@pytest.fixture
def example_dataset_02():
    y = np.array([0, 1, 2, 3])
    z = np.array([7, 8])
    data_02 = np.array(
        [
            [0, 2],
            [1, 1],
            [2, np.nan],
            [0, 0],
        ]
    )
    return xr.Dataset(
        data_vars=dict(
            var=(("y", "z"), data_02),
        ),
        coords=dict(
            y=y,
            z=z,
        ),
    )


def multiply_by_scalar(ds: xr.Dataset, scalar: float) -> xr.Dataset:
    return ds * scalar


def test_expand_and_assign_coords():
    # Create example datasets
    ds1 = xr.Dataset(
        {"var1": (("x", "y"), np.random.rand(10, 20))},
        coords={"x": range(10), "y": range(20)},
    )

    ds2 = xr.Dataset({"var2": ("z", np.random.rand(5))}, coords={"z": range(5)})

    select_dict = {"z": 2}

    # Call the function
    result = expand_and_assign_coords(ds1, ds2, select_dict)

    # Assert the dimensions and coordinates of the result
    assert result.dims == {"z": 1, "x": 10, "y": 20}
    assert set(result.coords) == {"x", "y", "z"}

    # Assert the values of the variables in the result
    assert np.allclose(result["var1"].values, ds1["var1"].expand_dims("z").values)
    assert np.allclose(result["x"].values, ds1["x"].values)
    assert np.allclose(result["y"].values, ds1["y"].values)
    assert np.allclose(result["z"].values, ds2["z"].isel(**select_dict).values)


ds = xr.Dataset(
    {
        "data": (
            ("x", "y", "time"),
            [
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
                [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]],
            ],
        )
    },
    coords={"time": range(5), "x": [2, 3], "y": [5, 6, 7]},
)

# Define subdataset selections
subdataset_selections = [{"x": 2, "y": 5}, {"x": 3, "y": 6}]

# Define function arguments
func_kwargs = {"scalar": 2}

# Define processing function
processing_function = multiply_by_scalar

# Call the function
ds_result = run_function_on_multiple_subdatasets(
    processing_function=processing_function,
    parent_dataset=ds,
    subdataset_selections=subdataset_selections,
    func_kwargs=func_kwargs,
)

# Expected result
ds_expected = xr.Dataset(
    {
        "data": (
            ("x", "y", "time"),
            [
                [[2, 4, 6, 8, 10], [np.nan, np.nan, np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan, np.nan, np.nan], [42, 44, 46, 48, 50]],
            ],
        )
    },
    coords={"time": range(5), "x": [2, 3], "y": [5, 6]},
)

# Assert equality
xr.testing.assert_equal(ds_result, ds_expected)
# @pytest.mark.parametrize("example_dataset", ["example_dataset_01", "example_dataset_02"])
# def test_multiple_runs_of_func(example_dataset):
#     ds = example_dataset  # Evaluate the example dataset fixture name

#     # Call the multiple_runs_of_func function
#     result = multiple_runs_of_func(ds=ds, func=simple_function_01, func_kwargs={}, number_of_runs=3, new_dimension="run")

#     # Perform assertions
#     assert isinstance(result, xr.Dataset)
#     assert "run" in result.dims
#     assert result.dims["run"] == 3
#     assert "var" in result.data_vars
#     assert "var_new" in result.data_vars
#     assert result.data_vars["var_new"].shape[0] == 3
#     assert result.data_vars["var_new"].shape[1:] == simple_function_01(ds=ds).data_vars["var_new"].shape

# def test_multiple_runs_of_func():

#     ds = example_dataset_02()
#     # Call the multiple_runs_of_func function
#     result = multiple_runs_of_func(ds=ds, func=simple_function_01, func_kwargs={}, number_of_runs=3, new_dimension="run")

#     # Perform assertions
#     assert isinstance(result, xr.Dataset)
#     assert "run" in result.dims
#     assert result.dims["run"] == 3
#     assert "var" in result.data_vars
#     assert "var_new" in result.data_vars
#     assert result.data_vars["var_new"].shape[0] == 3
#     assert result.data_vars["var_new"].shape[1:] == simple_function_01(ds=ds).data_vars["var_new"].shape

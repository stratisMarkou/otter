from typing import (
    Dict,
    List,
    Optional,
)

import xarray as xr

ERA5_START_DATE = "1979-01-01"
ERA5_END_DATE = "2023-12-31"


def split_into_different_variables_along_dim(
    dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:
    """Splits variables in a dataset to different sub-variables
    along the specified dimension.

    For example, if the dataset contains a variable `temperature`
    with the dimensions `time`, `lat`, `lon`, `level` where
    dataset.level == (1000, 850, 700), then calling this function
    with dim="level" will return a new dataset where the
    `temperature` variable has been split into the sub-variables

        `temperature/level_1000`,
        `temperature/level_850`,
        `temperature/level_700`.

    The function join_into_single_variable_along_dim can be used
    to reverse this operation.

    Args:
        dataset: xarray dataset to split
        dim: dimension to split the variables along

    Returns:
        xarray dataset with variables split along the dimension
    """

    dataset = dataset.copy()

    assert dim in dataset.dims, f"Dimension {dim} not found in zarr"

    dim_values = dataset[dim].values

    for variable in dataset.variables:
        if dim in dataset[variable].dims and not (variable in dataset.coords):
            for dim_value in dim_values:
                dataset[f"{variable}/{dim}_{dim_value}"] = dataset[
                    variable
                ].sel({dim: dim_value})
            del dataset[variable]

    del dataset[dim]

    return dataset


def join_into_single_variable_along_dim(
    dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:
    """Joins variables in a dataset that have been split into
    sub-variables along the specified dimension.

    For example, if the dataset contains a variable `temperature`
    with the variables `temperature/level_1000`, `temperature/level_850`,
    `temperature/level_700`, then calling this function with
    dim="level" will return a new dataset where the `temperature`
    variable has been joined back into a single variable along the
    `level` dimension.

    This performs the inverse operation of the function
    split_into_different_variables_along_dim.

    Args:
        dataset: xarray dataset to join
        dim: dimension to join the variables along

    Returns:
        xarray dataset with variables joined along the dimension
    """

    dataset = dataset.copy()

    assert not (dim in dataset.dims), f"Dimension {dim} found in zarr"

    # First determine the unique values of the dimension
    dim_values: Dict[str, set[str]] = {}
    resulting_joined_variables = set()

    for dataset_variable in dataset.variables:
        dataset_variable = str(dataset_variable)
        # If the dimension we're joining over is in the variable name
        # do the joining, otherwise continue.
        if f"/{dim}_" in dataset_variable:

            # The `dataset_variable` string is assumed to be of the form
            # `variable_dim_dimvalue`. We split the string by the dimension
            # and extract the variable name and the dimension value.
            variable, dim_value = dataset_variable.split(f"/{dim}_")

            if variable not in resulting_joined_variables:
                resulting_joined_variables.add(variable)

            if variable not in dim_values:
                dim_values[variable] = set()

            dim_values[variable].add(dim_value)

    # Check that all variables have the same dimension values
    dim_values_as_list = list(dim_values.values())
    assert all(
        dim_values_as_list[0] == dv for dv in dim_values_as_list
    ), f"Variables have different dimension values for {dim=}"

    # Check that there exists at least one variable to join
    assert (
        len(resulting_joined_variables) > 0
    ), f"No variables to join for {dim=}"

    # For each unique value of the dimension, join the variables along
    # the specified dimension.
    dim_values_sorted = sorted(list(dim_values_as_list[0]))

    for variable in resulting_joined_variables:
        dataset[variable] = xr.concat(
            [
                dataset[f"{variable}/{dim}_{dim_value}"]
                for dim_value in dim_values_sorted
            ],
            dim=dim,
        )

        for value in dim_values_sorted:
            del dataset[f"{variable}/{dim}_{value}"]

    return dataset


def stack_dataset_variable_and_levels(
    dataset: xr.Dataset,
    name: str = "stacked",
    vars_to_drop: Optional[List[str]] = None,
) -> xr.DataArray:
    if vars_to_drop:
        dataset = dataset.drop_vars(vars_to_drop)

    # Stack all variables and levels into a single dimension. By default,
    # xarray stacks the variables across the first dimension, so we then
    # reshape the dimensions to have the variables and levels as the last
    # dimension.
    data_array = dataset.to_dataarray(dim="variable_and_level", name=name)
    return data_array.transpose(*data_array.dims[1:], data_array.dims[0])


def unstack_data_array_variable_and_levels(
    data_array: xr.DataArray,
) -> xr.Dataset:

    # Unstack the variables and levels from the last dimension to the first
    dataset = data_array.to_dataset(dim="variable_and_level")
    return dataset.transpose(data_array.dims[-1], *data_array.dims[:-1])

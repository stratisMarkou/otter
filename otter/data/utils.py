import bisect
import os
import pickle
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import xarray as xr
from tqdm import trange

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
        if (dim in dataset[variable].dims) and not (
            variable in dataset.coords
        ):
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
    `level` dimension, with the dimension values *sorted*.

    This performs the reverse operation of the function
    split_into_different_variables_along_dim, assuming that the
    original dataarray was sorted before splitting.

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
        # If the variable name we're currently at in the iteration has
        # the dimension we're joining over, then we add it.
        if f"/{dim}_" in dataset_variable:

            # The `dataset_variable` string is assumed to be of the form
            # `variable/dim_dimvalue`. We split the string by the dimension
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

    # We will join the variables in sorted order.
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
    return data_array.to_dataset(dim="variable_and_level")


def save_xarray_dataset_as_memmap(
    dataset: xr.Dataset,
    path: str,
    iteration_dimension: str | None,
    sorted_dims: Optional[List[str]] = None,
) -> None:

    def _slice_axis_n_at_index_i(
        n: int, i: int, num_axes: int
    ) -> tuple[slice | int, ...]:
        if not (0 <= n <= num_axes):
            raise ValueError(
                f"Axis {n=} must be in the range [0, {num_axes=}]."
            )
        return (slice(None),) * n + (i,) + (slice(None),) * (num_axes - n - 1)

    os.makedirs(path)

    # Split the dataset into different variables along the
    # variable_and_level dimension.
    split_dataset = split_into_different_variables_along_dim(
        dataset,
        dim="level",
    )
    data_array_template = stack_dataset_variable_and_levels(split_dataset)
    data_array_template = data_array_template.sortby("variable_and_level")

    # Save dimensions.
    with open(os.path.join(path, "dim_names.pkl"), "wb") as file:
        pickle.dump(data_array_template.dims, file)

    os.makedirs(os.path.join(path, "dims"))
    for dim in data_array_template.dims:
        dim_path = os.path.join(path, "dims", f"{dim}.npy")
        np.save(dim_path, data_array_template[dim].values)

    # Save sorted dimensions.
    with open(os.path.join(path, "sorted.pkl"), "wb") as file:
        pickle.dump(sorted_dims, file)

    # Save shape of data array.
    with open(os.path.join(path, "shape.pkl"), "wb") as file:
        pickle.dump(data_array_template.shape, file)

    # Save data type.
    with open(os.path.join(path, "dtype"), "w") as file:  # type: ignore
        file.write(str(data_array_template.dtype))  # type: ignore

    # Open memmap to save data.
    data_memmap = np.memmap(
        os.path.join(path, "data.memmap"),
        dtype=data_array_template.dtype,
        mode="w+",
        shape=data_array_template.shape,
    )

    if iteration_dimension is None:
        data_memmap[...] = data_array_template.values

    else:
        iteration_index = list(data_array_template.dims).index(
            iteration_dimension
        )

        # Iterate over the data array and save the data.
        for i in trange(data_array_template[iteration_dimension].shape[0]):

            # Save data to memmap at index i along the iteration dim.
            slice_i = _slice_axis_n_at_index_i(
                iteration_index, i, data_array_template.ndim
            )

            split_dataset_slice_i = split_into_different_variables_along_dim(
                dataset.isel({iteration_dimension: i}), dim="level"
            )
            stacked_array_slice_i = stack_dataset_variable_and_levels(
                split_dataset_slice_i
            )
            stacked_array_slice_i = stacked_array_slice_i.sortby(
                "variable_and_level"
            )

            # Save data to memmap.
            data_memmap[slice_i] = stacked_array_slice_i.values


@dataclass
class Coordinate:
    name: str
    values: npt.NDArray[np.float32]
    is_sorted: bool


class LabelledArray:

    def __init__(
        self,
        data: npt.NDArray[Any],
        coords: List[Coordinate],
    ) -> None:

        self.data = data
        self.coords = coords

        if len(coords) != data.ndim:
            raise ValueError(
                "Number of coordinates must match number of dimensions in data"
            )

        for i, coord in enumerate(coords):
            if len(coord.values) != data.shape[i]:
                raise ValueError(
                    f"Length of coordinate `{coord.name}` must match the "
                    f"length of the data along dimension {i}, found "
                    f"{len(coord.values)} and {data.shape[i]} respectively."
                )

    def sel(self, **index_coords: Any) -> npt.NDArray[np.float32]:

        # Get indices along each coordinate.
        indices: List[List[int] | None] = []
        for coord in self.coords:

            if coord.name not in index_coords:
                indices.append(None)

            else:
                coord_indices: List[int] = []
                # Search for the index of the value in the coordinate.
                for value in index_coords[coord.name]:
                    if coord.is_sorted:
                        index = bisect.bisect_left(coord.values, value)
                        assert coord.values[index] == value, (
                            f"Value {value} not found in sorted coordinate "
                            f"{coord.name}"
                        )
                        coord_indices.append(index)

                    else:
                        index = int(np.where(coord.values == value)[0][0])
                        coord_indices.append(index)

                indices.append(coord_indices)

        # Select the data.
        result = self.data
        for i, idx in enumerate(indices):
            if idx is not None:
                result = np.take(result, idx, axis=i)

        return result


def open_memmap_and_dims(
    path: str,
) -> Tuple[Tuple[str, ...], npt.NDArray[np.float32]]:

    # Get dimensions.
    with open(os.path.join(path, "dim_names.pkl"), "rb") as file:
        dims = pickle.load(file)

    # Get shape of data array.
    with open(os.path.join(path, "shape.pkl"), "rb") as file:
        shape = pickle.load(file)

    # Get data type.
    with open(os.path.join(path, "dtype"), "r") as file:
        dtype = str(file.read())

    # Open memmap.
    data = np.memmap(
        os.path.join(path, "data.memmap"),
        dtype=dtype,
        mode="r",
        shape=shape,
    )

    return dims, data


def open_memmap_as_labelled_array(path: str) -> LabelledArray:

    # Get dimensions and data.
    dims, data = open_memmap_and_dims(path)

    # Get coordinates.
    coords = []
    for dim in dims:
        dim_path = os.path.join(path, "dims", f"{dim}.npy")
        values = np.load(dim_path, allow_pickle=True)
        is_sorted = False
        if os.path.exists(os.path.join(path, "sorted.pkl")):
            with open(os.path.join(path, "sorted.pkl"), "rb") as file:
                sorted_dims = pickle.load(file)
                if dim in sorted_dims:
                    is_sorted = True
        coords.append(Coordinate(dim, values, is_sorted))

    return LabelledArray(data=data, coords=coords)

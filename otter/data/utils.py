import xarray as xr

ERA5_START_DATE = "1979-01-01"
ERA5_END_DATE = "2023-12-31"


def split_into_different_variables_along_dim(
    dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:

    dataset = dataset.copy()

    assert dim in dataset.dims, f"Dimension {dim} not found in zarr"

    dim_values = dataset[dim].values
    variables = list(dataset.variables)

    for variable in variables:
        if dim in dataset[variable].dims and not (variable in dataset.coords):
            for dim_value in dim_values:
                dataset[f"{variable}_{dim}_{dim_value}"] = dataset[
                    variable
                ].sel({dim: dim_value})
            del dataset[variable]

    del dataset[dim]

    return dataset


def stack_dataset_variable_and_levels(
    dataset: xr.Dataset,
    name: str = "stacked",
) -> xr.DataArray:
    dataset = dataset.drop_vars(["time"])

    # Stack all variables and levels into a single dimension. By default,
    # xarray stacks the variables across the first dimension, so we then
    # reshape the dimensions to have the variables and levels as the last
    # dimension.
    data_array = dataset.to_dataarray(dim="variable_and_level", name=name)
    return data_array.transpose(*data_array.dims[1:], data_array.dims[0])

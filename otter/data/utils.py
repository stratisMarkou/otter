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


def stack_dataset_variable_and_levels(dataset: xr.Dataset) -> xr.Dataset:
    dataset = dataset.drop_vars(["time"])
    # Broadcast all variables which do not have time_index as a coordinate
    # to have time_index as a coordinate, repeating the variable along the
    # time_index dimension.
    for variable in dataset.data_vars:
        if "time_index" not in dataset[variable].dims:
            dataset[variable] = dataset[variable].expand_dims(
                {"time_index": dataset.time_index}
            )

    return dataset.to_stacked_array(
        new_dim="variable_and_level",
        sample_dims=("batch", "latitude", "longitude"),
    )

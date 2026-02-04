# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

results_path = Path("/home/jonas/Documents/code/otter/results.nc")
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", type=str)
parser.add_argument("--results_directory", type=str, default="_results")
args = parser.parse_args()

experiment_name = args.experiment_name

results_path = (
    Path(args.results_directory) / experiment_name / "weatherbench_results.nc"
)

ds = xr.open_dataset(results_path)
ds = ds.sel(region="global")

plottables = [
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("10m_u_component_of_wind", None),
    ("10m_v_component_of_wind", None),
    ("u_component_of_wind", 850),
    ("v_component_of_wind", 850),
    ("specific_humidity", 700),
    ("temperature", 850),
    ("geopotential", 500),
    ("geopotential", 850),
]

fig, axs = plt.subplots(len(plottables), 1, figsize=(8, 4 * len(plottables)))

for ax, (var, level) in zip(axs, plottables):
    rmse_var = f"rmse.{var}"
    if rmse_var not in ds:
        print(f"{var} not in dataset, skipping plot.")
        continue
    if level is not None:
        da = ds.sel(level=level)[rmse_var]
        title = f"{var} at {level} hPa"
    else:
        da = ds[rmse_var]
        title = var

    # Currently a nanosecond timedelta, convert to hours
    lead_times = da.lead_time
    lead_times_hours = lead_times / np.timedelta64(1, "h")

    ax.plot(lead_times_hours, da, label=title, marker="x", linestyle="--")
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("RMSE")
    ax.legend()

fig.savefig(
    f"_plots/{experiment_name}_rmse_over_time.pdf", bbox_inches="tight"
)

# Print 24h RMSE for all variables:
for var, level in plottables:
    rmse_var = f"rmse.{var}"
    if rmse_var not in ds:
        continue
    if level is not None:
        da = ds.sel(level=level)[rmse_var]
    else:
        da = ds[rmse_var]
    rmse_24h = da.sel(lead_time=np.timedelta64(24, "h")).values
    if level is not None:
        print(f"{var} at {level} hPa: {rmse_24h}")
    else:
        print(f"{var}: {rmse_24h:.4f}")

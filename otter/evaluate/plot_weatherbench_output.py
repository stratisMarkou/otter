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
    ("geopotential", 500, 44.96, 41.93),
    ("temperature", 850, 0.615, 0.593),
    ("specific_humidity", 700, 0.000527, 0.000513),
    ("u_component_of_wind", 850, 1.219, 1.172),
    ("v_component_of_wind", 850, 1.251, 1.203),
    ("2m_temperature", None, 0.539, 0.517),
    ("mean_sea_level_pressure", None, 56.22, 52.22),
    ("10m_u_component_of_wind", None, 0.783, 0.749),
    ("10m_v_component_of_wind", None, 0.819, 0.783),
]

fig, axs = plt.subplots(len(plottables), 1, figsize=(8, 4 * len(plottables)))

for ax, (var, level, arches_m, arches_4m) in zip(axs, plottables):
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

path = Path("_plots") / f"{experiment_name}_rmse_over_time.pdf"
path.parent.mkdir(exist_ok=True)

fig.savefig(path, bbox_inches="tight")

# Print 24h RMSE for all variables:
for var, level, arches_m, arches_4m in plottables:
    rmse_var = f"rmse.{var}"
    if rmse_var not in ds:
        continue
    if level is not None:
        da = ds.sel(level=level)[rmse_var]
    else:
        da = ds[rmse_var]
    rmse_24h = da.sel(lead_time=np.timedelta64(24, "h")).values
    percent_m = 100 * (arches_m - rmse_24h) / arches_m
    percent_4m = 100 * (arches_4m - rmse_24h) / arches_4m
    if level is not None:
        print(
            f"{var} at {level} hPa: {rmse_24h:.3f} (Arches M: {arches_m}, Arches 4M: {arches_4m})"
        )
    else:
        print(
            f"{var}: {rmse_24h:.3f} (Arches M: {arches_m}, Arches 4M: {arches_4m})"
        )
    print(f"Percent better than Arches M: {percent_m:.2f}%")
    print(f"Percent better than Arches 4M: {percent_4m:.2f}%\n")
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate

from otter.data.datasets import ForecastingSample
from otter.experiment.downsample import downsample
from otter.experiment.run_state import RunState
from otter.experiment.train import preprocess_batch
from otter.experiment.utils import _get_config
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.losses.loss_fn import (
    lwrmse,
)
from otter.models.losses.weighting import (
    apply_division_by_variable_diff_std,
    apply_pressure_weighting,
    apply_surface_weighting,
)
from otter.models.multiscale_model import (
    MultiscaleModel,
    get_ctx_vars_and_levels,
    get_trg_vars_and_levels,
)
from otter.plots.geoplot import GeoPlotter, GeoVariablePlotter

# %%
config = _get_config(
    config_module="otter.experiment.config",
    config_name="multiscale_probabilistic",
    extra_args=[
        "feedforward_network.hidden_features=2048",
        "constants.data.batch_size=4",
        "constants.data.num_workers=0",
        "constants.data.prefetch_factor=null",
        "constants.data.valid_num_target_frames=1",
    ],
)
# %%
config.model.device = "cuda"

experiment = instantiate(config)
# %%
scale_scheduler: ScaleScheduler = experiment.inference_scale_scheduler
data_scale, _ = scale_scheduler.sample_data_scale_pair()
rollout_scales = scale_scheduler.get_rollout_scales()

all_scales = [data_scale] + list(rollout_scales)
all_scales = [int(scale) for scale in all_scales]
print(all_scales)
# %%

dataloader = experiment.valid_dataloader
batch: ForecastingSample = next(iter(dataloader))
batch = batch.to("cuda")
model_input = preprocess_batch(batch, data_scale)
model_input = model_input.to("cuda")
# %%
model: MultiscaleModel = experiment.model.to("cuda")

# checkpoint_path = "/home/jonas/Documents/code/otter/_results/2025-03-25_12-03-daring-badger/checkpoints/latest.pt"
checkpoint_path = "/home/jonas/Documents/code/otter/_results/2025-05-08_11-05-graceful-fox/checkpoints/latest.pt"
checkpoint_path = Path(checkpoint_path)
run_state = RunState.load(checkpoint_path)
model.load_state_dict(run_state.model_state)

_ = model.eval()
# %%
VARS_TO_REMOVE = ["sea_surface_temperature"]

ctx_vars_and_levels = get_ctx_vars_and_levels(VARS_TO_REMOVE)
trg_vars_and_levels = get_trg_vars_and_levels(VARS_TO_REMOVE)

ctx_var_to_idx = {var: idx for idx, var in enumerate(ctx_vars_and_levels)}
trg_var_to_idx = {var: idx for idx, var in enumerate(trg_vars_and_levels)}

plotter = GeoVariablePlotter.default(ctx_var_to_idx, trg_var_to_idx)
# %%
weighting_fns = [
    apply_surface_weighting,
    apply_pressure_weighting,
    apply_division_by_variable_diff_std,
]

# TODO
weighting_fns = []

loss_inner = lwrmse
batch = next(iter(dataloader))
model_input = preprocess_batch(batch, data_scale)
model_input = model_input.to("cuda")

rmses = []
mean_summed = None

with torch.no_grad():
    prd_dist, _ = model(batch=model_input, scales=rollout_scales)
# %%
# prd_dist.mean.shape
batch.trg.shape
# %%
VAR_AND_LEVELS = [
    "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "2m_temperature",
    # "specific_humidity/level_500",
]

gb = plotter.new_grid_builder(VAR_AND_LEVELS)
gb.add_column(batch.trg[0, :, :, 0, :], is_trg_like=True, title="Target")

for i, scale in enumerate(all_scales):
    gb.add_column(
        prd_dist.mean[0, :, :, 0, i, :].to("cpu"),
        is_trg_like=True,
        title=f"Mean ({scale})",
    )

gb.plot(share_cmap="none")
# %%

ground_truths = []
for i, scale in enumerate(rollout_scales):
    gt_downsampled = downsample(batch.trg, scale)
    ground_truths.append(gt_downsampled[0, :, :, 0, :].to("cpu"))
ground_truths.append(batch.trg[0, :, :, 0, :].to("cpu"))

indices = [0]

p = GeoPlotter()
means = prd_dist.mean[0, :, :, 0, :, indices].permute(0, 1, 3, 2).to("cpu")
gt = (
    torch.stack(ground_truths, dim=-2)[..., indices]
    .permute(0, 1, 3, 2)
    .to("cpu")
)

errs = means - gt

tminus1 = batch.ctx[0, :, :, :-1, indices].repeat(1, 1, 1, len(all_scales))

plottable = torch.cat([tminus1, gt, means, errs], dim=-2)
# %%

p.plot_grid(
    plottable,
    col_titles=["T-1", "ground truth", "predicted mean", "error"],
    row_titles=all_scales,
    share_cmap="none",
)

# %%

rmses = []
for i in range(len(all_scales)):
    rmse = errs[:, :, 0, i].pow(2).mean().sqrt()
    rmses.append(rmse)

plt.plot(all_scales, rmses, "x")
plt.xlabel("Scale")
plt.ylabel("RMSE")
plt.xscale("log")

# %%

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from otter.data.datasets import ForecastingSample
from otter.experiment.run_state import RunState
from otter.experiment.train import preprocess_batch
from otter.experiment.utils import _get_config
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.distributions import Delta
from otter.models.losses.loss_fn import (
    apply_loss,
    lwrmse,
)
from otter.models.losses.weighting import (
    apply_division_by_variable_diff_std,
    apply_pressure_weighting,
    apply_surface_weighting,
)
from otter.models.multiscale_model import MultiscaleModel
from otter.plots.geoplot import GeoPlotter

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
gp = GeoPlotter()
# %%
weighting_fns = [
    apply_surface_weighting,
    apply_pressure_weighting,
    apply_division_by_variable_diff_std,
]

# TODO
weighting_fns = []

loss_inner = lwrmse
num_times = 1
samples_per_time = 20
ns = range(samples_per_time)
all_rmses = []
single_scale_rmses = []

for time in range(num_times):
    print("time: ", time)
    batch = next(iter(dataloader))

    rmses = []
    mean_summed = None

    with torch.no_grad():
        prd_dist, _ = model(batch=model_input, scales=[])
        prd_dist = Delta(prd_dist.mean.to("cpu"))

        single_scale_rmse, _ = apply_loss(
            batch.trg.to("cpu"),
            prd_dist,
            loss_inner,
            weighting_coefficients_power=0.5,
            weighting_fns=weighting_fns,
            trg_variables_and_levels=model.trg_variables_and_levels,
            scales=[data_scale],
        )
        single_scale_rmses.append(single_scale_rmse.item())

        for n in tqdm(ns):
            prd_dist, _ = model(batch=model_input, scales=rollout_scales)

            mean = prd_dist.mean.to("cpu")  # [..., :1, :]

            if mean_summed is None:
                mean_summed = mean
            else:
                mean_summed += mean

            mean = mean_summed / (n + 1)

            mean_dist = Delta(mean=mean)

            rmse_val, _ = apply_loss(
                batch.trg.to("cpu"),
                mean_dist,
                loss_inner,
                weighting_coefficients_power=0.5,
                weighting_fns=weighting_fns,
                trg_variables_and_levels=model.trg_variables_and_levels,
                scales=all_scales,  # [:1],
            )
            rmses.append(rmse_val.item())

    all_rmses.append(rmses)

# %%

all_rmses_arr = np.array(all_rmses)
mean_rmses = np.mean(all_rmses_arr, axis=0)

mean_single_scale_rmse = np.mean(single_scale_rmses)

plt.plot(np.arange(1, samples_per_time + 1), mean_rmses, "x-")
# plt.hlines(
#    mean_single_scale_rmse,
#    xmin=1,
#    xmax=max(ns) + 1,
#    color="red",
#    label="Single Scale RMSE",
# )
plt.xlabel("Number of Samples")
plt.ylabel("RMSE of Gaussian Mixture")
plt.legend()
plt.show()

# %%
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from otter.data.datasets import ForecastingSample
from otter.experiment.downsample import downsample
from otter.experiment.run_state import RunState
from otter.experiment.train import preprocess_batch
from otter.experiment.utils import _get_config
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.distributions import Delta
from otter.models.losses.loss_fn import apply_loss, mean_neg_log_prob
from otter.models.losses.loss_fn import lwrmse as otter_lwrmse
from otter.models.losses.weighting import (
    apply_pressure_weighting,
    apply_surface_weighting,
)
from otter.models.multiscale_model import MultiscaleModel


def lwrmse(trg, prd):
    prd_dist = Delta(prd)
    rmse_val, _ = apply_loss(
        trg.to("cpu"),
        prd_dist,
        otter_lwrmse,
        weighting_coefficients_power=0.5,
        weighting_fns=[],
        trg_variables_and_levels=model.trg_variables_and_levels,
        scales=[data_scale],
    )
    return rmse_val


# %%
config = _get_config(
    config_module="otter.experiment.config",
    config_name="multiscale_probabilistic",
    extra_args=[
        "feedforward_network.hidden_features=2048",
        "constants.data.batch_size=1",
        "constants.data.num_workers=0",
        "constants.data.prefetch_factor=null",
        # TODO
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
model: MultiscaleModel = experiment.model.to("cuda")

# checkpoint_path = "/home/jonas/Documents/code/otter/_results/2025-03-25_12-03-daring-badger/checkpoints/latest.pt"
# checkpoint_path = "/home/jonas/Documents/code/otter/_results/2025-03-25_12-03-daring-badger/scale_rollout_finetuning/checkpoints/latest.pt"
checkpoint_path = "/home/jonas/Documents/code/otter/_results/2025-05-08_11-05-graceful-fox/checkpoints/latest.pt"
checkpoint_path = Path(checkpoint_path)
run_state = RunState.load(checkpoint_path)
model.load_state_dict(run_state.model_state)

_ = model.eval()
# %%

dataloader = experiment.train_dataloader
batch: ForecastingSample = next(iter(dataloader))
batch = batch.to("cuda")
model_input = preprocess_batch(batch, data_scale)
model_input = model_input.to("cuda")
# %%
NUM_BATCHES = 5
NUM_SAMPLES = 5
means_batch = []
trgs = []

for _ in tqdm(range(NUM_BATCHES)):
    batch: ForecastingSample = next(iter(dataloader))

    batch = batch.to("cuda")

    model_input = preprocess_batch(batch, data_scale)

    means = []

    trgs.append(batch.trg.cpu())

    with torch.no_grad():
        for _ in tqdm(range(NUM_SAMPLES)):
            rollout_dist, lbl = model(
                batch=model_input,
                scales=rollout_scales,
            )
            means.append(rollout_dist.mean.cpu())

    means = torch.stack(means, dim=0)
    means_batch.append(means)

means_batch = torch.stack(means_batch, dim=0)
trgs = torch.stack(trgs, dim=0)

results = {scale: [] for scale in all_scales}

for i in range(1, NUM_SAMPLES + 1):
    means_first = means_batch[:, :i].mean(dim=1)
    for j, scale in enumerate(all_scales):
        trg = trgs[:, 0, :, :, :4].cpu()
        prd = means_first[:, 0, :, :, :4, j : j + 1]
        ensemble_lwrmse = (
            lwrmse(trg=trg, prd=prd).squeeze().mean(dim=0).sum().item()
        )

        results[scale].append(ensemble_lwrmse)

for scale, rmses in results.items():
    plt.plot(range(1, NUM_SAMPLES + 1), rmses, label=f"Scale {scale}")
plt.legend()
# %%
# dataloader = experiment.train_dataloader
dataloader = experiment.valid_dataloader

NUM_BATCHES = 5
NUM_SAMPLES = 2
nlls_by_scale = defaultdict(list)
with torch.no_grad():
    for _ in tqdm(range(NUM_BATCHES)):
        for i in range(NUM_SAMPLES):
            batch: ForecastingSample = next(iter(dataloader))
            batch = batch.to("cuda")
            model_input = preprocess_batch(batch, data_scale)
            model_input = model_input.to("cuda")
            prd_dist, _ = model(batch=model_input, scales=rollout_scales)

            nll, loss_info = apply_loss(
                batch.trg,
                prd_dist,
                mean_neg_log_prob,
                weighting_coefficients_power=1.0,
                weighting_fns=[],
                # weighting_fns=[
                #    apply_surface_weighting,
                #    apply_pressure_weighting,
                # ],
                trg_variables_and_levels=model.trg_variables_and_levels,
                scales=all_scales,
            )

            for s in all_scales:
                nlls_by_scale[s].append(loss_info[f"loss/scale_{s}"])

nlls_by_scale = {k: np.mean(v) for k, v in nlls_by_scale.items()}

xs = nlls_by_scale.keys()
ys = nlls_by_scale.values()

plt.plot(xs, ys, "x")
plt.xscale("log")
# %%

NUM_BATCHES = 2
nlls_by_scale = defaultdict(list)
# dataloader = experiment.valid_dataloader
dataloader = experiment.train_dataloader

with torch.no_grad():
    for i in range(NUM_BATCHES):
        batch = next(iter(dataloader))
        batch = batch.to("cuda")

        model_input = preprocess_batch(batch, data_scale)
        model_input = model_input.to("cuda")

        for i, scale in enumerate(all_scales):
            model_input = model_input.to("cuda")
            prd_dist, _ = model(
                batch=model_input,
                scales=[],
            )

            model_input = model_input.to("cpu")

            loss, loss_info = apply_loss(
                batch.trg,
                prd_dist,
                mean_neg_log_prob,
                weighting_coefficients_power=1.0,
                weighting_fns=[
                    apply_surface_weighting,
                    apply_pressure_weighting,
                ],
                trg_variables_and_levels=model.trg_variables_and_levels,
                scales=[scale],
            )
            nlls_by_scale[scale].append(loss_info[f"loss/scale_{scale}"])

            if i == len(all_scales) - 1:
                break

            # squeeze the scale dim
            sample = prd_dist.sample().detach().squeeze(-2)
            downsampled_trg = downsample(
                sample,
                int(all_scales[i + 1]),
            )  # (batch, lat, lon, 1, trg_var_and_level)

            model_input.downsampled_trg = downsampled_trg

nlls_by_scale = {k: np.mean(v) for k, v in nlls_by_scale.items()}
stds_by_scale = {k: np.std(v) for k, v in nlls_by_scale.items()}

xs = list(nlls_by_scale.keys())
ys = list(nlls_by_scale.values())
yerrs = list(stds_by_scale.values())

plt.errorbar(xs, ys, yerr=yerrs, fmt="x")
plt.xscale("log")

# %%

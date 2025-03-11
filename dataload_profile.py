# %%

from time import perf_counter

import hydra
from hydra import compose

# %%

with hydra.initialize_config_module(
    config_module="otter.experiment.config", version_base="1.3"
):
    config_yaml = compose(config_name="deterministic.yaml")

# %%
trainloader = hydra.utils.instantiate(config_yaml.train_dataloader)
train_iter = iter(trainloader)
# %%
start = perf_counter()
N_SAMPLES = 100
for _ in range(N_SAMPLES):
    next(train_iter)
end = perf_counter()

time_ms = (end - start) / N_SAMPLES * 1000
print(f"Time per sample: {time_ms:.2f} ms")

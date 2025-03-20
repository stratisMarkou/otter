import cProfile
import io
import pstats
from time import perf_counter

import hydra
from hydra import compose

# %%
with hydra.initialize_config_module(
    config_module="otter.experiment.config", version_base="1.3"
):
    config_yaml = compose(config_name="deterministic.yaml")
    config_yaml.backbone_network.feedforward_network.hidden_features = 256
    config_yaml.constants.data.num_workers = 0
    config_yaml.constants.data.prefetch_factor = None
# %%
trainloader = hydra.utils.instantiate(config_yaml.train_dataloader)
train_iter = iter(trainloader)

# %%
# Basic timing
start = perf_counter()
N_SAMPLES = 100
for _ in range(N_SAMPLES):
    next(train_iter)
end = perf_counter()
time_ms = (end - start) / N_SAMPLES * 1000
print(f"Time per sample: {time_ms:.2f} ms")

# %%
# More detailed profiling with cProfile
profiler = cProfile.Profile()
profiler.enable()

# Run the same loop
for _ in range(N_SAMPLES):
    next(train_iter)

profiler.disable()

# Print detailed stats
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
ps.print_stats(20)  # Top 20 functions by cumulative time
print(s.getvalue())

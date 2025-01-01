# Otter

__Priorities:__
- [ ] Implement task wrapper:
    1. Receive xarray datasets.
    2. (If relevant) apply processing for target differences.
    3. Apply normalisation (also support undoing normalisation).
    4. Predict for specified time step.
    5. Undo normalisation.
    6. Handle residual connections, and rollouts.
- [ ] Basic evaluation metrics.
- [ ] Wandb logging.

__Sub goals:__
- [x] Dry-run an MSE model on the gridded weather data task in a standalone script.
- [ ] Augment training with DDP.

__Functionality to implement:__
- [ ] Implement losses.
- [x] Normalisation statistics computation (as pure lists in repo).

__Tests to implement:__
- [x] Test that forward pass through small transformer architecture works.
- [ ] Extend existing tests to ensure all parameters are receiving gradients. (How do we get a list of parameters that should receive gradients though?)
- [ ] Test for gridded weather data task.
    - [ ] Decide if we need to profile this within tests (i.e. test within CI that loading is not too slow).
- [x] Test for data download.

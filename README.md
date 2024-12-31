# Otter

__Sub goals:__
- [ ] Train an MSE model on the gridded weather data task in a standalone script.
- [ ] Augment training with DDP.

__Functionality to implement:__
- [ ] Implement losses.
- [x] Normalisation statistics computation (as pure lists in repo).
- [ ] Normalisation wrapper.
- [ ] Residuals wrapper.
- [ ] Rollouts wrapper (should handle residuals etc).
- [ ] Evaluation metrics.
- [ ] Wandb logging.

__Tests to implement:__

- [x] Test that forward pass through small transformer architecture works.
- [ ] Extend existing tests to ensure all parameters are receiving gradients. (How do we get a list of parameters that should receive gradients though?)
- [ ] Test for gridded weather data task.
    - [ ] Decide if we need to profile this within tests (i.e. test within CI that loading is not too slow).
- [x] Test for data download.

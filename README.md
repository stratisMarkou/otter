# Otter


__Functionality to implement:__
- [ ] Implement losses.
- [ ] Input and output normalisation (as pure lists in repo).
- [ ] Wrapper for rollouts (should handle residuals etc).
- [ ] Evaluation metrics.
- [ ] Wandb logging.

__Tests to implement:__

- [x] Test that forward pass through small transformer architecture works.
- [ ] Extend existing tests to ensure all parameters are receiving gradients. (How do we get a list of parameters that should receive gradients though?)
- [ ] Test for gridded weather data task.
    - [ ] Decide if we need to profile this within tests (i.e. test within CI that loading is not too slow).
- [x] Test for data download.

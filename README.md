# Otter

__Priorities:__
- [x] Finalise faster training optimisations.
- [ ] Add spatial Fourier embeddings to model.
- [ ] Add model checkpointing.

__Sub goals:__
- [x] Dry-run an MSE model on the gridded weather data task in a standalone script.
- [ ] Train an MSE model to match Stormer.


__Tests to implement:__
- [x] Test that forward pass through small transformer architecture works.
- [ ] Test for gridded weather data task.
    - [ ] Decide if we need to profile this within tests (i.e. test within CI that loading is not too slow).
- [x] Test for data download.

"""Implementation of a cached queue dataloader.

This dataloader is a thin wrapper around a regular pytorch dataloader, to use
a cached queue to speed up training. The idea is to keep a cache of previously
loaded batches, and to occasionally return a batch from this cache.

Overal behaviour:

    * If the cache contains enough batches the loader may decide to return a
      batch from the cache. The usage count of that batch is increased, so its
      position in the queue is lower.

    * If the loader loads a fresh batch, it is added to the cache. If the cache
      already contains the maximum number of batches, the most-used batch is
      removed from the cache.
"""

from __future__ import annotations

import heapq
import random
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Tuple,
)

from torch.utils.data import DataLoader


class CachedQueueDataLoader:

    def __init__(
        self,
        min_cache_size: int,
        max_cache_size: int,
        prob_use_chache: float,
        dataloader: DataLoader[Any],
    ) -> None:

        self.min_cache_size = min_cache_size
        self.max_cache_size = max_cache_size
        self.prob_use_chache = prob_use_chache
        self.dataloader = dataloader

        self.iter_dataloader: Optional[Iterator[Any]] = None
        self.cached_queue: List[Tuple[int, Any]] = []

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> CachedQueueDataLoader:
        self.cached_queue = []
        self.iter_dataloader = iter(self.dataloader)
        return self

    def __next__(self) -> Any:
        assert self.iter_dataloader

        # If the cache is large enough and we decide to use it, return a batch.
        if (
            len(self.cached_queue) > self.min_cache_size
            and random.random() < self.prob_use_chache
        ):
            _, batch = self.cached_queue[0]
            return batch

        # If the cache is not large enough, or we decide not to use it, load a
        # fresh batch. If cache is full, remove the most used batch. Then add
        # most freshly used batch to the cache.
        else:
            batch = next(self.iter_dataloader)

            # Get batch with largest usage count.
            if len(self.cached_queue) >= self.max_cache_size - 1:
                most_used_batch = heapq.nlargest(1, self.cached_queue)[0]
                self.cached_queue.remove(most_used_batch)

            heapq.heappush(self.cached_queue, (1, batch))

            return batch

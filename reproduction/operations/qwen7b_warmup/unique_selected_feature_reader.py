"""Asynchronous CPU feature production for one selected own-G5 per question."""
import copy
import random
import time
import torch
from .replay_feature_reader_fast import PrefetchedReplayFeatureReader
from .replay_probe_layers import CAPTURE_KIND, REVISION


class SelectedG5PrefetchReader(PrefetchedReplayFeatureReader):
    def prepare(self, sampler):
        if sampler.pending is not None:
            raise ValueError('Feature lookahead requires an accepted sampler boundary')
        cursor = sampler.accepted_updates
        if not hasattr(self, '_prefetch'):
            from concurrent.futures import ThreadPoolExecutor
            import threading
            self._locals = threading.local()
            self._worker_readers = []
            self._pool = ThreadPoolExecutor(max_workers=self.parallel_workers,
                                            thread_name_prefix='CPU-selected-G5')
            self._prefetch, self._plans = {}, {}
            self.prefetch_statistics = dict(submitted_question_groups=0,
                consumed_question_groups=0, wait_seconds=0., prepare_calls=0,
                trajectories_per_question=5)
        self.prefetch_statistics['prepare_calls'] += 1
        self._plans = {c:p for c,p in self._plans.items() if c >= cursor}
        if len(self._plans) >= self.prefetch_batches // 2:
            return
        shadow = copy.copy(sampler)
        shadow.rng = random.Random()
        shadow.rng.setstate(sampler.rng.getstate())
        shadow.orders = {r:list(v) for r,v in sampler.orders.items()}
        shadow.positions, shadow.cycles = dict(sampler.positions), dict(sampler.cycles)
        shadow._partition_cache = dict(sampler._partition_cache)
        shadow.coverage_receipts = list(sampler.coverage_receipts)
        shadow.pending = None
        limit = min(self.prefetch_batches, sampler.steps_per_epoch - cursor)
        for offset in range(limit):
            batch = shadow.next_batch()
            position = cursor + offset
            if position in self._plans and self._plans[position]['token'] != batch['token']:
                raise ValueError('Queued selected G5 changed')
            keys = [tuple(view['trajectory_ids']) for view in batch['grouping']['groups']]
            self._plans[position] = dict(token=batch['token'], keys=keys)
            for key in keys:
                if key not in self._prefetch:
                    self._prefetch[key] = self._pool.submit(self._decode, list(key))
                    self.prefetch_statistics['submitted_question_groups'] += 1
            shadow.pending = None
            shadow.accepted_updates += 1
        valid = {k for plan in self._plans.values() for k in plan['keys']}
        for key in list(self._prefetch):
            if key not in valid:
                self._prefetch.pop(key).cancel()

    def get(self, ids, device):
        ids = list(ids)
        if len(ids) % 5:
            raise ValueError('Selected-feature fetch must contain complete own-G5 groups')
        if not hasattr(self, '_prefetch'):
            return super().get(ids, device)
        chunks = [tuple(ids[i:i+5]) for i in range(0, len(ids), 5)]
        for key in chunks:
            if key not in self._prefetch:
                self._prefetch[key] = self._pool.submit(self._decode, list(key))
        start = time.perf_counter()
        batches = [self._prefetch.pop(key).result() for key in chunks]
        self.prefetch_statistics['wait_seconds'] += time.perf_counter() - start
        self.prefetch_statistics['consumed_question_groups'] += len(chunks)
        result = {key:torch.cat([b[key] for b in batches], dim=0).to(device)
                  for key in ('features', 'attention_mask', 'position_ids')}
        result.update(capture_layer=self.layer, capture_kind=CAPTURE_KIND,
                      model_revision=REVISION, trajectory_ids=ids,
                      qids=[qid for b in batches for qid in b['qids']])
        return result

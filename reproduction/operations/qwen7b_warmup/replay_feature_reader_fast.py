"""Byte-identical replay decode with bounded CPU caches and no per-row Torch ops.

The inherited constructor still verifies every sealed input file. Per-row SHA,
BF16 finiteness, own delta, masks and causal positions remain mandatory. This
module alone never mutates a model or starts a job.
"""
from collections import OrderedDict
import hashlib
import os
import zlib
import numpy as np
import torch
from .replay_feature_reader import ReplayFeatureReader
from .replay_probe_layers import WIDTH, QMAX, CAPTURE_KIND, REVISION


class FastReplayFeatureReader(ReplayFeatureReader):
    question_cache_limit = 64 * 1024**2

    def handles(self, root):
        old = set(self.cache)
        result = super().handles(root)
        for evicted in old.difference(self.cache):
            self.prototypes.pop(evicted, None)
        return result

    @staticmethod
    def span(stream, offset, size):
        payload = os.pread(stream.fileno(), size, offset)
        if len(payload) != size:
            raise ValueError('Replay packed file became truncated after validation')
        return payload

    def _base_question(self, root, row, streams, prototype):
        if not hasattr(self, '_question_bases'):
            self._question_bases = OrderedDict()
            self._question_base_bytes = 0
        key = (root, row['question_offset'], row['question_bytes'], row['question_length'])
        if key in self._question_bases:
            self._question_bases.move_to_end(key)
            return self._question_bases[key]
        payload = self.span(streams['question.bin'], row['question_offset'], row['question_bytes'])
        raw = zlib.decompress(payload)
        qlen = row['question_length']
        if len(raw) != qlen * self.dimension * 2:
            raise ValueError('Question decompressed BF16-bit shape differs')
        question = np.bitwise_xor(prototype[:qlen], np.frombuffer(raw, dtype=np.uint16).reshape(qlen, self.dimension))
        self._question_bases[key] = question
        self._question_base_bytes += question.nbytes
        while self._question_base_bytes > self.question_cache_limit and len(self._question_bases) > 1:
            _, old = self._question_bases.popitem(last=False)
            self._question_base_bytes -= old.nbytes
        return question

    def get(self, ids, device):
        # Reuse repeated question spans only within this get. The next get
        # re-reads disk, retaining corruption detection across invocations.
        self._question_bases = OrderedDict()
        self._question_base_bytes = 0
        ids = list(ids)
        if not ids or any(tid not in self.records for tid in ids):
            raise ValueError('Every requested replay trajectory must exist in the sealed bank')
        shape = (len(ids), WIDTH, self.dimension)
        features = np.zeros(shape, dtype=np.uint16)
        masks = np.zeros(shape[:2], dtype=np.bool_)
        positions = np.zeros(shape[:2], dtype=np.int64)
        qids = []
        for i, tid in enumerate(ids):
            root, row = self.records[tid]
            streams, prototype = self.handles(root)
            qlen, plen = row['question_length'], row['policy_length']
            question = self._base_question(root, row, streams, prototype)
            if row['delta_bytes']:
                payload = self.span(streams['delta.bin'], row['delta_offset'], row['delta_bytes'])
                raw = zlib.decompress(payload)
                if len(raw) != qlen * self.dimension * 2:
                    raise ValueError('Own-trajectory question delta shape differs')
                question = np.bitwise_xor(question, np.frombuffer(raw, dtype=np.uint16).reshape(qlen, self.dimension))
            policy_raw = self.span(streams['policy.bin'], row['policy_offset'], plen*self.dimension*2)
            digest = hashlib.sha256(question.tobytes())
            digest.update(policy_raw)
            if digest.hexdigest() != row['retained_feature_sha256']:
                raise ValueError('Reconstructed own retained feature SHA differs')
            policy = np.frombuffer(policy_raw, dtype=np.uint16).reshape(plen, self.dimension)
            # All BF16 infinities/NaNs have exponent11111111; exact bit test.
            if np.any((question & 0x7f80) == 0x7f80) or np.any((policy & 0x7f80) == 0x7f80):
                raise ValueError('Nonfinite reconstructed real BF16 feature')
            features[i, :qlen] = question
            features[i, QMAX:QMAX+plen] = policy
            masks[i, :qlen] = True
            masks[i, QMAX:QMAX+plen] = True
            positions[i, :qlen] = np.arange(qlen)
            positions[i, QMAX:QMAX+plen] = row['policy_positions']
            qids.append(str(row['qid']))
        return dict(features=torch.from_numpy(features).view(torch.bfloat16).to(device),
                    attention_mask=torch.from_numpy(masks).to(device),
                    position_ids=torch.from_numpy(positions).to(device),
                    capture_layer=self.layer, capture_kind=CAPTURE_KIND,
                    model_revision=REVISION, trajectory_ids=ids, qids=qids)

    def close(self):
        super().close()
        if hasattr(self, '_question_bases'):
            self._question_bases.clear()
            self._question_base_bytes = 0


class ParallelFastReplayFeatureReader(FastReplayFeatureReader):
    """Parallel CPU-only own-row decode, one ordered transfer per microbatch.

    No sampler or RNG is touched by workers. Each worker owns file handles and
    bounded caches; all use immutable, already validated metadata. Calls must
    originate from a single training/evaluation thread.
    """
    parallel_workers = 8

    def _local_reader(self):
        if not hasattr(self._locals, 'reader'):
            reader = FastReplayFeatureReader.__new__(FastReplayFeatureReader)
            reader.layer, reader.dimension = self.layer, self.dimension
            reader.records = self.records
            reader.cache, reader.prototypes = OrderedDict(), {}
            reader.synthetic_fixture, reader.closed = self.synthetic_fixture, False
            self._locals.reader = reader
            self._worker_readers.append(reader)
        return self._locals.reader

    def _decode(self, ids):
        return self._local_reader().get(ids, 'cpu')

    def get(self, ids, device):
        ids = list(ids)
        if len(ids) <= 10:
            return super().get(ids, device)
        if self.closed:
            raise ValueError('Replay reader is closed')
        if not hasattr(self, '_pool'):
            from concurrent.futures import ThreadPoolExecutor
            import threading
            if type(self.parallel_workers) is not int or not 1 <= self.parallel_workers <= 8:
                raise ValueError('Parallel decoder requires1..8 declared workers')
            self._locals = threading.local()
            self._worker_readers = []
            self._pool = ThreadPoolExecutor(max_workers=self.parallel_workers, thread_name_prefix='own-feature-decode')
        futures = [self._pool.submit(self._decode, ids[i:i+10]) for i in range(0,len(ids),10)]
        batches = [future.result() for future in futures]
        result = {key: torch.cat([b[key] for b in batches],dim=0).to(device)
                  for key in ('features','attention_mask','position_ids')}
        result.update(capture_layer=self.layer,capture_kind=CAPTURE_KIND,model_revision=REVISION,
                      trajectory_ids=ids,qids=[qid for b in batches for qid in b['qids']])
        return result

    def close(self):
        if hasattr(self, '_pool'):
            self._pool.shutdown(wait=True)
            for reader in self._worker_readers:
                reader.close()
            del self._pool
            self._worker_readers.clear()
        super().close()

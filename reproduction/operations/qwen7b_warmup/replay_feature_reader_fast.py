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


class PrefetchedReplayFeatureReader(ParallelFastReplayFeatureReader):
    """CPU producer reads up to8 future B32 batches while GPU consumes one.

    The lookahead is a disposable sampler copy within the CURRENT epoch; hard
    pools stay fixed until its boundary. It never observes fabricated losses or
    advances the real sampler. The real next-batch token is checked before use.
    Queue data is disposable and absent from optimizer/RNG/checkpoint state.
    """
    prefetch_batches = 8

    def prepare(self, sampler):
        import copy
        import random
        if sampler.pending is not None:
            raise ValueError('Prefetch only from an accepted real sampler boundary')
        cursor = sampler.accepted_updates
        if not hasattr(self, '_prefetch'):
            from concurrent.futures import ThreadPoolExecutor
            import threading
            self._locals = threading.local()
            self._worker_readers = []
            self._pool = ThreadPoolExecutor(max_workers=self.parallel_workers,thread_name_prefix='CPU-next-features')
            self._prefetch, self._plans = {}, {}
            self.prefetch_statistics = dict(submitted_question_groups=0,consumed_question_groups=0,wait_seconds=0.,prepare_calls=0)
        self.prefetch_statistics['prepare_calls'] += 1
        self._plans = {c:p for c,p in self._plans.items() if c >= cursor}
        if len(self._plans) >= self.prefetch_batches//2:
            return
        shadow = copy.copy(sampler)
        shadow.rng = random.Random()
        shadow.rng.setstate(sampler.rng.getstate())
        shadow.orders = {r:list(v) for r,v in sampler.orders.items()}
        shadow.positions, shadow.cycles = dict(sampler.positions), dict(sampler.cycles)
        shadow._partition_cache = dict(sampler._partition_cache)
        # These are only read by next_batch; no observe or epoch crossing occurs.
        shadow.pending = None
        limit = min(self.prefetch_batches,sampler.steps_per_epoch-cursor%sampler.steps_per_epoch)
        for offset in range(limit):
            batch = shadow.next_batch()
            position = cursor+offset
            if position in self._plans and self._plans[position]['token'] != batch['token']:
                raise ValueError('Lookahead changed an already queued exact sampler batch')
            self._plans[position] = dict(token=batch['token'],keys=[])
            for qid in batch['source_qids']:
                key = tuple(sampler.groups[qid]['trajectory_ids'])
                self._plans[position]['keys'].append(key)
                if key not in self._prefetch:
                    self._prefetch[key] = self._pool.submit(self._decode,list(key))
                    self.prefetch_statistics['submitted_question_groups'] += 1
            # Shadow bookkeeping for a future selection ONLY. No fake observed
            # loss, hard-pool update, real accepted update, or persisted state.
            shadow.pending = None
            shadow.accepted_updates += 1
        valid = {k for p in self._plans.values() for k in p['keys']}
        for key in list(self._prefetch):
            if key not in valid:
                future = self._prefetch.pop(key)
                future.cancel()

    def reset_prefetch(self):
        if hasattr(self,'_prefetch'):
            for future in self._prefetch.values():future.cancel()
            self._prefetch.clear()
            self._plans.clear()

    def accept_batch(self, batch):
        cursor = batch['before']['accepted_updates']
        if not hasattr(self,'_plans') or cursor not in self._plans or self._plans[cursor]['token'] != batch['token']:
            raise ValueError('Real sampler next batch differs from exact prefetched token')

    def get(self, ids, device):
        import time
        ids = list(ids)
        if not hasattr(self,'_prefetch'):
            return super().get(ids,device)
        chunks = [tuple(ids[i:i+10]) for i in range(0,len(ids),10)]
        for key in chunks:
            if key not in self._prefetch:
                self._prefetch[key] = self._pool.submit(self._decode,list(key))
        start = time.perf_counter()
        batches = [self._prefetch.pop(key).result() for key in chunks]
        self.prefetch_statistics['wait_seconds'] += time.perf_counter()-start
        self.prefetch_statistics['consumed_question_groups'] += len(chunks)
        result = {key:torch.cat([b[key] for b in batches],dim=0).to(device)
                  for key in ('features','attention_mask','position_ids')}
        result.update(capture_layer=self.layer,capture_kind=CAPTURE_KIND,model_revision=REVISION,
                      trajectory_ids=ids,qids=[qid for b in batches for qid in b['qids']])
        return result

    def close(self):
        super().close()
        if hasattr(self,'_prefetch'):
            self._prefetch.clear()
            self._plans.clear()

    @classmethod
    def from_inherited_seal(cls, manifests, capture_layer, seal):
        """Fresh metadata and every USED row SHA, explicit inherited payload seal.

        The owned continuation worker first restores and verifies the complete
        original checkpoint. Its immutable original config and accepted step
        prove that the unmodified full-SHA reader constructor had completed.
        This mode deliberately does NOT claim another fresh all276GB hash.
        """
        import json
        from .replay_probe_layers import fingerprints, MODEL, QMAX, PREFIX, CAPTURE_KIND
        from .replay_feature_reader import FILES
        from . import probe_ablation_train as u
        config_ref=seal['source_config'];source=u.read(u.checked(config_ref))
        checkpoint=seal['source_checkpoint'];receipt=seal['restoration_receipt']
        if (config_ref['sha256']!='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393'
            or source['initialization_source']!='official_initial' or checkpoint['cursor']<128
            or checkpoint['native_steps']!=checkpoint['cursor'] or checkpoint['head_steps']!=checkpoint['cursor']
            or checkpoint['config_sha256']!=config_ref['sha256']
            or not checkpoint['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']
            or receipt.get('full_checkpoint_fresh_SHA_and_nine_state_restore_passed') is not True
            or receipt['accepted_hashes']!=checkpoint['accepted_hashes']
            or u.digest(manifests)!=u.digest(u.read(u.checked(source['features'])))):
            raise ValueError('Accepted original full-SHA constructor/complete state lineage missing')
        obj=cls.__new__(cls)
        obj.layer,obj.dimension=capture_layer,3584
        obj.records,obj.cache,obj.prototypes={},OrderedDict(),{}
        obj.manifests,obj.files=[],[]
        obj.synthetic_fixture,obj.closed=False,False
        questions={}
        for descriptor in manifests:
            path=obj.path(descriptor['path']);actual=fingerprints(path)
            if actual!=descriptor:raise ValueError('Fresh replay manifest SHA/bytes differ')
            manifest=json.loads(path.read_bytes())
            if (manifest['status']!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE'
                or manifest['source_runtime']!='huggingface_frozen_causal_prefill'
                or manifest['layout']!='lossless_question_base_xor_policy_v1'
                or manifest['capture_layer']!=capture_layer or manifest['feature_layer']!=capture_layer
                or manifest['capture_kind']!=CAPTURE_KIND or manifest['dimension']!=3584
                or manifest['question_max_tokens']!=QMAX or manifest['prefix_tokens']!=PREFIX
                or manifest['model']!=MODEL or manifest['model_revision']!=REVISION
                or manifest['actor_optimizer_updates']!=0 or manifest['bit_exact_native_vllm_claimed'] is not False
                or manifest.get('feature_dtype','bfloat16_bits_uint16')!='bfloat16_bits_uint16'
                or set(manifest['files'])!=FILES):
                raise ValueError('Inherited replay causal/source/geometry contract differs')
            for name,ref in manifest['files'].items():
                payload=obj.path(ref['path'])
                if payload!=path.parent/name or payload.stat().st_size!=ref['bytes']:
                    raise ValueError('Replay payload path/actual bytes differ')
                if name=='index.jsonl' and fingerprints(payload)!=ref:
                    raise ValueError('Fresh replay index SHA/bytes differ')
                obj.files.append(dict(ref,validation='fresh_SHA' if name=='index.jsonl' else 'inherited_full_payload_seal_and_each_used_retained_row_SHA'))
            if manifest['files']['prototype.bin']['bytes']!=QMAX*3584*2:
                raise ValueError('Prototype shape differs')
            count=0
            with(path.parent/'index.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line);obj.validate_row(row,manifest);tid=row['trajectory_id'];qid=str(row['qid'])
                    if tid in obj.records:raise ValueError('Duplicate retained trajectory ID')
                    binding=(row['question_length'],row['prompt_sha256'])
                    if questions.setdefault(qid,binding)!=binding:raise ValueError('Question source identity changed')
                    obj.records[tid]=(str(path.parent),row);count+=1
            if count!=manifest['count']:raise ValueError('Replay index count differs')
            obj.manifests.append(actual)
        obj.proof=dict(status='INHERITED_FULL_PAYLOAD_SEAL_FRESH_MANIFEST_INDEX_SIZE_AND_EACH_USED_ROW_SHA',
            fresh_all_payload_SHA_claimed=False,background_full_scan_started=False,
            source_config=config_ref,source_checkpoint=checkpoint,restoration_receipt=receipt,
            count=len(obj.records),capture_layer=capture_layer,dimension=3584,
            manifest_count=len(obj.manifests),file_count=len(obj.files),
            prefetch_batches=obj.prefetch_batches,CPU_producer_workers=obj.parallel_workers,
            max_prefetched_feature_bytes=8*32*10*293*3584*2)
        return obj

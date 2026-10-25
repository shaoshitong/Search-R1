"""Durable, idle-engine collection commits before any actor write.

A commit atomically binds candidate identities, original trajectory deltas,
Backstop ledger, driver/worker RNG and probe state. In-flight work after the
last commit may be replayed; committed questions are never sampled again.
"""
import copy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import random
import uuid


def snapshot_states(states):
    """Serialize fields synchronously without recursively copying native tokens.

    PrefixState fields contain ordinary lists/dicts/scalars. The caller passes
    this directly to atomic_save, which finishes serialization before the
    collector can mutate another state. Preserve shared immutable provenance
    objects for the pickle memo rather than cloning them once per trajectory.
    """
    return [{field.name: getattr(state, field.name) for field in fields(state)}
            for state in states]

from .mixed_backstop import _atomic_json


def load_reference(reference):
    import torch
    path = Path(reference['path'])
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    if h.hexdigest() != reference['sha256']:
        raise ValueError('Collection snapshot SHA changed: ' + str(path))
    return torch.load(path, map_location='cpu', weights_only=False)


def producer(identity):
    from reproduction.common8.behavior_trace import document_sha256
    return dict(run_id=identity['run_id'], slurm_job_id=identity['slurm_job_id'],
                source_map_sha256=document_sha256(identity['source_sha256']))


def find_resume(base, run, outer, prior_writes):
    roots = [Path(base)]
    # Actor state and unfinished collection can originate in different jobs:
    # a restored actor may have collected its next batch before failing with
    # zero new optimizer writes. Pin that terminal attempt's small pointer;
    # retain the actual tensor parts and producer identity at their origin.
    origin_path = Path(base)/'preparation/collection-resume-origin.json'
    origin = json.loads(origin_path.read_text()) if origin_path.is_file() else None
    if origin is not None and origin['outer'] == outer:
        root = Path(origin['source_base'])
        pointer = root/'preparation/dynrpo7b-collection'/('outer-%04d.json' % outer)
        if hashlib.sha256(pointer.read_bytes()).hexdigest() != origin['pointer_sha256']:
            raise ValueError('Pinned collection origin changed')
        roots.append(root)
    binding = run.get('full_state_restore', {})
    fork = run.get('objective_transition', {})
    skip_parent_suffix = (fork.get('old_run_suffix_not_imported') is True and
                          fork.get('source_base') == binding.get('source_base'))
    if binding.get('source_base') and not skip_parent_suffix:
        roots.append(Path(binding['source_base']))
    for root in roots:
        pointer = root/'preparation/dynrpo7b-collection'/('outer-%04d.json' % outer)
        if not pointer.exists():
            continue
        value = json.loads(pointer.read_text())
        if (value['run_id'], value['outer'], value['prior_writes']) != (run['run_id'], outer, prior_writes):
            raise ValueError('Collection commit and restored actor counters differ')
        # Never replay a collection whose actor has already made even one write.
        for path in (root/'logs').glob('optimizer-rank-*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get('kind') == 'OPTIMIZER_WRITE' and row.get('write_count', row.get('optimizer_writes_after', 0)) > prior_writes:
                    raise ValueError('Collection has later real Adam writes; restore that actor instead')
                if row.get('kind') == 'OPPORTUNITY' and row.get('optimizer_writes_after', 0) > prior_writes:
                    raise ValueError('Collection has later real Adam writes; restore that actor instead')
        return value
    return None


def begin_attempt(trainer, progress):
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    resume = find_resume(base, run, trainer.global_steps, progress.cumulative_writes)
    trainer._dynrpo7b_collection_resume = resume
    if resume is None:
        return progress.begin()
    attempt = resume['attempt']
    if progress.pending is not None or attempt['attempt_index'] != progress.attempt_index + 1:
        raise ValueError('Collection attempt is not the next unused actor attempt')
    progress.attempt_index = attempt['attempt_index']
    progress.pending = copy.deepcopy(attempt)
    return copy.deepcopy(attempt)


def restore_worker_rng(worker, references):
    import numpy as np
    import torch
    from search_g1.training_state import cpu_tree, worker_rng
    from search_g1.full_state_restore import equal_tree
    worker._assert_generation_phase_inactive()
    saved = load_reference(next(r for r in references if r['rank'] == worker.rank))
    run = json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
    if (saved['run_id'] != run['run_id'] or saved['rank'] != worker.rank or saved['world_size'] != 4 or
            saved['actual_optimizer_writes'] != getattr(worker.actor, '_optimizer_writes', 0) or
            saved['completed_outer'] != getattr(worker.actor, '_complete_outers', 0)):
        raise ValueError('Collection sampling RNG belongs to a different actor state')
    rng = saved['rng']; manager = worker.rollout_sharding_manager
    engine = worker.rollout.inference_engine.llm_engine
    if engine.has_unfinished_requests():
        raise ValueError('Cannot restore sampling RNG into an active engine')
    generators = engine.model_executor.worker.model_runner.get_generators()
    generators.clear()
    for key, value in rng['request_generators'].items():
        generator = torch.Generator(device=value['device']); generator.set_state(value['state'])
        generators[key] = generator
    manager.torch_random_states = cpu_tree(rng['actor_cuda_saved'])
    manager.gen_random_states = cpu_tree(rng['rollout_cuda_saved'])
    torch.set_rng_state(rng['torch_cpu'])
    torch.cuda.set_rng_state(rng['torch_cuda'], device=torch.cuda.current_device())
    np.random.set_state(rng['numpy']); random.setstate(rng['python'])
    equal_tree(worker_rng(worker), rng)
    return dict(rank=worker.rank, collection_rng_equal=True,
                actual_optimizer_writes=saved['actual_optimizer_writes'])


class CollectionJournal:
    def __init__(self, trainer, attempt, directory, resume=None):
        self.trainer, self.attempt = trainer, attempt
        self.base = Path(os.environ['SEARCH_G1_BASE'])
        self.run = json.loads((self.base/'logs/run-manifest.json').read_text())
        self.directory = Path(directory)
        self.pointer = self.base/'preparation/dynrpo7b-collection'/('outer-%04d.json' % trainer.global_steps)
        self.value = copy.deepcopy(resume) if resume else dict(
            schema_version=1, run_id=self.run['run_id'], outer=trainer.global_steps,
            prior_writes=trainer.inline_attempt_progress.cumulative_writes,
            attempt=attempt, parts=[], probe_ref=None)

    def part(self, **payload):
        from search_g1.training_state import atomic_save
        return atomic_save(self.directory/('part-'+uuid.uuid4().hex+'.pt'), payload)

    def commit(self, stage, stream, collector, *, part=None, ledger=None, extra=None):
        from search_g1.training_state import atomic_save, compact_stream_state, driver_rng
        runtime = self.trainer.qprobe_runtime
        if stage != 'candidates' and runtime._probe_ref is None:
            path = self.directory/('probe-'+uuid.uuid4().hex+'.pt')
            runtime._probe_ref = runtime.rpc('state_dict', dict(path=str(path)))[0]['reference']
        worker_dir = self.directory/('rng-'+uuid.uuid4().hex)
        workers = self.trainer.actor_rollout_wg.capture_mixed_state(str(worker_dir))
        if part is not None:
            self.value['parts'].append(part)
        checkpoint = dict(stage=stage, identity=collector.identity,
            stream=compact_stream_state(stream.state()), rng=driver_rng(), workers=workers,
            probe_ref=runtime._probe_ref, probe_version=runtime.version,
            collector=dict(meta=collector.meta, sampling=collector.sampling,
                phase_events=collector.phase_events,
                raw_generated_policy_tokens=collector.raw_generated_policy_tokens),
            ledger=copy.deepcopy(ledger.state) if ledger else None,
            **(extra or {}))
        self.value.update(stage=stage, probe_ref=runtime._probe_ref,
            checkpoint=atomic_save(self.directory/('commit-'+uuid.uuid4().hex+'.pt'), checkpoint))
        _atomic_json(self.pointer, self.value)
        return checkpoint

    def restore(self, stream, collector):
        import numpy as np
        import torch
        from search_g1.qprobe_runtime import QProbeCandidateStream
        saved = load_reference(self.value['checkpoint'])
        original = saved['identity']
        if (original['run_id'] != self.run['run_id'] or original['attempt_id'] != self.attempt['attempt_id']):
            raise ValueError('Committed candidate identity changed')
        # Original native chunks retain their own producer records. Only the
        # consumer/execution identity changes; no raw token or proof is relabeled.
        producers = list(original.get('generation_producers', [producer(original)]))
        current = dict(original, source_sha256=self.run['source_sha256'],
                       slurm_job_id=os.environ['SLURM_JOB_ID'])
        current_producer = producer(current)
        if current_producer not in producers:
            producers.append(current_producer)
        current['generation_producers'] = producers
        current['collection_resume_reference'] = self.value['checkpoint']
        collector.identity = current
        for name, value in saved['collector'].items():
            setattr(collector, name, value)
        self.trainer.actor_rollout_wg.restore_dynrpo7b_collection_rng(saved['workers'])
        runtime = self.trainer.qprobe_runtime
        if saved['probe_ref']:
            runtime.initialise()
            runtime.rpc('load_state_dict', dict(reference=saved['probe_ref']))
            runtime._probe_ref = saved['probe_ref']
        runtime.version = saved['probe_version']
        restored = QProbeCandidateStream(stream.loader, restored_state=saved['stream'])
        stream._source = restored._source
        torch.set_rng_state(saved['rng']['torch_cpu'])
        np.random.set_state(saved['rng']['numpy']); random.setstate(saved['rng']['python'])
        return saved


def authenticated_producers(identity):
    """Bind historical native producers to the actual hashed collection commit."""
    reference = identity.get('collection_resume_reference')
    if not reference:
        if identity.get('generation_producers'):
            raise ValueError('Historical generation producers need a collection commit')
        return None
    saved = load_reference(reference)
    old = saved['identity']
    for key in ('run_id','policy_snapshot_id','attempt_id','attempt_index','logical_update'):
        if identity[key] != old[key]:
            raise ValueError('Historical trajectory provenance changed: ' + key)
    producers = list(old.get('generation_producers', [producer(old)]))
    current = producer(identity)
    if current not in producers:
        producers.append(current)
    if identity['generation_producers'] != producers:
        raise ValueError('Historical producer list was not committed')
    return producers

"""One extra frozen-teacher outer: 5120 complete G5 groups, then head fit.

The reusable cache contains actual trajectories and all five strict EM labels.
It is neither a PPO update nor part of the 200 successful training outers.
Completed chunks are immutable; restart never regenerates accepted chunks.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import time

from search_g1.qprobe_features import sha_file, sha_json

QUESTIONS = 5120
GROUP_SIZE = 5
CHUNK_SIZE = 512
STATUS = 'FROZEN_TEACHER_WARMUP_5120_G5_FULL_SHA_PASS'


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('WARMUP_IMMUTABLE_FILE_DIFFERS: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def moments(vector):
    if not isinstance(vector, list) or len(vector) != GROUP_SIZE or any(
            type(x) is not int or x not in (0, 1) for x in vector):
        raise ValueError('WARMUP_REQUIRES_FIVE_ACTUAL_BINARY_EM_LABELS')
    count = sum(vector)
    return count, count / GROUP_SIZE, math.sqrt(count * (GROUP_SIZE-count) / 20)


def verify_frozen(before, after, expected_world_size=8):
    if type(expected_world_size) is not int or expected_world_size not in (4,8):
        raise ValueError('WARMUP_INVALID_BOUND_WORKER_TOPOLOGY')
    before=sorted(before,key=lambda x:x['rank']);after=sorted(after,key=lambda x:x['rank'])
    keys=('rank','parameter_sha256','optimizer_sha256','scheduler_sha256','actual_optimizer_writes','completed_outer')
    if [x['rank'] for x in before] != list(range(expected_world_size)) or [x['rank'] for x in after] != list(range(expected_world_size)) or any(
            {k:a[k] for k in keys} != {k:b[k] for k in keys} or a['actual_optimizer_writes'] or a['completed_outer']
            for a,b in zip(before,after)):
        raise ValueError('WARMUP_TEACHER_MODEL_OR_OPTIMIZER_WAS_NOT_FROZEN')
    return dict(before=before,after=after,actual_frozen_rank_readback=True,actual_world_size=expected_world_size,
                actual_eight_rank_frozen_readback=expected_world_size==8)


def teacher_identity(manifest, *, feature_spec_sha, sampling_config, agent_config,
                     max_turns, seed, questions):
    """A reusable official-teacher namespace, independent of a fresh run's ID."""
    from search_g1.qprobe_runtime import strict_scoring_sources
    scorer_sources = strict_scoring_sources()
    if any(manifest['source_sha256'].get(name) != sha for name, sha in scorer_sources.items()):
        raise ValueError('WARMUP_ACTUAL_STRICT_SCORER_DIFFERS_FROM_BOUND_SOURCE')
    return dict(schema=2, model_revision=manifest['model_revision'],
        teacher_frozen_evidence_required=True, training_world_size=manifest.get('training_world_size',8),
        model_files_sha256=manifest['initial_model_sha256'],
        train_parquet_sha256=manifest['input_sha256']['train.parquet'],
        data_manifest_sha256=manifest['budget']['common_manifest_sha256'],
        retriever_version=manifest['retriever_version'], feature_spec_sha=feature_spec_sha,
        sampling_config=sampling_config, agent_config=agent_config, max_turns=max_turns,
        seed=seed, questions=questions, scorer_sources=scorer_sources,
        rollout_sources={name:value for name,value in manifest['source_sha256'].items()
            if name.startswith(('search_r1/llm_agent/','verl/workers/rollout/vllm_rollout/')) or name in (
                'reproduction/common8/generation_behavior.py','reproduction/common8/behavior_runtime.py',
                'reproduction/common8/context_budget.py',
                'reproduction/common8/sampling_capture.py',
                'search_g1/async_rollout_runtime.py','search_g1/rollout_phase_contract.py',
                'verl/workers/fsdp_workers.py','verl/workers/sharding_manager/fsdp_vllm.py',
                'search_g1/qprobe_teacher_worker.py','search_g1/qprobe_warmup.py')})


class TeacherCache:
    def __init__(self, root, identity):
        self.root = Path(root).resolve()
        if not str(self.root).startswith(('/data/', '/mnt/', '/backup/')):
            raise ValueError('WARMUP_OUTPUT_MUST_BE_EXPLICIT_REMOTE_STORAGE')
        self.identity = identity
        if len(identity['questions']) != QUESTIONS or len({q['qid'] for q in identity['questions']}) != QUESTIONS:
            raise ValueError('WARMUP_REQUIRES_5120_UNIQUE_CANONICAL_TRAIN_QUESTIONS')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'writer.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        header = self.root / 'identity.json'
        if header.exists():
            if json.loads(header.read_text()) != identity:
                raise ValueError('WARMUP_TEACHER_DATA_SAMPLING_OR_SCORING_NAMESPACE_CHANGED')
        else:
            write_once(header, identity)

    def close(self):
        fcntl.flock(self.lock, fcntl.LOCK_UN); self.lock.close()

    def read_chunk(self, index):
        path = self.root / ('chunk-%02d.json' % index)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        self.validate_chunk(index, value)
        return value

    def validate_chunk(self, index, value):
        if value.get('index') != index or len(value.get('groups', [])) != CHUNK_SIZE:
            raise ValueError('WARMUP_INCOMPLETE_CHUNK')
        if self.identity.get('teacher_frozen_evidence_required'):
            frozen=value['frozen_workers']
            verify_frozen(frozen['before'],frozen['after'],self.identity.get('training_world_size',8))
        expected = self.identity['questions'][index*CHUNK_SIZE:(index+1)*CHUNK_SIZE]
        ids = set()
        for question, group in zip(expected, value['groups']):
            if any(group.get(k) != question[k] for k in ('qid', 'question_hash')):
                raise ValueError('WARMUP_QUESTION_BINDING_DIFFERS')
            count, mean, std = moments(group['strict_em'])
            if (group['correct_count'], group['mean'], group['std']) != (count, mean, std):
                raise ValueError('WARMUP_SAVED_MOMENTS_DO_NOT_MATCH_ACTUAL_EM')
            trajectories = group.get('trajectory_ids', [])
            if len(trajectories) != 5 or len(set(trajectories)) != 5 or ids.intersection(trajectories):
                raise ValueError('WARMUP_DUPLICATE_OR_MISSING_TRAJECTORY_IDS')
            ids.update(trajectories)
        proof = value['raw_rollout']
        path = self.root / proof['relative_path']
        if path.resolve().parent != self.root or path.stat().st_size != proof['bytes'] or sha_file(path) != proof['sha256']:
            raise ValueError('WARMUP_ACTUAL_ROLLOUT_FULL_SHA_DIFFERS')

    def commit_chunk(self, index, groups, raw_rollout, contracts, frozen_workers=None):
        value = dict(index=index, groups=groups, raw_rollout=raw_rollout, contracts=contracts)
        if frozen_workers is not None:value['frozen_workers']=frozen_workers
        self.validate_chunk(index, value)
        write_once(self.root / ('chunk-%02d.json' % index), value)
        print(json.dumps(dict(stage='frozen_teacher_warmup_chunk_complete',
            completed_questions=(index+1)*CHUNK_SIZE, total_questions=QUESTIONS,
            completed_trajectories=(index+1)*CHUNK_SIZE*GROUP_SIZE)), flush=True)
        return value

    def verify_complete(self):
        chunks = [self.read_chunk(i) for i in range(QUESTIONS//CHUNK_SIZE)]
        if any(c is None for c in chunks):
            raise ValueError('WARMUP_ALL_5120_TEACHER_GROUPS_NOT_COMPLETE')
        groups = [group for c in chunks for group in c['groups']]
        ids = [tid for g in groups for tid in g['trajectory_ids']]
        if len(ids) != QUESTIONS*GROUP_SIZE or len(set(ids)) != len(ids):
            raise ValueError('WARMUP_FULL_POOL_TRAJECTORY_ID_COLLISION')
        protocol = {k: chunks[0]['contracts'][k] for k in (
            'sampling_contract_sha', 'strict_scoring_contract_sha', 'retriever_version')}
        if any({k: c['contracts'][k] for k in protocol} != protocol for c in chunks):
            raise ValueError('WARMUP_CHUNKS_HAVE_DIFFERENT_ACTUAL_PROTOCOLS')
        receipt = dict(status=STATUS, questions=QUESTIONS, trajectories=len(ids),
            identity_sha256=sha_file(self.root/'identity.json'), protocol=protocol,
            chunks=[dict(relative_path='chunk-%02d.json'%i,
                         sha256=sha_file(self.root/('chunk-%02d.json'%i))) for i in range(len(chunks))])
        write_once(self.root/'complete.json', receipt)
        return receipt, groups, chunks


def run(trainer, manager, runtime, stream):
    """Called once before formal outer1; never calls update_actor or rewards."""
    cache_path = runtime.config.get('teacher_warmup_cache')
    if not cache_path:
        return None
    import copy
    import numpy as np
    import random
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from reproduction.common8.behavior_runtime import prepare_identity, attach_generation_identity
    from search_g1.mixed_runtime import strict_vectors
    from search_g1.qprobe_features import original_question_and_id, question_text_sha
    from search_g1.qprobe_runtime import _complete_generation, _contracts
    from search_g1.qprobe_selection import QuestionPool
    from search_g1.training_state import atomic_save, driver_rng

    if runtime.completed_outer != 0 or trainer.global_steps != 1 or runtime.head.update_count:
        raise ValueError('FROZEN_TEACHER_WARMUP_ONLY_PRECEDES_A_FRESH_ACTOR')
    if runtime.head.config.bootstrap_max_groups_per_outer != QUESTIONS:
        raise ValueError('WARMUP_HEAD_MUST_FIT_ALL_5120_GROUPS_WITHOUT_CAPPING')
    base = Path(os.environ['SEARCH_G1_BASE'])
    manifest = json.loads((base/'logs/run-manifest.json').read_text())
    before_driver = driver_rng()
    boundary = runtime.output_dir/'teacher-warmup-boundary'
    before_workers = trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(boundary/'before'))
    pool = QuestionPool(QUESTIONS)
    rows = []
    for index in np.random.default_rng(runtime.config['seed']).permutation(len(stream.loader.dataset)):
        original = stream.loader.dataset.dataframe.iloc[int(index)]
        question, qid = original_question_and_id(original, stream.loader.dataset.prompt_key)
        if pool.reserve(qid, question) is not None:
            rows.append((int(index), question, qid))
        if len(rows) == QUESTIONS:
            break
    if len(rows) != QUESTIONS:
        raise ValueError('WARMUP_CANONICAL_UNIQUE_QUESTION_POOL_EXHAUSTED')
    parents = DataProto.from_dict(tensors={'qprobe_source_index':torch.tensor([x[0] for x in rows])},
        non_tensors={'candidate_original_question':np.array([x[1] for x in rows], dtype=object),
                     'candidate_stable_id':np.array([x[2] for x in rows], dtype=object)})
    identity = teacher_identity(manifest, feature_spec_sha=runtime.feature_spec_sha,
        sampling_config=OmegaConf.to_container(trainer.config.actor_rollout_ref.rollout, resolve=True),
        agent_config={k: getattr(trainer.config.data, k) for k in (
            'max_start_length','max_prompt_length','max_response_length','max_obs_length')},
        max_turns=trainer.config.max_turns, seed=runtime.config['seed'],
        questions=[dict(qid=qid, question_hash=question_text_sha(q)) for _,q,qid in rows])
    cache = TeacherCache(cache_path, identity)
    try:
        reused = 0
        for chunk in range(QUESTIONS//CHUNK_SIZE):
            saved = cache.read_chunk(chunk)
            if saved is not None:
                reused += CHUNK_SIZE
                continue
            ordinals = list(range(chunk*CHUNK_SIZE,(chunk+1)*CHUNK_SIZE))
            parent = stream.materialize(parents, ordinals)
            raw_path = cache.root/('rollout-%02d.pt'%chunk)
            if raw_path.exists():
                raw = torch.load(raw_path, map_location='cpu', weights_only=False)
                output, generation_id, contracts = raw['output'], raw['identity'], raw['contracts']
                frozen=raw['frozen_workers'];verify_frozen(frozen['before'],frozen['after'],identity['training_world_size'])
                if raw['question_bindings'] != identity['questions'][ordinals[0]:ordinals[-1]+1]:
                    raise ValueError('WARMUP_UNCOMMITTED_REAL_ROLLOUT_QUESTIONS_DIFFER')
            else:
                generation_id = prepare_identity(parent, 0, repeats=5, teacher_warmup=True,
                    candidate_offset=ordinals[0])
                repeated = parent.repeat(repeat_times=5, interleave=True)
                gen = repeated.pop(batch_keys=['input_ids','attention_mask','position_ids'])
                attach_generation_identity(gen, generation_id, do_sample=True)
                manager.timing_raw = {}
                output = manager.run_llm_loop(gen_batch=gen,
                    initial_input_ids=gen.batch['input_ids'][:,-manager.config.max_start_length:].clone().long())
                _complete_generation(output, generation_id)
                contracts = _contracts(trainer, output, manifest)
                chunk_after=trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(boundary/('chunk-%02d'%chunk)))
                frozen=verify_frozen(before_workers,chunk_after,identity['training_world_size'])
                atomic_save(raw_path, dict(output=output, identity=generation_id, contracts=contracts,
                    frozen_workers=frozen,
                    question_bindings=identity['questions'][ordinals[0]:ordinals[-1]+1]))
            _complete_generation(output, generation_id)
            vectors = strict_vectors(parent, output, trainer.tokenizer)
            groups = []
            for local, vector in enumerate(vectors):
                count, mean, std = moments(vector)
                groups.append(dict(**identity['questions'][ordinals[local]], strict_em=vector,
                    correct_count=count, mean=mean, std=std,
                    trajectory_ids=generation_id['trajectory_ids'][local*5:(local+1)*5],
                    policy_snapshot_id=generation_id['policy_snapshot_id'], source_outer=0,
                    run_id=generation_id['run_id'], attempt_id='teacher-warmup-%02d'%chunk))
            cache.commit_chunk(chunk, groups, dict(relative_path=raw_path.name,
                bytes=raw_path.stat().st_size, sha256=sha_file(raw_path)), contracts, frozen)
        proof, groups, chunks = cache.verify_complete()
        labels = [dict(g, complete=True, selection_source='bootstrap_all_candidate',
            **proof['protocol']) for g in groups]
        features = runtime.cache.get_features([x[2] for x in rows], [x[1] for x in rows])
        fitted = runtime.head.bootstrap(labels, features, actor_outer=0,
            actor_snapshot_id=manifest['run_id']+':0')
        if fitted['bootstrap_groups'] != QUESTIONS:
            raise ValueError('WARMUP_HEAD_DID_NOT_FIT_ALL_5120_GROUPS')
        head_path = runtime.output_dir/'teacher-warmup-head/initial.pt'
        head_path.parent.mkdir(parents=True, exist_ok=True)
        head_receipt = runtime.head.save(head_path, actor_snapshot_id=manifest['run_id']+':0')
        after_workers = trainer.actor_rollout_wg.capture_frozen_teacher_boundary(str(boundary/'after'))
        verify_frozen(before_workers,after_workers,identity['training_world_size'])
        restored = trainer.actor_rollout_wg.restore_frozen_teacher_rng(str(boundary/'before'))
        if len(restored) != identity['training_world_size'] or not all(x['rng_equal'] for x in restored):
            raise ValueError('WARMUP_DID_NOT_RESTORE_INDEPENDENT_ACTOR_RNG_BOUNDARY')
        result = dict(status='FROZEN_TEACHER_WARMUP_HEAD_READY', cache_receipt=proof,
            reused_questions=reused, fitted_questions=fitted['bootstrap_groups'],
            head_metrics=fitted, head_checkpoint=head_receipt, before_workers=before_workers,
            after_workers=after_workers, actor_rng_restored=restored,
            extra_warmup_outers=1, completed_training_outers=0, target_training_outers=200,
            checked_unix=time.time())
        write_once(runtime.output_dir/'teacher-warmup-ready.json', result)
        print(json.dumps(dict(stage='frozen_teacher_warmup_head_ready', fitted_questions=QUESTIONS,
            reused_questions=reused, completed_training_outers=0, target_training_outers=200)), flush=True)
        return result
    finally:
        cache.close()
        torch.set_rng_state(before_driver['torch_cpu'])
        np.random.set_state(before_driver['numpy']); random.setstate(before_driver['python'])

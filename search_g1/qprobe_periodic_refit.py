"""Checkpoint-first collection and continuous twenty-epoch native-probe fit."""
import copy
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from search_g1.qprobe_periodic_model import PeriodicQwenProbe, restore_engine, fit_phase
from search_g1.qprobe_periodic_features import encode
from search_g1.qprobe_runtime import _sha, _write_json, _contracts, _complete_generation


def _restore_driver_rng(state):
    torch.set_rng_state(state['torch_cpu'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def run(trainer, manager, runtime, checkpoint):
    from reproduction.hpc3_dynrpo_periodic.warmup import save
    from reproduction.hpc3_dynrpo_periodic.common import publish
    from reproduction.common8.behavior_runtime import prepare_identity, attach_generation_identity
    from search_g1.qprobe_checkpoint import publish_completed_pair, read_source
    from search_g1.training_state import driver_rng, compact_stream_state
    from search_g1.qprobe_features import original_question_and_id
    from search_g1.qprobe_selection import PredictQuestion
    from search_g1.mixed_runtime import strict_vectors
    from verl import DataProto
    outer = runtime.completed_outer
    directory = runtime.output_dir/'periodic-refits'/str(outer)
    directory.mkdir(parents=True,exist_ok=True)
    complete = directory/'complete.json'
    actor_id = json.loads((runtime.base/'logs/run-manifest.json').read_text())['run_id']+':'+str(outer)
    if complete.exists():
        accepted = json.loads(complete.read_text())
        state_path = directory/'post-probe-state.pt'
        if _sha(state_path) != accepted['post_state_sha256']:
            raise ValueError('PERIODIC_ACCEPTED_POST_STATE_CHANGED')
        runtime.restore(state_path,actor_id)
        runtime.post_refit_state_path = str(state_path)
        return accepted
    pre_receipt = runtime.output_dir/'pending-probe-pairs'/('global_step_'+str(outer))/'source-receipt.json'
    if not pre_receipt.exists():
        if runtime.execution_phase != 'PROBE_PENDING':
            raise ValueError('PERIODIC_REFIT_REQUIRES_CHECKPOINTED_PENDING_STAGE')
        publish_completed_pair(trainer,checkpoint,periodic_phase='before_probe')
    _,pre,_,_,_ = read_source(pre_receipt)
    if pre['execution_phase'] != 'PROBE_PENDING' or pre['step'] != outer:
        raise ValueError('PERIODIC_PRE_PROBE_FULL_CHECKPOINT_NOT_ACCEPTED')
    state_path = directory/'latest-state.pt'
    manifest = directory/'latest-state.json'
    ledger = directory/'update-ledger.json'
    if manifest.exists():
        pointer = json.loads(manifest.read_text())
        if _sha(state_path) != pointer['sha256']:
            raise ValueError('PERIODIC_PHASE_RECOVERY_SHA_CHANGED')
        saved = torch.load(state_path,map_location='cpu',weights_only=False)
        if saved['outer'] != outer or saved['pre_receipt_sha256'] != _sha(pre_receipt):
            raise ValueError('PERIODIC_PHASE_BELONGS_TO_DIFFERENT_STUDENT_CHECKPOINT')
        if ledger.exists() and json.loads(ledger.read_text())['possible_adam_steps'] > saved['head']['update_count']:
            raise ValueError('PERIODIC_UNSAVED_ADAM_WRITES_AFTER_HARD_KILL_REQUIRE_EXPLICIT_DECISION')
        runtime.head = PeriodicQwenProbe(restore_engine(saved['head'],device=runtime.config['head_device']),
            runtime.feature_spec_sha,completed_outer=outer,actor_snapshot_id=actor_id,
            phases=runtime.head.phases)
        phase,indices,labels = saved['phase'],saved['indices'],saved['labels']
        frozen_before = saved['frozen_before']
        runtime.refit_rng.bit_generator.state = saved['refit_rng']
        for row in saved['worker_rng']:
            if _sha(row['path']) != row['sha256']:
                raise ValueError('PERIODIC_PHASE_WORKER_RNG_SHA_CHANGED')
        restored = trainer.actor_rollout_wg.restore_periodic_probe_rng(str(Path(saved['worker_rng'][0]['path']).parent))
        if len(restored) != 4 or any(not row['rng_equal'] for row in restored):
            raise ValueError('PERIODIC_PHASE_WORKER_RNG_RESTORE_FAILED')
        _restore_driver_rng(saved['driver_rng'])
        feature_hashes = saved.get('feature_hashes')
    else:
        indices = runtime.refit_rng.choice(len(trainer.train_dataloader.dataset),size=5120,replace=False).tolist()
        labels,feature_hashes = [],None
        phase = dict(start_steps=runtime.head._adam_steps(),epoch=0,batch=0,order=None,
                     minibatches=[],summaries=[])
        frozen_before = trainer.actor_rollout_wg.capture_periodic_probe_boundary(str(directory/'frozen-before'))
        saved = dict(worker_rng=[row['rng_receipt'] for row in frozen_before])
    if len(indices) != len(set(indices)) or len(indices) != 5120:
        raise ValueError('PERIODIC_REFIT_REQUIRES_ALL_5120_UNIQUE_ROWS')
    questions = []
    dataset = trainer.train_dataloader.dataset
    for ordinal,index in enumerate(indices):
        question,qid = original_question_and_id(dataset.dataframe.iloc[index],dataset.prompt_key)
        questions.append(PredictQuestion(ordinal,qid,question))
    if len({q.qid for q in questions}) != 5120 or len({q.content_hash for q in questions}) != 5120:
        raise ValueError('PERIODIC_REFIT_QUESTION_IDENTITIES_NOT_UNIQUE')
    worker_rng = saved['worker_rng']
    def persist(reason, state=None):
        payload = dict(outer=outer,pre_receipt_sha256=_sha(pre_receipt),
            head=runtime.head._engine.state_dict(),phase=phase,indices=indices,labels=labels,
            frozen_before=frozen_before,feature_hashes=feature_hashes,
            refit_rng=copy.deepcopy(runtime.refit_rng.bit_generator.state),
            driver_rng=driver_rng(),worker_rng=worker_rng,reason=reason)
        save(state_path,payload)
        publish(manifest,dict(sha256=_sha(state_path),outer=outer,epoch=phase['epoch'],
            batch=phase['batch'],head_adam_steps=runtime.head._adam_steps(),stage=reason,checked_unix=time.time()))
    persist('COLLECTING_CURRENT_STUDENT_LABELS')
    # Small labels and real rollout contract proofs persist; feature tensors do not.
    for start in range(len(labels),5120,512):
        chunk = questions[start:start+512]
        parents = DataProto.from_dict(tensors={'qprobe_source_index':torch.tensor(indices[start:start+512])},
            non_tensors=dict(candidate_original_question=np.array([q.question for q in chunk],dtype=object),
                             candidate_stable_id=np.array([q.qid for q in chunk],dtype=object)))
        parent = trainer.mixed_stream.materialize(parents,range(512))
        identity = prepare_identity(parent,outer,repeats=5,candidate_offset=start,probe_refit=True)
        repeated = parent.repeat(repeat_times=5,interleave=True)
        gen = repeated.pop(batch_keys=['input_ids','attention_mask','position_ids'])
        attach_generation_identity(gen,identity,do_sample=True)
        manager.timing_raw = {}
        output = manager.run_llm_loop(gen_batch=gen,
            initial_input_ids=gen.batch['input_ids'][:,-manager.config.max_start_length:].clone().long())
        _complete_generation(output,identity)
        vectors = strict_vectors(parent,output,trainer.tokenizer)
        if len(vectors) != 512 or any(len(v)!=5 or any(type(x) is not int or x not in (0,1) for x in v) for v in vectors):
            raise ValueError('PERIODIC_INCOMPLETE_REAL_G5_STRICT_EM')
        run_manifest = json.loads((runtime.base/'logs/run-manifest.json').read_text())
        contracts = _contracts(trainer,output,run_manifest)
        labels.extend(dict(qid=q.qid,question_hash=q.question_hash,strict_em=v,correct_count=sum(v))
                      for q,v in zip(chunk,vectors))
        chunk_dir = directory/('chunk-'+str(start//512))
        worker_rng = trainer.actor_rollout_wg.capture_mixed_state(str(chunk_dir/'worker-rng'))
        _write_json(chunk_dir/'real-rollout-audit.json',dict(outer=outer,questions=512,
            trajectories=2560,identity=identity,contracts=contracts,labels=labels[start:start+512],
            policy_tokens=int(output.batch['sampling_mask'].sum()),timings=manager.timing_raw))
        del output,repeated,gen,parent
        persist('CURRENT_STUDENT_LABEL_CHUNK_COMPLETE')
    provider = encode(trainer,questions,runtime.cache)
    actual_hashes = provider.hashes()
    if feature_hashes is not None and actual_hashes != feature_hashes:
        raise ValueError('PERIODIC_RECOMPUTED_FEATURES_DIFFER_FROM_SAME_STUDENT_SNAPSHOT')
    feature_hashes = actual_hashes
    frozen_after = trainer.actor_rollout_wg.capture_periodic_probe_boundary(str(directory/('frozen-after-'+str(time.time_ns()))))
    for before,after in zip(sorted(frozen_before,key=lambda r:r['rank']),sorted(frozen_after,key=lambda r:r['rank'])):
        for key in ('rank','parameter_sha256','local_parameter_count','actual_optimizer_writes',
                    'completed_outer','scheduler_sha256','adam_steps'):
            if before[key] != after[key]:
                raise ValueError('PERIODIC_STUDENT_CHANGED_DURING_LABEL_FEATURE_COLLECTION_'+key)
    runtime.execution_phase = 'PROBE_FITTING'
    persist('PROBE_FITTING')
    def callback(kind,state):
        # The ledger catches unsaved writes after uncatchable OOM/SIGKILL.
        possible = runtime.head._adam_steps() + (1 if kind=='before_step' else 0)
        publish(ledger,dict(possible_adam_steps=possible,outer=outer,stage=kind,checked_unix=time.time()))
        if kind in ('epoch','complete','failure','permutation'):
            persist('PROBE_'+kind.upper())
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminated(signum,frame):
        raise KeyboardInterrupt('Persisting continuous probe after signal '+str(signum))
    signal.signal(signal.SIGTERM,terminated)
    try:
        result = fit_phase(runtime.head._engine,provider,
            [runtime.cache.index[q.qid]['length'] for q in questions],
            [row['correct_count'] for row in labels],[q.qid for q in questions],phase,callback)
    finally:
        signal.signal(signal.SIGTERM,old_handler)
        del provider
    if result['head_optimizer_updates'] != 400 or phase['epoch'] != 20:
        raise ValueError('PERIODIC_REFIT_DID_NOT_FINISH_TWENTY_FULL_EPOCHS')
    runtime.head._version_cache = None
    runtime.head.phases.append(dict(outer=outer,**result))
    runtime.execution_phase = 'STUDENT_READY'
    post_path = directory/'post-probe-state.pt'
    runtime.save(post_path,actor_id,compact_stream_state(trainer.mixed_stream.state()),trainer.reward_fn.expected_probe_version)
    restored_head = PeriodicQwenProbe.load(post_path.with_name(post_path.name+'.head.pt'),
        expected_actor_snapshot_id=actor_id,expected_feature_spec_sha=runtime.feature_spec_sha)
    if restored_head.version != runtime.version or restored_head._adam_steps() != runtime.head._adam_steps():
        raise ValueError('PERIODIC_POST_REFIT_HEAD_READBACK_FAILED')
    runtime.post_refit_state_path = str(post_path)
    accepted = dict(stage='PERIODIC_PROBE_REFIT20_ACCEPTED',outer=outer,
        pre_checkpoint_receipt=str(pre_receipt),pre_checkpoint_sha256=_sha(pre_receipt),
        post_state_sha256=_sha(post_path),student_updated=False,student_parameters_copied=False,
        optimizer_reset=False,dynamic_feature_tensors_persisted=False,trajectories=25600,
        **result,checked_unix=time.time())
    _write_json(complete,accepted)
    return accepted

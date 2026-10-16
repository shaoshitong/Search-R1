"""Actual candidate-prefix, frozen acquisition, selected-continuation transaction."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from search_g1.prefix20_contract import top_questions
from search_g1.prefix20_generation import PrefixCollector,PrefixState
from search_g1.qprobe_runtime import _complete_generation,_contracts,_write_json
from search_g1.training_state import atomic_save,compact_stream_state,driver_rng


def _frozen_workers(before,after,*,rng_equal=False):
    from search_g1.full_state_restore import equal_tree
    if [r['rank'] for r in before]!=list(range(4)) or [r['rank'] for r in after]!=list(range(4)):
        raise ValueError('PREFIX20_ACTUAL_FOUR_RANK_BOUNDARIES_MISSING')
    for first,last in zip(before,after):
        a=torch.load(first['path'],map_location='cpu',weights_only=False)
        b=torch.load(last['path'],map_location='cpu',weights_only=False)
        for key in ('run_id','rank','world_size','actual_optimizer_writes','completed_outer',
                    'optimizer_calls','optimizer_rollbacks','actual_adam_steps','actual_scheduler_last_epoch'):
            equal_tree(a[key],b[key])
        if a['world_size']!=4:raise ValueError('PREFIX20_ACTUAL_WORKER_WORLD_CHANGED')
        if rng_equal:equal_tree(a['rng'],b['rng'])
    return dict(actual_optimizer_and_scheduler_unchanged=True,actor_rng_unchanged=rng_equal)


def selected_identity(identity,ordinals):
    """Subset existing parent identities; never mint another policy or prefix ID."""
    if len(ordinals)!=512 or len(set(ordinals))!=512 or any(type(i) is not int or not 0<=i<5120 for i in ordinals):
        raise ValueError('PREFIX20_INVALID_SELECTED_CANDIDATE_ORDINALS')
    if len(identity['parent_group_ids'])!=5120 or len(identity['trajectory_ids'])!=25600:
        raise ValueError('PREFIX20_FULL_CANDIDATE_IDENTITY_REQUIRED')
    result=copy.deepcopy(identity)
    result['parent_group_ids']=[identity['parent_group_ids'][i] for i in ordinals]
    result['trajectory_ids']=[identity['trajectory_ids'][5*i+j] for i in ordinals for j in range(5)]
    return result


def collect(trainer,manager,stream,attempt_identity):
    from reproduction.common8.behavior_runtime import prepare_identity
    from reproduction.common8.context_budget import protocol
    from search_g1.mixed_runtime import subset,strict_vectors
    from search_g1.full_state_restore import equal_tree
    runtime=trainer.qprobe_runtime
    if runtime.pending is not None or attempt_identity is None:
        raise ValueError('PREFIX20_REQUIRES_IDLE_HEAD_AND_EXPLICIT_ACTOR_ATTEMPT')
    outer=int(trainer.global_steps);tag=attempt_identity['attempt_id']
    if len(tag)!=32 or any(c not in '0123456789abcdef' for c in tag):
        raise ValueError('PREFIX20_UNSAFE_ATTEMPT_ID')
    directory=runtime.output_dir/'attempts'/tag;directory.mkdir(parents=True,exist_ok=False)
    run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
    started=time.monotonic();collector=None;identity=None
    atomic_save(directory/'before-selection.pt',dict(stream=compact_stream_state(stream.state()),
        driver_rng=driver_rng(),selection_rng=copy.deepcopy(runtime.selection_rng.bit_generator.state),head_version=runtime.version))
    try:
        candidates,questions=stream.collect_pool(5120)
        parents=stream.materialize(candidates,list(range(5120)))
        identity=prepare_identity(parents,outer,repeats=5,candidate_prefix=True,
                                  **{k:attempt_identity[k] for k in ('attempt_id','attempt_index')})
        limit=min(manager.config.max_start_length,protocol(manager.config)['total_tokens']-500)
        ids=parents.batch['input_ids'][:,-limit:].detach().cpu().long()
        mask=parents.batch['attention_mask'][:,-limit:].detach().cpu().long()
        if ids.shape!=mask.shape or not bool(((mask==0)|(mask==1)).all()):
            raise ValueError('PREFIX20_INITIAL_PROMPT_MASK_CHANGED')
        prompts=[[int(t) for t,m in zip(row,valid) if m] for row,valid in zip(ids.tolist(),mask.tolist())]
        if any(not row or manager.tokenizer.pad_token_id in row for row in prompts):
            raise ValueError('PREFIX20_EMPTY_OR_PAD_IN_ACTUAL_PROMPT')
        states=[PrefixState(i,identity['trajectory_ids'][i],questions[i//5].qid,i%5,prompts[i//5])
                for i in range(25600)]
        collector=PrefixCollector(manager,identity,states)
        before=trainer.actor_rollout_wg.capture_mixed_state(str(directory/'before-prefix-worker-rng'))
        collector.phase(states,identity,prefix_only=True)
        prefixes=copy.deepcopy(collector.prefix_rows())
        after_prefix=trainer.actor_rollout_wg.capture_mixed_state(str(directory/'after-prefix-worker-rng'))
        prefix_freeze=_frozen_workers(before,after_prefix)
        probabilities=runtime.score(prefixes)
        after_prediction=trainer.actor_rollout_wg.capture_mixed_state(str(directory/'after-prediction-worker-rng'))
        prediction_freeze=_frozen_workers(after_prefix,after_prediction,rng_equal=True)
        prediction_version=runtime.version
        wire=[dict(qid=q.qid,trajectory_ids=identity['trajectory_ids'][i*5:(i+1)*5],
              em_probabilities=[probabilities[t] for t in identity['trajectory_ids'][i*5:(i+1)*5]])
              for i,q in enumerate(questions)]
        selected=top_questions(wire);ordinals=[r['candidate_ordinal'] for r in selected]
        chosen=set(ordinals);selected_ids=selected_identity(identity,ordinals)
        selection=dict(stage='prefix20_selection_frozen_before_continuation',outer=outer,attempt_id=tag,
            policy_snapshot_id=identity['policy_snapshot_id'],head_version=prediction_version,
            candidate_questions=5120,selected_questions=512,exploration_questions=0,
            questions=[dict(ordinal=i,qid=q.qid,content_sha256=q.content_hash,
                pool_index=int(parents.batch['pool_index'][i]),
                canonical_source_index=int(parents.batch['canonical_source_index'][i]),
                pool_cycle=int(parents.batch['pool_cycle'][i]),
                trajectory_ids=wire[i]['trajectory_ids'],em_probabilities=wire[i]['em_probabilities'],
                selected=i in chosen) for i,q in enumerate(questions)],selected=selected,
            prefix_freeze=prefix_freeze,prediction_freeze=prediction_freeze)
        _write_json(directory/'selection.json',selection)
        atomic_save(directory/'candidate-prefixes.pt',dict(identity=identity,prefix_rows=prefixes,
            states=[asdict(s) for s in states],worker_files=after_prediction,selection=selection,
            stream=compact_stream_state(stream.state()),driver_rng=driver_rng()))
        selected_states=[states[5*i+j] for i in ordinals for j in range(5)]
        selected_prefixes=[prefixes[5*i+j] for i in ordinals for j in range(5)]
        collector.phase(selected_states,selected_ids,prefix_only=False)
        equal_tree(selected_prefixes,collector.prefix_rows(selected_states))
        output=collector.output(selected_states,selected_ids);_complete_generation(output,selected_ids)
        after=trainer.actor_rollout_wg.capture_mixed_state(str(directory/'after-continuation-worker-rng'))
        continuation_freeze=_frozen_workers(after_prediction,after)
        parent=subset(parents,ordinals)
        parent.non_tensor_batch['selection_source']=np.asarray(['predicted_mixed']*512,dtype=object)
        vectors=strict_vectors(parent,output,trainer.tokenizer)
        if len(vectors)!=512 or any(len(v)!=5 or any(type(y) is not int or y not in (0,1) for y in v) for v in vectors):
            raise ValueError('PREFIX20_ACTUAL_SELECTED_LABELS_INCOMPLETE')
        rows=[dict(row,strict_em=vectors[i//5][i%5]) for i,row in enumerate(selected_prefixes)]
        runtime.begin_outer(outer,tag,rows,{r['trajectory_id']:probabilities[r['trajectory_id']] for r in rows},
                            identity['policy_snapshot_id'])
        contracts=_contracts(trainer,output,run)
        categories=['all_wrong' if sum(v)==0 else 'all_correct' if sum(v)==5 else 'mixed' for v in vectors]
        audit=dict(selection_mode='prefix20_probe',attempted_candidates=5120,selected_groups=512,
            selected_mixed=categories.count('mixed'),selected_all_correct=categories.count('all_correct'),
            selected_all_wrong=categories.count('all_wrong'),selected_all_correct_fallback=0,
            selected_random_exploration=0,selected_predicted_mixed=512,
            raw_generated_trajectories=25600,completed_selected_trajectories=2560,
            raw_generated_policy_tokens=sum(len(r['raw_generated_token_ids']) for r in collector.captured_records),
            raw_actual_retrieval_calls=sum(s.actual_retrieval for s in states),
            candidate_fill_seconds=time.monotonic()-started,step=outer,attempt_id=tag,
            policy_snapshot_id=identity['policy_snapshot_id'],question_probe_version=prediction_version,
            selected_mixed_fraction=categories.count('mixed')/512,
            source_consumed=stream.consumed,no_post_rollout_refill=True,
            continuation_freeze=continuation_freeze,phases=collector.phase_events,**contracts)
        _write_json(directory/'rollout-audit.json',audit)
        atomic_save(directory/'real-rollout.pt',dict(parents=parent,output=output,identity=selected_ids,
            selected_ordinals=ordinals,strict_correctness=vectors,probe_transaction=runtime.pending,
            worker_files=after,stream=compact_stream_state(stream.state()),driver_rng=driver_rng()))
        repeated=parent.repeat(repeat_times=5,interleave=True)
        repeated.pop(batch_keys=['input_ids','attention_mask','position_ids'])
        print(json.dumps(dict(stage='prefix20_collection_complete',outer=outer,attempt_id=tag,
            candidate_prefixes=25600,completed_selected=2560,selected_mixed_fraction=audit['selected_mixed_fraction'])),flush=True)
        return repeated,output,selected_ids,audit
    except BaseException as error:
        atomic_save(directory/'failure.pt',dict(error_type=type(error).__name__,identity=identity,
            actual_sampling_records=[] if collector is None else collector.captured_records,
            states=[] if collector is None else [asdict(s) for s in collector.states],
            stream=compact_stream_state(stream.state()),driver_rng=driver_rng(),actor_or_probe_update_started=False))
        raise

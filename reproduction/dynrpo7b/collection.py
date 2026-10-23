"""One actor snapshot: prefix -> probe -> same-G5 continuation -> actual EM."""
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .recipe import G, SELECTED, probe_update, prefix_tokens
from .generation import PrefixCollector, PrefixState
from .selection import Candidate, select_top_prefix, select_actual_mixed, group_counts
from .mixed_backstop import BackstopLedger


def sub_identity(identity, ordinals):
    result = copy.deepcopy(identity)
    result['parent_group_ids'] = [identity['parent_group_ids'][i] for i in ordinals]
    result['trajectory_ids'] = [identity['trajectory_ids'][G*i+j] for i in ordinals for j in range(G)]
    return result


def groups_from_states(states, ordinals, *, labels=None, keep=160):
    groups = []
    for i in ordinals:
        rows = states[G*i:G*(i+1)]
        group = dict(qid=rows[0].qid, prompt_token_ids=rows[0].prompt,
                     policy_token_ids=[([t for t,m in zip(s.response,s.info_mask) if m]+s.current_raw)[:keep]
                                       for s in rows])
        if labels is not None:
            group['strict_em'] = list(labels[i])
        groups.append(group)
    return groups


def collect(trainer, manager, stream, attempt_identity):
    from reproduction.common8.behavior_runtime import prepare_identity
    from reproduction.common8.context_budget import protocol
    from search_g1.qprobe_runtime import _collect_pool, _contracts, _complete_generation, _write_json
    from search_g1.mixed_runtime import subset, strict_vectors
    from search_g1.training_state import atomic_save, compact_stream_state, driver_rng

    runtime = trainer.qprobe_runtime
    outer = int(trainer.global_steps)
    periodic = probe_update(outer)
    count = 2560 if periodic else 5120
    tag = attempt_identity['attempt_id']
    directory = runtime.output_dir/'attempts'/tag
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    candidates, questions = _collect_pool(stream, count)
    parents = stream.materialize(candidates, list(range(count)))
    identity = prepare_identity(parents, outer, repeats=G, candidate_prefix=True,
        **{k:attempt_identity[k] for k in ('attempt_id','attempt_index')})
    limit = min(manager.config.max_start_length, protocol(manager.config)['total_tokens']-500)
    ids = parents.batch['input_ids'][:,-limit:].detach().cpu().long()
    mask = parents.batch['attention_mask'][:,-limit:].detach().cpu().long()
    prompts = [[int(t) for t,m in zip(row,valid) if m] for row,valid in zip(ids.tolist(),mask.tolist())]
    states = [PrefixState(i, identity['trajectory_ids'][i], questions[i//G].qid, i%G, prompts[i//G])
              for i in range(count*G)]
    collector = PrefixCollector(manager, identity, states, prefix_tokens=prefix_tokens(outer))
    labels = {}
    completed = {}

    def finish(ordinals):
        selected_states = [states[G*i+j] for i in ordinals for j in range(G)]
        part_identity = sub_identity(identity, ordinals)
        collector.phase(selected_states, part_identity, prefix_only=False)
        output = collector.output(selected_states, part_identity)
        vectors = strict_vectors(subset(parents,ordinals), output, trainer.tokenizer)
        rows = []
        for i,vector in zip(ordinals,vectors):
            labels[i] = vector
            item = replace(frozen[i], strict_em=tuple(vector))
            completed[i] = item
            rows.append(item)
        # Only remote state is persisted. No tensors travel through the control Mac.
        ref = atomic_save(directory/('continuation-%05d.pt'%len(completed)),
            dict(ordinals=ordinals, states=[asdict(s) for s in selected_states], strict_em=vectors))
        return rows, {r.question_id:ref for r in rows}

    try:
        collector.phase(states, identity, prefix_only=True)
        scores = runtime.score_groups(groups_from_states(states,range(count),keep=prefix_tokens(outer)),
                                      prefix=prefix_tokens(outer))
        frozen = [Candidate(questions[i].qid,i,float(scores[i]),payload=dict(ordinal=i)) for i in range(count)]
        _write_json(directory/'scores.json', dict(outer=outer,actor_version=identity['policy_snapshot_id'],
            probe_version=runtime.version,prefix_tokens=prefix_tokens(outer),
            scores=[dict(qid=c.question_id,ordinal=c.ordinal,score=c.score) for c in frozen]))
        atomic_save(directory/'prefixes.pt',dict(identity=identity,states=[asdict(s) for s in states],
            stream=compact_stream_state(stream.state()),rng=driver_rng()))
        first = frozen if periodic else select_top_prefix(frozen,1024)
        full, refs = finish([r.ordinal for r in first])
        initial = select_actual_mixed(full,SELECTED)
        outcome_pool = [completed.get(c.ordinal,c) for c in frozen]
        ledger = BackstopLedger(directory/'backstop',outcome_pool,initial,outer=outer,
            actor_version=identity['policy_snapshot_id'],probe_version=runtime.version)
        while True:
            batch = ledger.next_batch()
            if not batch:
                break
            full, refs = finish([r.ordinal for r in batch])
            ledger.commit_batch(full,artifact_refs=refs)
        chosen = ledger.selected()
        ordinals = [r.ordinal for r in chosen]
        selected_identity = sub_identity(identity,ordinals)
        chosen_states = [states[G*i+j] for i in ordinals for j in range(G)]
        output = collector.output(chosen_states,selected_identity)
        _complete_generation(output,selected_identity)
        parent = subset(parents,ordinals)
        parent.non_tensor_batch['selection_source'] = np.asarray(['predicted_mixed']*SELECTED,dtype=object)
        periodic_groups = groups_from_states(states,range(count),labels=labels) if periodic else None
        runtime.begin_outer(outer,tag,periodic_groups,identity['policy_snapshot_id'],directory)
        run = json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
        counts = group_counts(chosen)
        top_counts = group_counts([completed[c.ordinal] for c in first])
        audit = dict(selection_mode='dynrpo7b',attempted_candidates=count,selected_groups=SELECTED,
            selected_mixed=counts['mixed'],selected_all_correct=counts['all_correct'],
            selected_all_wrong=counts['all_wrong'],selected_all_correct_fallback=counts['all_correct'],
            raw_generated_trajectories=count*G,completed_selected_trajectories=SELECTED*G,
            raw_generated_policy_tokens=sum(len(r['raw_generated_token_ids']) for r in collector.captured_records),
            raw_actual_retrieval_calls=sum(s.actual_retrieval for s in states),
            candidate_fill_seconds=time.monotonic()-started,pool_retry=0,step=outer,attempt_id=tag,
            probe_top_group_counts=top_counts,backstop=ledger.summary(),
            selected_mixed_fraction=counts['mixed']/SELECTED,question_probe_version=runtime.version,
            policy_snapshot_id=identity['policy_snapshot_id'],prefix_tokens=prefix_tokens(outer),
            completed_candidate_groups=len(completed),**_contracts(trainer,output,run))
        _write_json(directory/'rollout-audit.json',audit)
        runtime.last_rollout_ref = atomic_save(directory/'real-rollout.pt',dict(parents=parent,output=output,
            identity=selected_identity,selected_ordinals=ordinals,strict_em=[labels[i] for i in ordinals],
            probe_fit_groups=periodic_groups,stream=compact_stream_state(stream.state()),rng=driver_rng()))
        runtime.rollout_checkpoint_reference=runtime.last_rollout_ref
        repeated = parent.repeat(repeat_times=G,interleave=True)
        repeated.pop(batch_keys=['input_ids','attention_mask','position_ids'])
        print('DYNRPO_COLLECTION='+json.dumps(audit),flush=True)
        return repeated,output,selected_identity,audit
    except BaseException as error:
        atomic_save(directory/'failure.pt',dict(error=repr(error),identity=identity,
            states=[asdict(s) for s in states],strict_em=labels,
            worker_rng=trainer.actor_rollout_wg.capture_mixed_state(str(directory/'failure-worker-rng')),
            stream=compact_stream_state(stream.state()),rng=driver_rng(),actor_update_started=False))
        raise

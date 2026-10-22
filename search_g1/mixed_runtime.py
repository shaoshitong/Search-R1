"""Frozen-policy candidate collection before any reward, probe or Adam update."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from verl import DataProto
from search_g1.mixed_selection import MixedFirstSelector, InsufficientEligibleGroups
from search_g1.signals import decode_segments, contexts
from verl.utils.reward_score import qa_em


ROW_META = ('turns_stats', 'active_mask', 'valid_action_stats',
            'valid_search_stats', 'actual_retrieval_stats',
            'clipped_observation_stats', 'early_final_turn_stats')

def subset(batch, indices):
    meta = batch.meta_info.copy()
    for key in ROW_META:
        if key in meta:
            if len(meta[key]) != len(batch):
                raise ValueError("Per-trajectory metadata length differs: " + key)
            meta[key] = [meta[key][i] for i in indices]
    return DataProto(batch=batch.batch[indices],
                     non_tensor_batch={k:v[indices] for k,v in batch.non_tensor_batch.items()},
                     meta_info=meta)


class CandidateStream:
    def __init__(self, loader, restored_state=None):
        self.loader, self.pending, self.offset = loader, None, 0
        if restored_state is None:
            self.order, self.cursor, self.epoch = list(iter(loader.sampler)), 0, 0
            self.consumed = 0
        else:
            # A restored order must not draw a new sampler permutation/RNG.
            from search_g1.training_state import compact_stream_state
            state = compact_stream_state(restored_state)
            self.order = list(state['candidate_order'])
            assert sorted(self.order) == list(range(len(loader.dataset)))
            self.cursor, self.epoch = state['candidate_cursor'], state['epoch']
            assert 0 <= self.cursor <= len(self.order) and self.epoch >= 0
            self.consumed = state['consumed_source_rows']
            self.pending, self.offset = state['pending_source'], state['pending_offset']

    def state(self):
        return dict(candidate_order=self.order, candidate_cursor=self.cursor,
                    epoch=self.epoch, consumed_source_rows=self.consumed,
                    pending_source=self.pending, pending_offset=self.offset)

    def one(self):
        if self.pending is None or self.offset == len(self.pending):
            if self.cursor == len(self.order):
                self.order = list(iter(self.loader.sampler))
                self.cursor, self.epoch = 0, self.epoch+1
            indices = self.order[self.cursor:self.cursor+512]
            self.cursor += len(indices)
            records=[]
            for i in indices:
                record=self.loader.dataset[i]
                original=self.loader.dataset.dataframe.iloc[i]
                chat=original[self.loader.dataset.prompt_key]
                question='\n'.join(message['content'] for message in chat)
                extra=original.get('extra_info', {})
                upstream_id=extra.get('index') if isinstance(extra,dict) else None
                if upstream_id is None:
                    from search_g1.mixed_selection import question_identity
                    upstream_id=question_identity(question)
                record['candidate_original_question']=question
                record['candidate_stable_id']=json.dumps([str(original['data_source']),str(upstream_id)])
                records.append(record)
            raw = self.loader.collate_fn(records)
            self.pending, self.offset = DataProto.from_single_dict(raw), 0
        row = subset(self.pending, [self.offset]); self.offset += 1; self.consumed += 1
        return row

    def parents(self, selector, tokenizer):
        rows, ordinals = [], []
        # Each source record is visited at most once in this fill. Exhaustion is
        # explicit rather than looping indefinitely over a duplicate-only epoch.
        examined = 0
        while len(rows) < 512:
            if examined >= len(self.loader.dataset):
                raise RuntimeError('UNIQUE_CANDIDATE_SOURCE_EXHAUSTED')
            row = self.one(); examined += 1
            question = row.non_tensor_batch['candidate_original_question'][0]
            key = row.non_tensor_batch['candidate_stable_id'][0]
            ordinal = selector.reserve(key, question)
            if ordinal is not None:
                rows.append(row); ordinals.append(ordinal)
        return DataProto.concat(rows), ordinals


def strict_vectors(parent, output, tokenizer):
    values = []
    for i in range(len(output)):
        item = output[i]; width = item.batch['prompts'].shape[-1]
        count = int(item.batch['attention_mask'][width:].sum())
        ids = item.batch['responses'][:count]
        mask = item.batch['info_mask'][width:width+count]
        text, ranges = decode_segments(tokenizer, ids, mask)
        ctx = contexts('', text, ranges)
        gt = parent.non_tensor_batch['reward_model'][i//5]['ground_truth']
        values.append(int(ctx is not None and qa_em.em_check(ctx['answer'], gt['target'])))
    return [values[i:i+5] for i in range(0,len(values),5)]


def padded_concat(outputs, pad_id):
    widths = {int(x.batch['prompts'].shape[-1]) for x in outputs}
    if len(widths) != 1:
        raise ValueError('Candidate prompt widths differ')
    max_response = max(x.batch['responses'].shape[-1] for x in outputs)
    token_keys = {'responses','responses_with_info_mask','input_ids','prompts'}
    for output in outputs:
        extra = max_response-output.batch['responses'].shape[-1]
        if not extra:
            continue
        for key in output.batch.keys():
            if key == 'prompts' or output.batch[key].ndim != 2:
                continue
            value = pad_id if key in token_keys else 0
            output.batch[key] = torch.nn.functional.pad(output.batch[key],(0,extra),value=value)
    merged = DataProto.concat(outputs)
    merged.meta_info = outputs[0].meta_info.copy()
    for key in ROW_META:
        if any(key in x.meta_info for x in outputs):
            if not all(key in x.meta_info and len(x.meta_info[key]) == len(x) for x in outputs):
                raise ValueError("Incomplete per-trajectory metadata: " + key)
            merged.meta_info[key] = [v for x in outputs for v in x.meta_info[key]]
    return merged


def collect(trainer, manager, stream, attempt_identity):
    """Discard insufficient fixed pools before any reward or optimizer update."""
    retries = 0
    discarded = []
    while True:
        try:
            return _collect_once(trainer, manager, stream, attempt_identity, retries, discarded)
        except InsufficientEligibleGroups as exc:
            if not trainer.config.data.get('selection_fixed_candidate_budget', False):
                raise
            discarded.append(exc.pool_summary)
            retries += 1
            print(json.dumps(dict(stage='mixed_insufficient_pool_consumed',
                step=trainer.global_steps,pool_retry=retries,
                **exc.pool_summary)),flush=True)


def _collect_once(trainer, manager, stream, attempt_identity, pool_retry=0, discarded=()):
    from reproduction.common8.behavior_runtime import prepare_identity, attach_generation_identity
    started = time.monotonic()
    raw_policy_tokens = raw_retrieval_calls = 0
    generation_timings = {}
    generation_phases = []
    config = trainer.config.data
    limit = int(config.get('selection_candidate_questions',2048))
    target = int(config.get('selection_selected_questions',512))
    fixed = bool(config.get('selection_fixed_candidate_budget',False))
    if target != 512 or limit % 512:
        raise ValueError('Native G5 mixed collection requires selected512 and full512 candidate chunks')
    selector = MixedFirstSelector(target=target,limit=limit,fixed_pool=fixed)
    candidates = {}; identities = []
    snapshot = None
    base = Path(os.environ['SEARCH_G1_BASE'])
    tag = attempt_identity['attempt_id'] if attempt_identity is not None else str(trainer.global_steps)
    pool = base/'artifacts/mixed-candidate-pools'/(tag+'-pool-'+str(pool_retry) if fixed else tag)
    pool.mkdir(parents=True, exist_ok=False)
    raw_pool = stream.collect_pool(limit)[0] if fixed else None
    while len(selector.reserved) < limit and (fixed or selector.mixed_count < target):
        if fixed:
            begin = len(selector.reserved)
            ordinals = list(range(begin,begin+512))
            parent = stream.materialize(raw_pool,ordinals)
            for local, ordinal in enumerate(ordinals):
                actual = selector.reserve(parent.non_tensor_batch['candidate_stable_id'][local],
                                          parent.non_tensor_batch['candidate_original_question'][local])
                if actual != ordinal:
                    raise ValueError('Fixed eligible candidate pool changed order or uniqueness')
        else:
            parent, ordinals = stream.parents(selector, trainer.tokenizer)
        identity = prepare_identity(parent, trainer.global_steps, repeats=5,
            candidate_offset=pool_retry*limit+ordinals[0], **({k:attempt_identity[k] for k in ('attempt_id','attempt_index')}
                                           if attempt_identity is not None else {}))
        if snapshot is None:
            snapshot = identity['policy_snapshot_id']
        identity['policy_snapshot_id'] = snapshot
        repeated = parent.repeat(repeat_times=5,interleave=True)
        gen = repeated.pop(batch_keys=['input_ids','attention_mask','position_ids'])
        attach_generation_identity(gen, identity, do_sample=True)
        actor_rollout = getattr(trainer.config, 'actor_rollout_ref', None)
        reuse_weights = actor_rollout.rollout.get('reuse_snapshot_weights', False) if actor_rollout is not None else False
        if type(reuse_weights) is not bool:
            raise ValueError('Snapshot weight reuse must be an explicit bool')
        if reuse_weights:
            # Only a complete fixed candidate pool has a known final chunk.
            # Keep all five independent phase/RNG/KV-cache boundaries; retain
            # only the immutable vLLM-owned weights between those boundaries.
            if not fixed:
                raise ValueError('Snapshot weight reuse requires a fixed candidate pool')
            gen.meta_info.update(snapshot_weight_reuse=True,
                retain_snapshot_weights=len(selector.reserved) < limit)
        worker_proofs = trainer.actor_rollout_wg.capture_mixed_state(
            str(pool/(str(ordinals[0])+'-worker-rng')))
        torch.save({'stream':stream.state(),'reserved':selector.reserved,
                    'identity':identity,'torch_rng':torch.get_rng_state(),
                    'numpy_rng':np.random.get_state(),'worker_sampling_state_files':worker_proofs},
                   pool/(str(ordinals[0])+'-before-generation.pt'))
        manager.timing_raw = {}
        output = manager.run_llm_loop(gen_batch=gen,
            initial_input_ids=gen.batch['input_ids'][:,-manager.config.max_start_length:].clone().long())
        if 'async_rollout' in output.meta_info:
            generation_phases.append(output.meta_info['async_rollout'])
        # Save the actual post-generation sampling boundary as well as the
        # pre-generation one. A later backward failure must not require replay
        # of an already completed candidate group to reconstruct its RNG.
        from search_g1.training_state import atomic_save, driver_rng, compact_stream_state
        after_workers = trainer.actor_rollout_wg.capture_mixed_state(
            str(pool/(str(ordinals[0])+'-after-generation-worker-rng')))
        after_stream = compact_stream_state(stream.state())
        atomic_save(pool/(str(ordinals[0])+'-after-generation.pt'),
                    dict(stream=after_stream,worker_files=after_workers,
                         driver_rng=driver_rng(),identity=identity))
        raw_policy_tokens += int(output.batch["sampling_mask"].sum().item())
        raw_retrieval_calls += sum(output.meta_info["actual_retrieval_stats"])
        for key, value in manager.timing_raw.items():
            generation_timings[key] = generation_timings.get(key, 0.0) + float(value)
        vectors = strict_vectors(parent, output, trainer.tokenizer)
        # Complete generated tensors remain remote for interruption diagnosis.
        torch.save({'parents':parent,'output':output,'identity':identity,
                    'ordinals':ordinals,'strict_correctness':vectors,
                    'source_consumed':stream.consumed,'pending_source':stream.pending,
                    'pending_offset':stream.offset,'stream':stream.state(),'torch_rng':torch.get_rng_state(),
                    'numpy_rng':np.random.get_state()}, pool/(str(ordinals[0])+'.pt'))
        for local, ordinal in enumerate(ordinals):
            selector.complete(ordinal, identity['trajectory_ids'][local*5:(local+1)*5], vectors[local])
            candidates[ordinal] = (subset(parent,[local]),subset(output,list(range(local*5,(local+1)*5))))
        identities.append(identity)
        with (pool/'progress.json').open('w') as f:
            json.dump({'step':trainer.global_steps,'reserved':selector.reserved,
                       'complete_ordinals':sorted(selector.results),
                       'policy_snapshot_id':snapshot,'source_consumed':stream.consumed},f)
            f.flush();os.fsync(f.fileno())
        print(json.dumps({'stage':'mixed_candidate_fill','step':trainer.global_steps,
                          'candidates':len(selector.reserved),'mixed':selector.mixed_count}),flush=True)
    try:
        audit = selector.audit()
    except Exception as exc:
        summary=dict(status=str(exc),reserved=len(selector.reserved),mixed=selector.mixed_count,
            optimizer_started=False,pool_path=str(pool),
            raw_generated_trajectories=len(selector.reserved)*5,
            raw_generated_policy_tokens=raw_policy_tokens,raw_actual_retrieval_calls=raw_retrieval_calls,
            candidate_fill_seconds=time.monotonic()-started,candidate_generation_timings=generation_timings,
            candidate_generation_phases=generation_phases)
        with (pool/'selection-failure.json').open('x') as f:
            json.dump(summary,f);f.flush();os.fsync(f.fileno())
        if isinstance(exc,InsufficientEligibleGroups):exc.pool_summary=summary
        raise
    members = selector.select()
    parents = DataProto.concat([candidates[x.ordinal][0] for x in members])
    parents.non_tensor_batch["selection_source"] = np.array([
        "mixed" if x.category == "mixed" else "all_correct_fallback" for x in members], dtype=object)
    output = padded_concat([candidates[x.ordinal][1] for x in members], trainer.tokenizer.pad_token_id)
    identity = identities[0].copy()
    identity['parent_group_ids'] = [identities[x.ordinal//512]['parent_group_ids'][x.ordinal%512] for x in members]
    identity['trajectory_ids'] = [tid for x in members for tid in x.trajectory_ids]
    output.meta_info['behavior_identity'] = identity
    output.meta_info['policy_snapshot_id'] = snapshot
    base = Path(os.environ['SEARCH_G1_BASE']); directory = base/'artifacts/mixed-selection';directory.mkdir(parents=True,exist_ok=True)
    tag = attempt_identity['attempt_id'] if attempt_identity is not None else str(trainer.global_steps)
    run = json.loads((base/'logs/run-manifest.json').read_text())
    audit.update(step=trainer.global_steps,policy_snapshot_id=snapshot,source_consumed=stream.consumed,
                 source_commit=run['source_commit'],model_revision=run['model_revision'],
                 data_manifest_sha256=run['budget']['common_manifest_sha256'],
                 frozen_probe_version=getattr(trainer.reward_fn,'expected_probe_version',None),
                 raw_generated_trajectories=len(selector.reserved)*5,
                 raw_generated_policy_tokens=raw_policy_tokens,
                 raw_actual_retrieval_calls=raw_retrieval_calls,
                 candidate_fill_seconds=time.monotonic()-started,
                 candidate_generation_timings=generation_timings)
    if fixed:
        audit.update(fixed_candidate_budget=True,pool_retry=pool_retry,pool_path=str(pool),
                     discarded_insufficient_pools=list(discarded),
                     candidate_generation_phases=[phase for row in discarded
                         for phase in row['candidate_generation_phases']]+generation_phases,
                     eligible_pool_binding=dict(stream.binding))
        for key in ('raw_generated_trajectories','raw_generated_policy_tokens',
                    'raw_actual_retrieval_calls','candidate_fill_seconds'):
            audit[key] += sum(row[key] for row in discarded)
        for row in discarded:
            for key,value in row['candidate_generation_timings'].items():
                audit['candidate_generation_timings'][key]=audit['candidate_generation_timings'].get(key,0.)+value
        phases=audit['candidate_generation_phases']
        if phases:
            output.meta_info['async_rollout']=dict(role='fixed_candidate_pool_full_rollouts',phases=phases,
                completed_trajectories=sum(p['completed_trajectories'] for p in phases),
                generation_requests=sum(p['generation_requests'] for p in phases),
                environment_actions=sum(p['environment_actions'] for p in phases),
                peak_inflight_trajectories=max(p['peak_inflight_trajectories'] for p in phases),
                retrieval_concurrency=phases[0]['retrieval_concurrency'],
                phase_elapsed_seconds=sum(p['phase_elapsed_seconds'] for p in phases),
                selected_trajectories=len(output),actor_updates_during_phase=0)
    with (directory/(tag+'.json')).open('x') as f:
        json.dump(audit,f);f.flush();os.fsync(f.fileno())
    parents = parents.repeat(repeat_times=5,interleave=True)
    parents.pop(batch_keys=['input_ids','attention_mask','position_ids'])
    parents.batch['selection_fallback'] = torch.as_tensor(
        parents.non_tensor_batch['selection_source'] == 'all_correct_fallback')
    return parents, output, identity, audit

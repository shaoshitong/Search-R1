"""Independent, read-only CPU consumer for prefix20 joint4 artifacts.

No Torch import, subprocess, model reload, network, GPU allocation or training.
Missing actual COMPLETE200 is rejected before Arrow/tokenizer/Torch-ZIP reads.
Receipt PASS is explicitly a CPU recorded-evidence acceptance, not a new GPU
recovery claim. Actual recovery is independently consumed when supplied.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import string
import unicodedata
import sys
import time

if __package__:
    from .safe_pt import Archive
else:
    from safe_pt import Archive

STEPS = (5, 50, 100, 150, 200)
POOL_SHA = 'ae13710254e7ae77cf5531b62a755ec2000bc3fba20051bbafaff1f6714aa8fc'
TRAIN_SHA = '5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1'
MANIFEST_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
HEAD_SHA = '535c7bfee07374dde88b7d7f9e4f08ac874d46088ef961f15ea6ab7dd849a560'
MODEL_REV = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
SETTINGS = dict(probe_architecture='qwen2_prefix20_trajectory_em_v1', candidate_questions=5120,
    selected_questions=512, exploration_questions=0, responses_per_question=5,
    head_learning_rate=1e-6, head_batch_size=256, head_epochs_per_outer=1,
    replay_window_outer=0, seed=1729, eligible_pool_sha256=POOL_SHA)
BOUND_SOURCES = ('search_g1/prefix20_runtime.py', 'search_g1/prefix20_collection.py',
    'search_g1/prefix20_contract.py', 'search_g1/prefix20_probe_worker.py',
    'search_g1/prefix20_generation.py', 'search_g1/prefix20_split.py',
    'search_g1/prefix20_advantage.py', 'search_g1/prefix20_data.py',
    'verl/trainer/ppo/ray_trainer.py', 'verl/trainer/ppo/core_algos.py',
    'verl/workers/fsdp_workers.py', 'verl/utils/dataset/rl_dataset.py',
    'verl/utils/torch_functional.py', 'search_r1/qa_protocol.py',
    'search_g1/signals.py', 'verl/utils/reward_score/qa_em.py')


class Rejected(ValueError):
    pass


def need(value, reason):
    if not value:
        raise Rejected(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'DUPLICATE_JSON_KEY')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(Rejected('NONFINITE_JSON_NUMBER')))


def under(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def integer(value, lo=0, hi=None):
    return type(value) is int and lo <= value and (hi is None or value <= hi)


def same(actual, expected):
    if type(expected) is int:return type(actual) is int and actual==expected
    if type(expected) is float:return type(actual) in (int,float) and actual==expected
    return type(actual) is type(expected) and actual==expected


def finite(value, lo=None, hi=None):
    return type(value) in (int, float) and math.isfinite(value) and (lo is None or value >= lo) and (hi is None or value <= hi)


class Reads:
    def __init__(self, base):
        self.base = Path(base)
        need(self.base.is_absolute() and self.base.resolve() == self.base and self.base.is_dir(), 'EXPLICIT_REAL_BASE_REQUIRED')
        self.files = {}

    def path(self, relative):
        path = self.base / relative
        need(under(path, self.base) and path.resolve() == path and path.is_file() and not path.is_symlink(), 'UNSAFE_OWNED_ARTIFACT:' + str(relative))
        return path

    def record(self, path, expected=None, *, external=False):
        path = Path(path)
        if not external:
            need(under(path, self.base) and path.resolve() == path and not path.is_symlink(), 'UNSAFE_OWNED_FILE')
        need(path.is_file(), 'FILE_ABSENT:' + path.name)
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno()); h = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
            after = os.fstat(stream.fileno())
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) ==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'FILE_CHANGED_DURING_READ')
        digest = h.hexdigest()
        need(expected is None or expected == digest, 'FULL_SHA_MISMATCH:' + path.name)
        item = dict(path=str(path), bytes=after.st_size, sha256=digest, mtime_ns=after.st_mtime_ns)
        self.files[str(path)] = item
        return item

    def raw(self, relative):
        path = self.path(relative)
        need(path.stat().st_size < 2 * 1024**3, 'JSON_EVIDENCE_EXCEEDS_BOUND')
        record = self.record(path); value = path.read_bytes()
        need(hashlib.sha256(value).hexdigest() == record['sha256'], 'FILE_CHANGED_BEFORE_PARSE')
        return value

    def json(self, relative):
        return strict(self.raw(relative))

    def rows(self, relative):
        raw = self.raw(relative)
        need(not raw or raw.endswith(b'\n'), 'TRUNCATED_JSONL')
        return [strict(row) for row in raw.splitlines()]

    def pt(self, relative):
        path = self.path(relative); self.record(path)
        return Archive(path)


def early_history(reads, commit, run_id):
    """Deliberately first: a pending job cannot trigger expensive final audits."""
    path = reads.base / 'logs/actual-updates.jsonl'
    need(path.is_file(), 'MISSING_COMPLETE200: actual-updates.jsonl absent')
    updates = reads.rows('logs/actual-updates.jsonl')
    need([r.get('step') for r in updates] == list(range(1, 201)) and
         all(integer(r.get('step'), 1, 200) and r.get('run_id') == run_id for r in updates),
         'MISSING_COMPLETE200_OR_UPDATE201: exact fresh1..200 required')
    run = reads.json('logs/run-manifest.json')
    need(re.fullmatch('[0-9a-f]{40}', commit) and re.fullmatch('[0-9a-f]{32}', run_id), 'EXPLICIT_COMMIT_RUN_ID_REQUIRED')
    need(run.get('run_id') == run_id and run.get('source_commit') == commit and
         run.get('training_world_size') == 4 and run.get('actor_micro_batch_per_rank') == 4 and
         run.get('data_selection_mode') == 'prefix20_probe' and run.get('mixed_first_selection') is False and
         run.get('target_updates') == 200 and run.get('completed_updates') == 0 and
         run.get('model_revision') == MODEL_REV and not run.get('full_state_restore'), 'FRESH_PREFIX20_NATIVE4_BINDING_DIFFERS')
    need(all(same(run.get('question_probe', {}).get(k),v) for k,v in SETTINGS.items()), 'PREFIX20_SCIENTIFIC_SETTINGS_DIFFER')
    return run, updates


def source_and_config(reads, run):
    commit = run['source_commit']; sources = run['source_sha256']
    for relative in BOUND_SOURCES:
        need(relative in sources, 'REQUIRED_LOADED_SOURCE_NOT_BOUND:' + relative)
        reads.record(reads.path('repo/' + relative), sources[relative])
        reads.record(reads.path('source/' + commit + '/' + relative), sources[relative])
    resolved = reads.json('logs/resolved-training-config.json')
    need(resolved['run_id']==run['run_id'], 'RESOLVED_CONFIG_RUN_DIFFERS')
    config=resolved['resolved_config']
    need(resolved['resolved_config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(), 'RESOLVED_CONFIG_SHA_DIFFERS')
    actor = config['actor_rollout_ref']['actor']; rollout = config['actor_rollout_ref']['rollout']
    expected = dict(ppo_mini_batch_size=128, ppo_micro_batch_size=16, ppo_epochs=1, grad_clip=1.,
        clip_ratio=.2, state_masking=True, use_kl_loss=True, entropy_coeff=0., kl_loss_coef=0.,
        use_dynamic_bsz=False)
    need(all(actor.get(k) == v for k,v in expected.items()), 'ACTOR_PPO_MATHEMATICS_DIFFER')
    need(actor['optim']['lr'] == 1e-6 and actor['optim']['lr_warmup_steps'] == 57 and
         config['algorithm']['adv_estimator'] == 'grpo' and config['algorithm']['all_correct_virtual_zero'] is True,
         'ACTOR_LR_OR_VIRTUAL0_CONFIG_DIFFERS')
    need(config['trainer']['n_gpus_per_node'] == 4 and config['trainer']['total_training_steps'] == 201 and
         config['trainer']['save_freq'] == config['trainer']['test_freq'] == 50 and
         config['trainer']['acceptance_step'] == 5 and config['data']['selection_mode'] == 'prefix20_probe',
         'FOUR_GPU_HORIZON_OR_SELECTION_CONFIG_DIFFERS')
    need(rollout['n_agent'] == 5 and rollout['temperature'] == rollout['top_p'] == 1. and
         all(same(config['prefix20_probe'].get(k),v) for k,v in SETTINGS.items()), 'ACTUAL_PROBE_OR_SAMPLING_CONFIG_DIFFERS')
    init = reads.json('logs/prefix20-configuration-prepared.json')
    need(init['source_commit'] == commit and init['run_id'] == run['run_id'] and
         init['training_world_size'] == 4 and init['teacher_warmup_updates'] == 0 and
         init['initial_head_epoch'] == 68 and init['initial_head_optimizer'] == 'fresh Adam', 'PRETRAINED_ONLY_INITIALIZATION_DIFFERS')
    submission=reads.json('preparation/training-submission.json');claim=reads.json('logs/training-launch-claim.json')
    allocation=reads.json('logs/native-allocation-admission.json');job=submission['job']
    need(submission['source_commit']==commit and submission['returncode']==0 and submission['training_world_size']==4 and
         claim['run_id']==allocation['run_id']==run['run_id'] and claim['slurm_job_id']==allocation['slurm_job_id']==job and
         allocation['accepted'] is True and allocation['training_world_size']==4 and
         len(allocation['actual_cuda_runtime_uuids'])==len(set(allocation['actual_cuda_runtime_uuids']))==4 and
         len(allocation['actual_allocated_indices'])==len(set(allocation['actual_allocated_indices']))==4, 'ACTUAL_NATIVE4_ALLOCATION_BINDING_DIFFERS')
    actual=init['actual_slurm_job_binding']
    need(actual['JobId']==job and actual['Account']=='jxu957' and actual['UserId']=='jxu957(14090)' and
         actual['Partition']=='emergency_acd' and actual['QOS']=='formal-user' and actual['NumCPUs']=='48' and
         actual['TresPerNode'] in ('gres/gpu:4','gres:gpu:4') and
         actual['WorkDir']==str(reads.base/'repo') and
         actual['Command']==str(reads.base/'source'/commit/'reproduction/prefix20_joint4/train.sh'), 'ACTUAL_OWNED_SLURM_SUBMISSION_BINDING_DIFFERS')
    reads._job=job
    reads.record(Path(run['question_probe']['head_safetensors']), HEAD_SHA)
    return dict(source_files_checked=len(BOUND_SOURCES), actual_config_bound=True,
                native4=True, warmup57=True, all_correct_virtual_zero=True)


def actual_attempts(reads, run, updates):
    attempts = reads.rows('logs/actual-attempts.jsonl')
    completed = 0; writes = 0; ids = set(); by_id = {}; passed = []
    for ordinal, row in enumerate(attempts, 1):
        aid = row.get('attempt_id')
        need(isinstance(aid,str) and re.fullmatch('[0-9a-f]{32}',aid) and aid not in ids, 'ATTEMPT_ID_REPEATED')
        need(row['run_id'] == run['run_id'] and row['slurm_job_id']==reads._job and row['attempt_index'] == ordinal and row['logical_update'] == completed+1 <= 200,
             'ATTEMPT_OUTER_AFTER200_OR_BINDING_DIFFERS')
        need(row['all_ranks_agreed'] is True and row['real_rank_count'] == 4 and
             row['optimizer_writes_before'] == writes, 'ACTUAL_FOUR_RANK_COUNTERS_DIFFER')
        count = row['optimizer_writes_this_attempt']; processed = row['opportunities_processed']
        need(integer(count,0,20) and integer(processed,1,20) and count <= processed and
             row['optimizer_writes_after'] == writes+count and len(row['opportunity_statuses']) == processed and
             sum(s in ('written','written_with_post_guard') for s in row['opportunity_statuses']) == count,
             'ACTUAL_WRITE_OR_OPPORTUNITY_COUNTS_DIFFER')
        complete = count > 0
        need(row['complete_outer'] is complete and row['scheduler_advances'] == int(complete) and
             row['completed_outer'] == completed+int(complete), 'ATTEMPT_COMPLETION_ADVANCE_DIFFERS')
        if complete:
            item = updates[completed]
            need(item['attempt_id'] == aid and item['attempt_index'] == ordinal and
                 item['actual_optimizer_writes'] == count and item['optimizer_writes_before'] == writes and
                 item['optimizer_writes_after'] == writes+count, 'COMPLETED_UPDATE_NOT_ACTUAL_ATTEMPT')
            passed.append(row)
        ids.add(aid); by_id[aid] = row; writes += count; completed += int(complete)
    need(completed == 200, 'ACTUAL_ATTEMPTS_NOT_COMPLETE200')
    for rank in range(4):
        scheduler = reads.rows('logs/scheduler-rank-%d.jsonl' % rank)
        expected = [(a['attempt_id'],k) for a in attempts for k in
                    (('BEGIN','COMMIT_PREPARED','COMPLETE') if a['complete_outer'] else ('BEGIN','ABORTED_ATTEMPT'))]
        need([(r['attempt_id'],r['kind']) for r in scheduler] == expected, 'SCHEDULER_EXTRA_MISSING_OR_AFTER200')
        for row in scheduler:
            a = by_id[row['attempt_id']]; outer = a['logical_update']; terminal = row['kind'] != 'BEGIN'
            complete = row['kind'] == 'COMPLETE'; after = outer if complete else outer-1
            need(row['run_id'] == run['run_id'] and row['source_commit'] == run['source_commit'] and
                 row['rank'] == rank and row['world_size'] == 4 and row['slurm_job_id']==reads._job and row['logical_update'] == outer and
                 row['scheduler_last_epoch_before'] == outer-1 and row['scheduler_last_epoch_after'] == after and
                 row['scheduler_step_count_before'] == outer and row['scheduler_step_count_after'] == after+1 and
                 row['actual_optimizer_writes'] == (a['optimizer_writes_this_attempt'] if terminal else 0) and
                 row['writes_before'] == a['optimizer_writes_before'] and
                 row['writes_after'] == (a['optimizer_writes_after'] if terminal else a['optimizer_writes_before']) and
                 row['consumed_learning_rates'] and all(lr == 1e-6*min((outer-1)/57,1) for lr in row['consumed_learning_rates']),
                 'ACTUAL_SCHEDULER_COUNTER_OR_CONSUMED_LR_DIFFERS')
        worker = reads.rows('logs/optimizer-rank-%d.jsonl' % rank)
        retained = [r for r in worker if r['kind'] == 'OPTIMIZER_WRITE']
        need([r['write_count'] for r in retained] == list(range(1,writes+1)), 'ACTUAL_ADAM_WRITE_SEQUENCE_DIFFERS')
        for row in worker:
            context = row.get('context') or row; aid = context.get('attempt_id')
            need(aid in by_id and row['run_id'] == run['run_id'] and row['rank'] == rank and
                 context['logical_update'] == by_id[aid]['logical_update'] <= 200 and
                 context['attempt_index'] == by_id[aid]['attempt_index'], 'WORKER_EXTRA_OR_AFTER200')
        for a in attempts:
            scoped = [r for r in worker if (r.get('context') or r).get('attempt_id') == a['attempt_id']]
            need(len([r for r in scoped if r['kind']=='OPPORTUNITY']) == a['opportunities_processed'] and
                 len([r for r in scoped if r['kind']=='OPTIMIZER_WRITE']) == a['optimizer_writes_this_attempt'],
                 'ATTEMPT_PHYSICAL_WORKER_RECEIPTS_DIFFER')
    return attempts, dict(actual_attempts=len(attempts),completed_outers=200,
        aborted_attempts=len(attempts)-200,actual_actor_adam_writes_per_rank=writes,
        actor_full20_outers=sum(a['optimizer_writes_this_attempt']==20 for a in passed),no_update201=True)


def acquisition(probabilities):
    need(len(probabilities) == 5 and all(finite(p,0,1) for p in probabilities), 'INVALID_G5_PREDICTIONS')
    return max(0.,min(1.,1-math.prod(probabilities)-math.prod(1-p for p in probabilities)))


def verify_selection(selection, eligible):
    need(selection['stage'] == 'prefix20_selection_frozen_before_continuation' and
         selection['candidate_questions'] == 5120 and selection['selected_questions'] == 512 and
         selection['exploration_questions'] == 0, 'TOP512_ALL_BUDGET_DIFFERS')
    questions = selection['questions']
    need(len(questions) == 5120 and [q['ordinal'] for q in questions] == list(range(5120)) and
         len({q['qid'] for q in questions}) == 5120 and
         len({tid for q in questions for tid in q['trajectory_ids']}) == 25600, 'CANDIDATE_IDENTITY_OR_UNIQUENESS_DIFFERS')
    for q in questions:
        need(integer(q['pool_index'],0,len(eligible)-1) and integer(q['pool_cycle'],0), 'INVALID_POOL_INDEX_OR_CYCLE')
        record = eligible[q['pool_index']]
        need(record['qid'] == q['qid'] and record['source_index'] == q['canonical_source_index'] and record['normalized_question_hash']==q['content_sha256'], 'ELIGIBLE_CANONICAL_MEMBERSHIP_DIFFERS')
    ranked = sorted(range(5120), key=lambda i:(-acquisition(questions[i]['em_probabilities']),i))[:512]
    selected = selection['selected']
    need([r['candidate_ordinal'] for r in selected] == ranked, 'NOT_EXACT_TOP512_WITH_STABLE_TIES')
    for row,i in zip(selected,ranked):
        q = questions[i]
        need(row['qid'] == q['qid'] and row['trajectory_ids'] == q['trajectory_ids'] and
             row['em_probabilities'] == q['em_probabilities'] and
             row['mixed_probability'] == acquisition(q['em_probabilities']), 'TOP512_ACQUISITION_VALUE_OR_ID_CHANGED')
    need([q['ordinal'] for q in questions if q['selected']] == sorted(ranked), 'SELECTED_FLAGS_DIFFER')
    need(selection['prefix_freeze']['actual_optimizer_and_scheduler_unchanged'] is True and
         selection['prediction_freeze'] == dict(actual_optimizer_and_scheduler_unchanged=True,actor_rng_unchanged=True),
         'FROZEN_SELECTION_COUNTER_RECEIPTS_DIFFER')
    return ranked


def state_prefix(state):
    tokens=[]; positions=[]; segments=[]
    actions=[(turn['turn_index'],turn['native_chunks'][0]['actual_request_token_ids'],turn['prefix_cut']['kept_token_ids'])
             for turn in state['turns']]
    if state['current_raw']:
        actions.append((state['ordinal'],state['chunks'][0]['actual_request_token_ids'],state['current_raw']))
    for turn,request,raw in actions:
        take=min(20-len(tokens),len(raw))
        if take:
            segments.append(dict(turn_index=turn,policy_start=len(tokens),policy_length=take,
                position_start=len(request),request_length=len(request),context_tokens=request+raw[:take]))
            tokens.extend(raw[:take]);positions.extend(range(len(request),len(request)+take))
        if len(tokens)==20:break
    return dict(trajectory_id=state['trajectory_id'],qid=state['qid'],response_index=state['response_index'],
        initial_prompt_token_ids=state['prompt'],policy_token_ids=tokens,
        policy_position_ids=positions,prefix_segments=segments)


def native_chunks(turn, trajectory_id, snapshot, run=None):
    need(turn['role'] == 'prefix20_composite_logical_action_v1' and
         turn['trajectory_id'] == trajectory_id and turn['policy_snapshot_id'] == snapshot,
         'NATIVE_LOGICAL_TURN_IDENTITY_DIFFERS')
    chunks = turn['native_chunks']; raw=[]; probs=[]; first=None;stop_set=None
    need(chunks and len(chunks) <= 21, 'MISSING_NATIVE_CHUNKS')
    for ordinal,c in enumerate(chunks):
        request=c['actual_request_token_ids'];maximum=c['sampling']['max_tokens']
        if first is None:first=request
        need(request == first+raw and c['row_id'] == trajectory_id and c['policy_snapshot_id'] == snapshot and
             c['rollout_ordinal'] == turn['turn_index'] and integer(maximum,1,500-len(raw)), 'NATIVE_RESUME_REQUEST_OR_BUDGET_CHANGED')
        sampler=c['sampling']; required=dict(n=1,temperature=1.,top_p=1.,top_k=-1,min_p=0.,
            presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,ignore_eos=False,logprobs=1)
        need(all(sampler.get(k) == v for k,v in required.items()), 'NATIVE_SAMPLER_CHANGED')
        proof=c['effective_stop_provenance'];effective=proof['effective_sampling'];generation=proof['generation_config']
        need(proof['request_id']==c['request_id'] and proof['rollout_call_id']==c['rollout_call_id'] and
             proof['row_id']==trajectory_id and proof['caller_is_actual_group_input'] is True and
             proof['actual_request_tokens_sha256']==sha_value(request) and
             proof['caller_sampling_sha256']==sha_value(sampler) and proof['effective_sampling_sha256']==sha_value(effective) and
             {k:v for k,v in effective.items() if k!='stop_token_ids'}=={k:v for k,v in sampler.items() if k!='stop_token_ids'}, 'NATIVE_EFFECTIVE_SAMPLER_OR_REQUEST_PROVENANCE_DIFFERS')
        caller_stop=set(sampler['stop_token_ids'] or []);generation_stop=set(generation['generation_config_eos_token_ids'])
        seq_eos=proof['seq_eos_token_id'];stops=caller_stop|generation_stop|({seq_eos} if seq_eos is not None else set())
        need(set(proof['effective_all_stop_token_ids'])==stops and
             set(proof['effective_stop_token_ids'])==caller_stop|(generation_stop-{seq_eos}) and
             effective['stop_token_ids']==proof['effective_stop_token_ids'] and
             proof['tokenizer_eos_token_id']==c['eos_token_id'] and proof['tokenizer_pad_token_id']==c['pad_token_id'], 'NATIVE_EFFECTIVE_STOP_SET_DIFFERS')
        if run is not None:
            need(generation['run_id']==run['run_id'] and generation['source_commit']==run['source_commit'] and
                 generation['source_map_sha256']==sha_value(run['source_sha256']), 'NATIVE_GENERATION_LOADED_SOURCE_BINDING_DIFFERS')
        if stop_set is None:stop_set=stops
        need(stops==stop_set, 'NATIVE_STOP_SET_CHANGED_DURING_ACTION')
        ids=c['raw_generated_token_ids'];lp=c['selected_logprobs']
        need(len(ids) == c['raw_generated_token_count'] <= maximum and len(lp) == len(ids) and
             all(integer(t) for t in ids) and all(finite(v,hi=0) for v in lp), 'NATIVE_TOKEN_LOGPROB_COUNTS_DIFFER')
        if ordinal<len(chunks)-1:
            need(len(ids)==maximum and c['finish_reason']=='length' and not stops.intersection(ids), 'CONTINUATION_AFTER_NATIVE_TERMINAL')
        raw.extend(ids);probs.extend(lp)
    count=turn['kept_token_count'];cut=turn['prefix_cut']
    need(integer(count,0,len(raw)) and not (stop_set|{chunks[0]['pad_token_id'],chunks[0]['eos_token_id']}).intersection(raw[:count]) and cut['kept_token_ids'] == raw[:count] and
         cut['raw_tokens_sha256'] == sha_value(raw) and cut['retained_tokens_sha256'] == sha_value(raw[:count]),
         'NATIVE_PREFIX_CUT_CHANGED')
    return first,raw[:count],probs[:count]


def normalize(text):
    text=re.sub(r'\b(a|an|the)\b',' ', ''.join(c for c in text.lower() if c not in string.punctuation))
    return ' '.join(text.split())


def final_answer(text):
    eos=re.compile(r'(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|</s>))+$')
    reserved=re.compile(r'</?(?:think|search|answer|information)\b')
    text=eos.sub('',text.strip()).strip();actions=list(re.finditer(r'<(search|answer)>(.*?)</\1>',text,re.S))
    if len(actions)!=1:return None
    action=actions[0];payload=action.group(2).strip();prefix=text[:action.start()].strip()
    if not payload or reserved.search(payload) or text[action.end():].strip():return None
    if reserved.search(prefix):
        thought=re.fullmatch(r'<think>(.*?)</think>',prefix,re.S)
        if thought is None or reserved.search(thought.group(1)):return None
    return payload if action.group(1)=='answer' else None


def strict_label(actor_row, tokenizer, target):
    width=actor_row['prompt_width'];ids=actor_row['input_ids'][width:]
    att=actor_row['attention_mask'][width:];mask=actor_row['info_mask'][width:]
    active=[(token,m) for token,a,m in zip(ids,att,mask) if a]
    need(active and all(m in (0,1) for _,m in active), 'INVALID_AUTHORED_MASK')
    spans=[];start=0
    while start<len(active):
        end=start+1
        while end<len(active) and active[end][1]==active[start][1]:end+=1
        if active[start][1]:spans.append(tokenizer.decode([token for token,_ in active[start:end]],skip_special_tokens=False))
        start=end
    answer=final_answer(spans[-1]) if spans else None
    targets=[target] if isinstance(target,str) else target
    return int(answer is not None and any(normalize(answer)==normalize(t) for t in targets))


def binary_metrics(probabilities, labels):
    clipped=[min(1-1e-12,max(1e-12,p)) for p in probabilities]
    return dict(binary_accuracy=sum((p>=.5)==y for p,y in zip(probabilities,labels))/len(labels),
        brier=math.fsum((p-y)**2 for p,y in zip(probabilities,labels))/len(labels),
        bce=-math.fsum(y*math.log(p)+(1-y)*math.log1p(-p) for p,y in zip(clipped,labels))/len(labels),
        positive_fraction=sum(labels)/len(labels),trajectories=len(labels))


class CanonicalTokenizer:
    def __init__(self,backend,config):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.backend=backend;self.config=config
        environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
        def reject(message):raise Rejected('CANONICAL_CHAT_TEMPLATE_REJECTED:'+str(message))
        environment.globals['raise_exception']=reject
        self.template=environment.from_string(config['chat_template']) if config.get('chat_template') else None

    def decode(self,ids,skip_special_tokens=False):
        return self.backend.decode(ids,skip_special_tokens=skip_special_tokens)

    @lru_cache(maxsize=8192)
    def expected_prompt(self,chat_json):
        chat=json.loads(chat_json)
        text=self.template.render(messages=chat,add_generation_prompt=True,**self.config) if self.template else chat[0]['content']
        return self.backend.encode(text,add_special_tokens=False).ids[-2048:]


def canonical_inputs(reads, run):
    """Arrow and tokenizer imports happen only after real COMPLETE200 gate."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    pool_path=Path(run['question_probe']['eligible_pool_path']);reads.record(pool_path,POOL_SHA)
    pool=strict(pool_path.read_bytes())
    need(pool['canonical_train_sha256']==TRAIN_SHA and pool['canonical_rows']==169028 and
         pool['eligible_unique_questions']==168510 and len(pool['records'])==168510 and
         pool['excluded_diagnostic_questions']==512, 'ELIGIBLE_POOL_CONTRACT_DIFFERS')
    data=reads.base/'data';reads.record(data/'manifest.json',MANIFEST_SHA,external=True)
    manifest=strict((data/'manifest.json').read_bytes())
    frames={}
    for split in ('train','stage','validation','final'):
        record=manifest['splits'][split];path=data/record['path']
        reads.record(path,record['sha256'],external=True)
        frames[split]=pq.read_table(path,columns=['prompt','data_source','reward_model']+(['extra_info'] if split=='train' else [])).to_pylist()
        need(len(frames[split])==record['rows']==dict(train=169028,stage=64,validation=512,final=51713)[split], 'CANONICAL_SPLIT_ROW_COUNT_DIFFERS')
    need(manifest['splits']['train']['sha256']==TRAIN_SHA, 'CANONICAL_TRAIN_SHA_DIFFERS')
    model=Path(run['initial_model']);tokenizer_path=model/'tokenizer.json'
    reads.record(tokenizer_path,run['initial_model_sha256']['tokenizer.json'],external=True)
    tokenizer_config=model/'tokenizer_config.json'
    reads.record(tokenizer_config,run['initial_model_sha256']['tokenizer_config.json'],external=True)
    need(not strict(tokenizer_config.read_bytes()).get('clean_up_tokenization_spaces',False), 'TOKENIZER_CLEANUP_PROJECTION_NOT_SUPPORTED')
    for record in pool['records']:
        source=frames['train'][record['source_index']];question='\n'.join(m['content'] for m in source['prompt'])
        normalized=' '.join(unicodedata.normalize('NFKC',question).split())
        index=(source.get('extra_info') or {}).get('index')
        if index is None:index=hashlib.sha256(normalized.encode()).hexdigest()
        qid=json.dumps([str(source['data_source']),str(index)])
        need(qid==record['qid'] and hashlib.sha256(question.encode()).hexdigest()==record['question_hash'] and
             hashlib.sha256(normalized.encode()).hexdigest()==record['normalized_question_hash'], 'ELIGIBLE_CANONICAL_QUESTION_ID_OR_TEXT_DIFFERS')
    return pool['records'],frames,manifest,CanonicalTokenizer(Tokenizer.from_file(str(tokenizer_path)),strict(tokenizer_config.read_bytes()))


def audit_outer(reads, run, attempt, completed, eligible, train, tokenizer):
    aid=attempt['attempt_id'];outer=attempt['logical_update'];relative='artifacts/question-probe/attempts/'+aid+'/'
    selection=reads.json(relative+'selection.json');audit=reads.json(relative+'rollout-audit.json')
    snapshot=run['run_id']+':'+str(outer)+':policy-'+aid
    need(selection['outer']==outer and selection['attempt_id']==aid and selection['policy_snapshot_id']==snapshot and
         selection['head_version']=='prefix20-epoch68-outer-%04d'%(outer-1), 'OUTER_SELECTION_BINDING_DIFFERS')
    ordinals=verify_selection(selection,eligible);questions=selection['questions']
    for q in questions:
        cycle=q['pool_cycle'];seen=reads._candidate_cycles.setdefault(cycle,set())
        need(q['pool_index'] not in seen, 'ELIGIBLE_QUESTION_REPEATED_WITHIN_ACTUAL_POOL_CYCLE')
        seen.add(q['pool_index'])
    selected_ids=[tid for i in ordinals for tid in questions[i]['trajectory_ids']]
    need(audit['step']==outer and audit['attempt_id']==aid and audit['policy_snapshot_id']==snapshot and
         audit['selection_mode']=='prefix20_probe' and audit['selected_groups']==512 and
         audit['selected_random_exploration']==0 and audit['selected_predicted_mixed']==512 and
         audit['raw_generated_trajectories']==25600 and audit['completed_selected_trajectories']==2560,
         'ACTUAL_ROLLOUT_BUDGET_OR_ID_DIFFERS')
    need([(p['phase'],p['trajectories'],p['actor_updates_during_phase']) for p in audit['phases']] ==
         [('candidate_prefix',25600,0),('selected_continuation',2560,0)], 'ACTUAL_PREFIX_CONTINUATION_PHASES_DIFFER')
    with reads.pt(relative+'candidate-prefixes.pt') as archive:
        candidate=archive.value;identity=candidate['identity'];prefixes=candidate['prefix_rows'];states=candidate['states']
        need(identity['run_id']==run['run_id'] and identity['source_sha256']==run['source_sha256'] and
             identity['logical_update']==outer and identity['attempt_id']==aid and identity['policy_snapshot_id']==snapshot and
             identity['trajectory_ids']==[tid for q in questions for tid in q['trajectory_ids']], 'AUTHENTIC_CANDIDATE_IDENTITY_DIFFERS')
        need(candidate['stream']['consumed_source_rows']==5120*attempt['attempt_index'], 'ACTUAL_CANDIDATE_SOURCE_CONSUMPTION_DIFFERS')
        need(len(prefixes)==len(states)==25600 and candidate['selection']==selection, 'RAW_CANDIDATE_SELECTION_DIFFER')
        for i,(prefix,state) in enumerate(zip(prefixes,states)):
            q=questions[i//5]
            canonical_row=train[q['canonical_source_index']]
            expected_prompt=tokenizer.expected_prompt(json.dumps(canonical_row['prompt'],ensure_ascii=False))
            need(state['prompt']==expected_prompt, 'NATIVE_PREFIX_QUESTION_PROMPT_NOT_CANONICAL_TOKENIZATION')
            for turn in state['turns']:native_chunks(turn,state['trajectory_id'],snapshot,run)
            if state['chunks']:
                partial=dict(role='prefix20_composite_logical_action_v1',trajectory_id=state['trajectory_id'],
                    policy_snapshot_id=snapshot,turn_index=state['ordinal'],native_chunks=state['chunks'],
                    kept_token_count=state['cut']['kept_token_count'],prefix_cut=state['cut'])
                request,kept,lp=native_chunks(partial,state['trajectory_id'],snapshot,run)
                need(state['current_raw']==[t for c in state['chunks'] for t in c['raw_generated_token_ids']], 'PAUSED_RAW_PREFIX_NOT_OWN_NATIVE_OUTPUT')
            need(state_prefix(state)==prefix and prefix['trajectory_id']==identity['trajectory_ids'][i] and
                 prefix['qid']==q['qid'] and prefix['response_index']==i%5 and 1<=len(prefix['policy_token_ids'])<=20 and
                 (len(prefix['policy_token_ids'])==20 or state['status']=='complete'), 'AUTHENTIC_OWN_PREFIX20_TOKEN_OR_POSITION_DIFFERS')
        selected_prefixes=[prefixes[5*i+j] for i in ordinals for j in range(5)]
        selected_states=[states[5*i+j] for i in ordinals for j in range(5)]
        # Release the unselected raw candidate metadata before loading the full rollout.
        del candidate,states,prefixes
    with reads.pt(relative+'real-rollout.pt') as archive:
        raw=archive.value;pending=raw['probe_transaction'];vectors=raw['strict_correctness']
        need(raw['selected_ordinals']==ordinals and raw['identity']['trajectory_ids']==selected_ids and
             raw['identity']['policy_snapshot_id']==snapshot and pending['attempt_id']==aid and
             pending['outer']==outer and pending['policy_snapshot_id']==snapshot, 'SAME_SELECTED2560_CONTINUATION_IDENTITY_CHANGED')
        rows=pending['rows'];labels=[y for group in vectors for y in group]
        need(len(vectors)==512 and all(len(g)==5 and all(type(y) is int and y in (0,1) for y in g) for g in vectors) and
             len(rows)==2560 and [{k:v for k,v in r.items() if k!='strict_em'} for r in rows]==selected_prefixes and
             [r['strict_em'] for r in rows]==labels, 'OWN_TRAJECTORY_TRAINING_ROWS_OR_LABELS_DIFFER')
        expected_p={tid:p for i in ordinals for tid,p in zip(questions[i]['trajectory_ids'],questions[i]['em_probabilities'])}
        need(pending['probabilities']==expected_p, 'PREQUENTIAL_SELECTED_PREDICTIONS_DIFFER')
    freeze_keys=('run_id','source_commit','rank','world_size','actual_optimizer_writes','completed_outer',
                 'optimizer_calls','optimizer_rollbacks','actual_adam_steps','actual_scheduler_last_epoch')
    for rank in range(4):
        prior=None;prior_rng=None
        for phase in ('before-prefix','after-prefix','after-prediction','after-continuation'):
            with reads.pt(relative+phase+'-worker-rng/rank-%d.pt'%rank) as state_archive:
                state=state_archive.value;signature={k:state[k] for k in freeze_keys}
                need(state['run_id']==run['run_id'] and state['source_commit']==run['source_commit'] and
                     state['rank']==rank and state['world_size']==4 and state['completed_outer']==outer-1 and
                     state['actual_optimizer_writes']==attempt['optimizer_writes_before'], 'ACTUAL_FROZEN_WORKER_COUNTERS_DIFFERS')
                if prior is not None:need(signature==prior, 'ACTOR_CHANGED_DURING_SELECTION_OR_CONTINUATION')
                rng=state_archive.signature(state['rng'])
                if phase=='after-prediction':need(rng==prior_rng, 'PROBE_PREDICTION_MUTATED_ACTOR_RNG')
                prior=signature;prior_rng=rng
    gate=reads.json('logs/behavior-probability-attempt-'+aid+'.json')
    if completed:
        update_row=reads._update_by_id[aid]
        need(update_row['behavior_probability_gate_sha256']==reads.files[str(reads.base/('logs/behavior-probability-attempt-'+aid+'.json'))]['sha256'], 'COMPLETED_UPDATE_GATE_SHA_DIFFERS')
    trace_path='artifacts/behavior-traces/attempt-'+aid+'.json'
    trace=reads.json(trace_path)
    need(gate['source_sha256']==run['source_sha256'] and gate['policy_snapshot_id']==snapshot, 'SAMPLING_GATE_SOURCE_OR_POLICY_DIFFERS')
    need(gate['accepted'] is True and gate['run_id']==run['run_id'] and gate['logical_update']==outer and
         gate['attempt_id']==aid and gate['private_trace_sha256']==reads.files[str(reads.base/trace_path)]['sha256'] and
         gate['trajectory_count']==2560 and gate['actual_frozen_old_all_finite'] is True, 'SAMPLING_GATE_RAW_TRACE_OR_ACTUAL_FINITE_BINDING_DIFFERS')
    need(trace['run_id']==run['run_id'] and trace['source_sha256']==run['source_sha256'] and
         trace['logical_update']==outer and trace['policy_snapshot_id']==snapshot and
         [t['trajectory_id'] for t in trace['trajectories']]==selected_ids, 'ACTUAL_RAW_BEHAVIOR_TRACE_BINDING_DIFFERS')
    need(trace['slurm_job_id']==gate['slurm_job_id'] and trace['sampling']['temperature']==trace['sampling']['top_p']==1., 'RAW_TRACE_JOB_OR_SAMPLER_DIFFERS')
    all_calls=[(chunk['request_id'],chunk['rollout_call_id']) for t in trace['trajectories'] for turn in t['turns'] for chunk in turn['native_chunks']]
    need(len({r for r,c in all_calls})==len(all_calls) and len({c for r,c in all_calls})==len(all_calls), 'NATIVE_REQUEST_OR_CALL_ID_REUSED')
    actor_rows={r['trajectory_id']:r for r in trace['actor_rows']}
    need(len(actor_rows)==2560, 'RAW_ACTOR_ROWS_DUPLICATE_OR_MISSING')
    measured=[]
    for i,(tid,state,trajectory) in enumerate(zip(selected_ids,selected_states,trace['trajectories'])):
        turns=trajectory['turns'];need(trajectory['parent_occurrence_index']==i//5 and trajectory['sample_index']==i%5, 'G5_GROUPS_CHANGED')
        need(turns[:len(state['turns'])]==state['turns'], 'COMPLETED_PREFIX_ACTION_REGENERATED')
        if state['chunks']:
            need(len(turns)>len(state['turns']), 'PAUSED_PREFIX_NEVER_RESUMED')
            turn=turns[len(state['turns'])]
            need(turn['native_chunks'][:len(state['chunks'])]==state['chunks'], 'PAUSED_NATIVE_PREFIX_REGENERATED')
        row=actor_rows[tid];width=row['prompt_width']
        need([t for t,a in zip(row['input_ids'][:width],row['attention_mask'][:width]) if a]==state['prompt'], 'ACTOR_PROMPT_CHANGED')
        response=[t for t,a in zip(row['input_ids'][width:],row['attention_mask'][width:]) if a]
        model_mask=[m for m,a in zip(row['info_mask'][width:],row['attention_mask'][width:]) if a]
        cursor=0
        for j,turn in enumerate(turns):
            need(turn['turn_index']==j, 'ACTUAL_ACTION_ORDER_CHANGED')
            request,kept,lp=native_chunks(turn,tid,snapshot,run)
            need(request==state['prompt']+response[:cursor] and response[cursor:cursor+len(kept)]==kept and
                 model_mask[cursor:cursor+len(kept)]==[1]*len(kept),
                 'ACTOR_RESUMED_TOKENS_NOT_ORIGINAL_NATIVE_SAMPLE')
            cursor+=len(kept);environment=turn['observation_token_ids']+turn['correction_token_ids']
            need(response[cursor:cursor+len(environment)]==environment and model_mask[cursor:cursor+len(environment)]==[0]*len(environment), 'ACTOR_ENVIRONMENT_CONTEXT_CHANGED')
            cursor+=len(environment)
        need(cursor==len(response), 'EXTRA_ACTOR_RESPONSE_TOKENS')
        canonical_row=train[questions[ordinals[i//5]]['canonical_source_index']]
        measured.append(strict_label(row,tokenizer,canonical_row['reward_model']['ground_truth']['target']))
    need(measured==labels, 'STRICT_TERMINAL_OWN_TRAJECTORY_EM_DIFFERS')
    correct=[sum(g) for g in vectors];mixed=sum(0<k<5 for k in correct)/512
    need(audit['selected_mixed_fraction']==mixed and audit['selected_mixed']==sum(0<k<5 for k in correct), 'ACTUAL_MIXED_FRACTION_DIFFERS')
    if completed:
        event=reads.json('artifacts/question-probe/completed/outer-%04d.json'%outer)
        need(event['outer']==outer and event['attempt_id']==aid and event['actor_snapshot_id']==run['run_id']+':'+str(outer) and
             event['label_policy_snapshot_id']==snapshot and event['prediction_version']==selection['head_version'] and
             event['updated_probe_version']=='prefix20-epoch68-outer-%04d'%outer, 'ACTOR_BEFORE_PROBE_TRANSACTION_BINDING_DIFFERS')
        update=event['head_update'];expected=dict(optimizer_steps_per_rank=10,actual_adam_steps_per_rank=outer*10,
            local_batch=64,global_batch=256,new_trajectories=2560,learning_rate=1e-6,epochs=1,replay_trajectories=0)
        need(all(update.get(k)==v for k,v in expected.items()) and update['selected_mixed_fraction']==mixed, 'ACTUAL_PROBE_BUDGET_OR_REPLAY_DIFFERS')
        before=binary_metrics([expected_p[t] for t in selected_ids],labels)
        need(all(math.isclose(update['before_'+k],v,rel_tol=1e-10,abs_tol=1e-12) for k,v in before.items()), 'PREQUENTIAL_BCE_ACCURACY_BRIER_RECOMPUTATION_DIFFERS')
        ranks=event['rank_receipts'];need([r['rank'] for r in ranks]==list(range(4)), 'ACTUAL_PROBE_FOUR_RANK_RECEIPTS_MISSING')
        for rank in ranks:
            need(rank['completed_outer']==outer and rank['actual_steps']==10*outer and rank['local_trajectories']==640 and
                 rank['global_batch']==256 and rank['learning_rate']==1e-6 and
                 finite(rank['own_trajectory_em_bce'],0) and finite(rank['training_binary_accuracy'],0,1), 'ACTUAL_PROBE_RANK_COUNTER_OR_FINITE_BCE_DIFFERS')
    else:
        event=reads.json('artifacts/question-probe/aborted/'+aid+'.json')
        need(event['outer']==outer and event['attempt_id']==aid and event['probe_updated'] is False, 'ABORTED_ACTOR_ADVANCED_PROBE')
    return dict(outer=outer,attempt_id=aid,completed=completed,selected_mixed_fraction=mixed,
        selected_strict_correct=sum(labels),all_correct_groups=correct.count(5),all_wrong_groups=correct.count(0),
        raw_authentic_prefixes=25600,same_completed_selected_trajectories=2560,
        selected_ids_sha256=sha_value(selected_ids),raw_strict_em_recomputed=True)


def question_id(prompt):
    text=prompt[0]['content'];matches=list(re.finditer(r'Question:\s*',text))
    need(matches, 'UNKNOWN_CANONICAL_PROMPT')
    return hashlib.sha256(re.sub(r'\s+',' ',text[matches[-1].end():]).strip().casefold().encode()).hexdigest()


def evaluate_identity(proof, expected, split, step, run_id, manifest):
    ids=[question_id(r['prompt']) for r in expected];sources=[r['data_source'] for r in expected]
    em=proof['em'];f1=proof['f1']
    need(proof['run_id']==run_id and proof['split']==split and proof['step']==step and
         proof['final'] is (split=='final') and proof['question_ids']==ids and proof['data_sources']==sources and
         proof['validation_sha256']==manifest['splits'][split]['sha256'] and
         proof['counts']==dict(Counter(sources)), 'ORDERED_CANONICAL_EVALUATION_IDENTITY_DIFFERS')
    need(len(em)==len(f1)==len(ids) and all(type(v) in (int,float) and v in (0,1) for v in em) and
         all(finite(v,0,1) for v in f1) and all(a<=b for a,b in zip(em,f1)), 'EM_F1_NOT_FINITE_OR_ROW_ALIGNED')
    need(proof['correct']==sum(em) and math.isclose(proof['weighted_em'],sum(em)/len(em),abs_tol=1e-15) and
         math.isclose(proof['weighted_f1'],math.fsum(f1)/len(f1),abs_tol=1e-15), 'WEIGHTED_EVALUATION_METRICS_DIFFER')
    per_source={}
    for source in sorted(set(sources)):
        pos=[i for i,s in enumerate(sources) if s==source]
        per_source[source]=dict(rows=len(pos),correct=sum(em[i] for i in pos),
            em=sum(em[i] for i in pos)/len(pos),f1=math.fsum(f1[i] for i in pos)/len(pos))
    return dict(split=split,step=step,rows=len(ids),ordered_question_ids_sha256=sha_value(ids),
        correct=sum(em),weighted_em=sum(em)/len(em),weighted_f1=math.fsum(f1)/len(f1),
        macro_em=math.fsum(v['em'] for v in per_source.values())/len(per_source),
        macro_f1=math.fsum(v['f1'] for v in per_source.values())/len(per_source),per_source=per_source)


def validations(reads,run,frames,manifest):
    ledger=reads.rows('logs/validation-identities.jsonl');result=[]
    # Actual producer uses stage64 at5 AND a separate formal development512
    # after5. Both remain independently identified; never rename stage64 as512.
    for step in STEPS:
        split='validation'
        matches=[r for r in ledger if r.get('step')==step and r.get('split')==split and not r.get('final')]
        need(len(matches)==1, 'FIXED_BOUNDARY_EVALUATION_MISSING_OR_DUPLICATE')
        result.append(evaluate_identity(matches[0],frames[split],split,step,run['run_id'],manifest))
    proof=reads.json('logs/final-identity.json')
    final=evaluate_identity(proof,frames['final'],'final',200,run['run_id'],manifest)
    need(final['rows']==51713 and len(final['per_source'])==7, 'FINAL_NOT_NATIVE51713_SEVEN_SOURCES')
    need(len([r for r in ledger if r.get('final')])==1 and next(r for r in ledger if r.get('final'))==proof,
         'FINAL_IDENTITY_LEDGER_DIFFERS')
    return dict(fixed_boundaries=result,final_native51713=final,
        boundary5_semantics='fixed512 after5 consumed from actual validation identity ledger; separate stage64 is not relabeled')


def checkpoints(reads, run):
    result=[]
    for outer in STEPS:
        rel='checkpoints/search-g1-common8/actor/global_step_%d/'%outer
        complete=reads.json(rel+'complete.json');manifest=reads.json(rel+'training_state/manifest.json')
        need(complete['run_id']==run['run_id'] and complete['step']==outer and complete['finite_all_saved_tensors'] is True and
             manifest['completed_outer']==outer, 'ACTUAL_COMPLETED_CHECKPOINT_BINDING_DIFFERS')
        for name,digest in complete['files_sha256'].items():reads.record(reads.path(rel+name),digest)
        with reads.pt(rel+'training_state/question-probe-state.pt') as archive:
            qstate=archive.value
            need(qstate['schema_version']==3 and qstate['kind']=='prefix20_trajectory_probe' and
                 qstate['completed_outer']==outer and qstate['actor_snapshot_id']==run['run_id']+':'+str(outer) and
                 qstate['head_version']=='prefix20-epoch68-outer-%04d'%outer and
                 len(qstate['completed_attempts'])==len(set(qstate['completed_attempts']))==outer and
                 [r['rank'] for r in qstate['rank_files']]==list(range(4)), 'PREFIX20_PAIRED_STATE_SCHEMA_OR_COUNTER_DIFFERS')
            candidate_signature=archive.signature(qstate['candidate_state'])
            probe_attempts=set(qstate['completed_attempts'])
            expected_attempts={r['attempt_id'] for r in reads._updates[:outer]}
            need(probe_attempts==expected_attempts, 'CHECKPOINT_PROBE_ACTOR_COMPLETED_ATTEMPT_IDS_DIFFER')
            reference=None
            for record in qstate['rank_files']:
                rank=record['rank'];name='prefix20-rank-%d.pt'%rank
                need(Path(record['path']).name==name, 'PROBE_RANK_FILENAME_DIFFERS')
                reads.record(reads.path(rel+'training_state/'+name),record['sha256'])
                with reads.pt(rel+'training_state/'+name) as state_archive:
                    state=state_archive.value
                    need(state['rank']==rank and state['completed_outer']==outer and state['actual_steps']==10*outer and
                         state['model'] and state['optimizer']['state'] and set(state['rng'])=={'python','numpy','torch_cpu','torch_cuda'} and
                         all(g['lr']==1e-6 for g in state['optimizer']['param_groups']), 'PROBE_FINAL_ACTUAL_MODEL_ADAM_RNG_COUNTER_DIFFERS')
                    need(all(state_archive.scalar(v['step'])==outer*10 for v in state['optimizer']['state'].values()), 'PROBE_ACTUAL_ADAM_STEP_NOT_10_PER_OUTER')
                    signature=state_archive.signature(dict(model=state['model'],optimizer=state['optimizer']))
                    if reference is None:reference=signature
                    else:need(reference==signature, 'FOUR_PROBE_MODEL_ADAM_STORAGE_BYTES_DIVERGE')
            need(qstate['candidate_state']['prefix20_pool_binding']['eligible_pool_sha256']==POOL_SHA,
                 'SAVED_CANDIDATE_ELIGIBLE_BINDING_DIFFERS')
        with reads.pt(rel+'training_state/driver.pt') as driver_archive:
            driver=driver_archive.value
            need(driver['completed_outer']==outer and driver['attempt_progress']['completed_outer']==outer and
                 driver_archive.signature(driver['candidate_stream'])==candidate_signature, 'SAVED_ACTOR_PROBE_CANDIDATE_DATA_DIFFER')
        index=reads.json(rel+'model.safetensors.index.json')
        need(len(index['weight_map'])==complete['validated_tensor_count'] and
             all(name in complete['files_sha256'] for name in set(index['weight_map'].values())), 'HF_CHECKPOINT_INDEX_OR_SHARD_COVERAGE_DIFFERS')
        need([r['rank'] for r in manifest['worker_files']]==list(range(4)), 'ACTOR_CHECKPOINT_FOUR_RANK_STATES_MISSING')
        for record in manifest['worker_files']:
            name=Path(record['path']).name
            with reads.pt(rel+'training_state/'+name) as archive:
                state=archive.value
                need(state['run_id']==run['run_id'] and state['rank']==record['rank'] and state['world_size']==4 and
                     state['completed_outer']==outer and state['scheduler']['last_epoch']==outer and
                     state['scheduler']['_step_count']==outer+1 and state['optimizer']['state'], 'ACTOR_ADAM_SCHEDULER_RNG_SAVED_BOUNDARY_DIFFERS')
                need(all(archive.scalar(v['step'])==state['actual_optimizer_writes'] for v in state['optimizer']['state'].values()),
                     'ACTOR_ACTUAL_ADAM_COUNTER_DIFFERS')
        result.append(dict(outer=outer,actor_ranks=4,probe_ranks=4,probe_actual_adam_steps_per_rank=outer*10,
                           full_recorded_checkpoint_sha_checked=True))
    actual_steps=[]
    for path in (reads.base/'checkpoints/search-g1-common8/actor').glob('global_step_*'):
        need(path.name[len('global_step_'):].isdigit(), 'UNKNOWN_CHECKPOINT_STEP_NAME')
        actual_steps.append(int(path.name[len('global_step_'):]))
    need(set(actual_steps)==set(STEPS), 'EXTRA_MISSING_CHECKPOINT_OR_UPDATE201')
    return result


def recovery(reads,run,path,expected_sha):
    """Strictly consume a producer-independent restore receipt, never perform it.

    This schema is deliberately explicit; current producer has no final200
    restore-only receipt. Absent receipt reports PENDING, never acceptance.
    """
    if path is None:
        return dict(accepted=False,status='PENDING_ACTUAL_NATIVE4_FINAL200_RECOVERY_RECEIPT')
    path=Path(path);need(path.is_absolute() and path.resolve()==path and not path.is_symlink(), 'UNSAFE_RECOVERY_RECEIPT')
    reads.record(path,expected_sha,external=True);value=strict(path.read_bytes())
    need(value.get('kind')=='prefix20_final200_native4_actual_restore' and value.get('source_base')==str(reads.base) and
         value.get('source_run_id')==run['run_id'] and value.get('source_commit')==run['source_commit'] and
         value.get('completed_outer')==200 and value.get('world_size')==4 and
         value.get('accepted') is True and value.get('restore_only') is True, 'ACTUAL_RECOVERY_IDENTITY_OR_KIND_DIFFERS')
    need(value.get('optimizer_step_increments')==0 and value.get('scheduler_step_increments')==0 and
         value.get('probe_adam_step_increments')==0 and value.get('generation_calls')==0 and
         value.get('training_updates')==0 and value.get('dataequal') is True, 'RECOVERY_PERFORMED_UPDATE_OR_DATA_CHANGED')
    rank_rows=value.get('actual_rank_readbacks',[])
    need([r.get('rank') for r in rank_rows]==list(range(4)), 'RECOVERY_FOUR_ACTUAL_RANKS_MISSING')
    for rank in rank_rows:
        need(rank.get('actor_model_optimizer_scheduler_rng_equal') is True and
             rank.get('probe_model_optimizer_private_rng_equal') is True and
             rank.get('probe_actual_adam_steps')==2000 and rank.get('readback_count')==2,
             'RECOVERY_ACTUAL_TWICE_MODEL_ADAM_SCHEDULER_RNG_READBACK_MISSING')
    need(value.get('selector_rng_equal') is True and value.get('driver_rng_equal') is True and
         value.get('eligible_mapping_equal') is True and value.get('candidate_pending_cursor_equal') is True,
         'RECOVERY_SELECTOR_DRIVER_MAPPING_OR_CURSOR_MISSING')
    return dict(accepted=True,receipt=reads.files[str(path)],status='CONSUMED_ACTUAL_NATIVE4_RESTORE_RECEIPT',
                physical_execution='Authenticated by separate actual restore producer; this CPU consumer allocates no GPU')


def audit(base,commit,run_id,recovery_receipt=None,recovery_sha=None):
    need('torch' not in sys.modules, 'CPU_CONSUMER_PROCESS_ALREADY_IMPORTED_TORCH')
    reads=Reads(base);run,updates=early_history(reads,commit,run_id)
    reads._updates=updates;reads._update_by_id={r['attempt_id']:r for r in updates}
    reads._candidate_cycles={}
    source=source_and_config(reads,run);attempts,history=actual_attempts(reads,run,updates)
    eligible,frames,manifest,tokenizer=canonical_inputs(reads,run)
    outer_rows=[]
    for attempt in attempts:
        outer_rows.append(audit_outer(reads,run,attempt,attempt['complete_outer'],eligible,frames['train'],tokenizer))
    cycles=reads._candidate_cycles
    need(sorted(cycles)==list(range(len(cycles))) and
         all(len(cycles[c])==len(eligible) for c in sorted(cycles)[:-1]), 'ACTUAL_ELIGIBLE_CYCLE_OMITTED_QUESTIONS')
    history['candidate_questions_consumed']=sum(len(v) for v in cycles.values())
    need(history['candidate_questions_consumed']==5120*len(attempts), 'ACTUAL_CANDIDATE_TOTAL_CONSUMPTION_DIFFERS')
    evaluations=validations(reads,run,frames,manifest);saved=checkpoints(reads,run)
    restored=recovery(reads,run,recovery_receipt,recovery_sha)
    need('torch' not in sys.modules, 'CPU_CONSUMER_IMPORTED_TORCH')
    return dict(schema_version=1,kind='prefix20_joint4_independent_cpu_recorded_evidence_audit',
        status='RECORDED_EVIDENCE_PASS',accepted=True,source_base=str(reads.base),source_commit=commit,run_id=run_id,
        checked_unix=time.time(),consumer_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('audit_prefix20.py','safe_pt.py')},source_and_config=source,actual_history=history,outers=outer_rows,
        saved_checkpoints=saved,evaluations=evaluations,actual_final200_recovery=restored,
        goal_complete=False,torch_imported=False,model_reloads=0,gpu_allocations=0,training_updates=0,
        files=list(reads.files.values()),coverage=dict(
            raw_recorded='Top512 exact acquisition; all25600 authentic prefix tokens; same2560 continuation native token/logprob IDs; canonical strict EM; actual four rank Adam2000; ordered native51713',
            bound_execution='BCE mean/global256/local64/onepass and actor-before-probe are bound to immutable executed source/config, raw own2560 transaction, completed actor attempt and actual four rank10-step receipts',
            virtual_zero='Strict-EM all-correct predicate independently recomputed; actual enabled code path and ddof1/epsilon1e-6 source binding checked; no per-token advantage tensor persisted',
            checkpoint='Fresh fullSHA and safe ZIP scalar/storage metadata; saved finite-all-tensors receipt consumed, not recomputed by model load'),
        limitations=['This reader does not authenticate GPU probabilities from JSON or independently re-execute BCE gradients.',
            'No retained actor advantage tensors exist to independently compare every applied virtual-zero normalized value.',
            'Archive transport and separate actual native4 final200 restore are distinct final goal gates.',
            'Final200 restore receipt schema requires integration with the independent restore producer; unavailable receipt stays PENDING.'])


def write_once(path,value):
    path=Path(path);need(path.is_absolute(), 'EXPLICIT_ABSOLUTE_AUDIT_OUTPUT_REQUIRED')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser()
    for name in ('base','commit','run-id','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--recovery-receipt');parser.add_argument('--recovery-sha256');args=parser.parse_args()
    try:
        result=audit(args.base,args.commit,args.run_id,args.recovery_receipt,args.recovery_sha256)
    except Exception as error:
        result=dict(schema_version=1,kind='prefix20_joint4_independent_cpu_recorded_evidence_audit',
            status='REJECTED',accepted=False,goal_complete=False,source_base=args.base,source_commit=args.commit,
            run_id=args.run_id,checked_unix=time.time(),error_type=type(error).__name__,reason=str(error),
            torch_imported='torch' in sys.modules,model_reloads=0,gpu_allocations=0,training_updates=0)
        write_once(args.output,result);print(json.dumps(result,sort_keys=True));return 1
    write_once(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','accepted','goal_complete','source_base','source_commit','run_id')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

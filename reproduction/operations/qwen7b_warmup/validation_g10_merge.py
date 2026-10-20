"""Remote CPU acceptance of two precommitted authentic validation G5 events.

All eight completed generic shards are required BEFORE an output tree is made.
The immutable plan and label-blind fixedG5 IDs are frozen before any own EM is
read. Real A5+B5 labels are then joined by their predeclared IDs; no EM selects
five responses. This is viewed recipe-selection validation, never untouched
final or fitting data. Existing complete banks are freshly verified, not
rewritten. This module samples no tokens and loads no model/tensor.
"""
import argparse
import ast
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from .g10_grouping import fixed_g5
from .replay_probe_layers import normalize_prefix

ROLE = 'RECIPE_SELECTION_VALIDATION_TWO_FRESH_G5_NOT_UNTOUCHED_FINAL'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
STATUS = 'VALIDATION_ACTUAL_TWO_FRESH_G5_G10_SHA_OWN_EM_VERIFIED'
PRODUCER_COMMIT = 'c8b02d3b22b6a3267481e7c632b2f7113b018e93'
MODEL_ACTOR = 'official7b-'+REVISION
GEOMETRY = (26, 3584, 273, 20)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def proof(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def within(path,base):
    path,base=Path(path),Path(base)
    if path.resolve()!=path or not str(path).startswith(str(base)+'/'):
        raise ValueError('Artifact escapes its own immutable event: '+str(path))
    return path


def checked(item,base=None):
    path=Path(item['path'])
    if base is not None:within(path,base)
    actual=proof(path)
    if actual['bytes']!=item['bytes'] or actual['sha256']!=item['sha256']:
        raise ValueError('Source SHA/size changed: '+str(path))
    return path


def immutable(path,value):
    path=Path(path);raw=canonical(value)+b'\n'
    if path.exists():
        if path.read_bytes()!=raw:
            raise ValueError('Conflicting immutable output; preserve it: '+str(path))
        return
    # Exclusive final creation follows an fsynced temporary file. The merge
    # lock prevents another owned writer from publishing the same destination.
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(str(temporary),str(path))


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf8') as stream:
        for line in stream:yield json.loads(line)


def indexed(path):
    result={}
    for row in rows(path):
        tid=row['trajectory_id']
        if tid in result:raise ValueError('Repeated own trajectory: '+str(path))
        result[tid]=row
    return result


def pure_seed_function(config):
    """Use the SHA-bound producer's actual pure functions, without importing it.

    Only the canonical()/seed_for() AST definitions are compiled; no producer
    imports, CUDA initialization, module body or request generation executes.
    """
    root=Path(config['code_root'])
    paths=('reproduction/prefix20_online/pool.py','reproduction/prefix_selection200/rollout.py')
    definitions=[]
    for relative,wanted in zip(paths,('canonical','seed_for')):
        expected=config['collector_code_files_sha256'][relative]
        path=root/relative
        if proof(path)['sha256']!=expected:raise ValueError('Producer seed source changed')
        matches=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name==wanted]
        if len(matches)!=1:raise ValueError('Missing unique producer seed function')
        definitions.extend(matches)
    scope=dict(hashlib=hashlib,json=json)
    module=ast.Module(body=definitions,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<SHA-bound-pure-producer-seed>','exec'),scope)
    return scope['seed_for']


def frozen_plan(base,expected_questions):
    material=read(base/'materialization.json')
    if (material['status']!='RECIPE_SELECTION_VALIDATION_TWO_FRESH_G5_PLANS_PREPARED_NOT_GENERATED' or
            material['untouched_final'] is not False or material['tuning_allowed'] is not True or
            material['fitting_on_validation_allowed'] is not False):
        raise ValueError('Wrong validation materialization role')
    if material['questions']!=expected_questions or material['planned_real_trajectories']!=10*expected_questions:
        raise ValueError('Materialized total question/trajectory budget differs')
    plan_path=checked(material['fixed_G5_plan'],base)
    if plan_path!=base/'planned-g10-and-fixed-g5.json':raise ValueError('Wrong precommitted plan')
    plan=read(plan_path)
    if (plan['role']!=ROLE or plan['untouched_final'] is not False or
            plan['recipe_selection_allowed'] is not True or plan['fitting_on_validation_allowed'] is not False or
            plan['labels_observed'] is not False or plan['actual_responses_generated']!=0):
        raise ValueError('Plan was not explicitly frozen without labels')
    groups=plan['planned_G10'];qids=[group['qid'] for group in groups]
    if len(qids)!=expected_questions or len(set(qids))!=expected_questions:
        raise ValueError('Predeclared question pool count/identity differs')
    if len(set(plan['run_ids']))!=2 or len(set(plan['sampling_attempt_ids']))!=2:
        raise ValueError('Two distinct actual event seed namespaces required')
    tids=set()
    for group in groups:
        if 'labels' in group or 'strict_em' in group:raise ValueError('EM entered the pre-generation plan')
        own=group['trajectory_ids'];events=group['event_trajectory_ids']
        if len(events)!=2 or any(len(x)!=5 for x in events) or own!=events[0]+events[1] or len(set(own))!=10 or tids.intersection(own):
            raise ValueError('Precommitted A5+B5 IDs are missing, reused or reordered')
        tids.update(own)
    fixed=plan['fixed_G5']
    expected=fixed_g5(groups,selection='random',seed=plan['fixed_seed'],epoch=0,role='validation')
    if fixed!=expected or fixed['summary']['labels_consulted'] is not False:
        raise ValueError('Fixed validation G5 differs from label-blind precommit')
    question_file=read(base/'inputs/questions.json');question_rows=question_file['rows']
    if [row['qid'] for row in question_rows]!=qids:
        raise ValueError('Question input order differs from frozen G10 IDs')
    for group,row in zip(groups,question_rows):
        if (group['question_hash']!=row['question_hash'] or
                group['canonical_question_sha256']!=row['canonical_question_sha256']):
            raise ValueError('Frozen question identity/content binding differs')
    freeze=dict(schema_version=1,role=ROLE,source_plan=proof(plan_path),
        source_question_inputs=proof(base/'inputs/questions.json'),ordered_qids=qids,
        planned_G10=groups,fixed_G5=fixed,labels_consulted=False,
        precommitted_plan_SHA_bound_in_both_generating_configs=True)
    return material,plan,question_rows,freeze


def check_config(base,config,index,plan,plan_file):
    event=base/('event-A' if index==0 else 'event-B')
    required=dict(validation_role=ROLE,model_revision=REVISION,evaluation_only=True,
        fitting_authorized=False,actor_updates_authorized=False,actor_optimizer_updates=0,
        untouched_final=False,tuning_allowed=True,capture_prefix_tokens=20,question_max_tokens=273,
        feature_layer=26,fixed_view_precommitted=True,same_generation_protocol_as_training=True,
        warmup_only=True,fit_epochs=0,source_commit=PRODUCER_COMMIT)
    if any(config.get(k)!=v for k,v in required.items()) or config.get('confirmation_role') is not None:
        raise ValueError('Event is not the explicit frozen validation collector')
    if (config['base']!=str(event) or config['event_index']!=index or
            config['run_id']!=plan['run_ids'][index] or config['sampling_attempt_id']!=plan['sampling_attempt_ids'][index]):
        raise ValueError('Event/run/attempt identity differs from the precommit')
    if not any(item==plan_file for item in config['immutable_preparation_sources']):
        raise ValueError('Generator did not SHA-bind this exact precommitted plan')
    for item in config['immutable_preparation_sources']:checked(item)
    for relative,digest in config['collector_code_files_sha256'].items():
        if proof(Path(config['code_root'])/relative)['sha256']!=digest:
            raise ValueError('Frozen producer code changed: '+relative)
    checked(config['actor_complete'])
    actor=read(config['actor_complete']['path'])
    if actor['status']!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' or actor['revision']!=REVISION:
        raise ValueError('Wrong actual actor migration receipt')
    return event,pure_seed_function(config)


def require_all_complete(base,expected_questions):
    """No bank directory, freeze or label read before all eight real receipts."""
    for event in ('event-A','event-B'):
        for shard in range(4):
            path=base/event/'sampling'/('shard-%02d'%shard)/'complete.json'
            if not path.is_file():raise ValueError('Incomplete actual event/shard: '+str(path))
            receipt=read(path)
            if (receipt['status']!='ALL_ASSIGNED_CANONICAL_G5_COMPLETE' or
                    receipt['shard']!=shard or receipt['questions']!=len(range(shard,expected_questions,4)) or
                    receipt['trajectories']!=5*receipt['questions']):
                raise ValueError('Generic event/shard completion count differs')


def native_metadata(descriptor,prefixes,event):
    path=checked(descriptor,event);manifest=read(path)
    if (manifest['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or manifest['count']!=len(prefixes) or
            tuple(manifest[k] for k in ('feature_layer','dimension','question_max_tokens','prefix_tokens'))!=GEOMETRY):
        raise ValueError('Not completed native Layer26 Prefix20 features')
    names={'question.bin','delta.bin','policy.bin','index.jsonl','prototype.bin'}
    if set(manifest['files'])!=names:raise ValueError('Native packed layout file set differs')
    for name,item in manifest['files'].items():
        if Path(item['path'])!=path.parent/name:raise ValueError('Native packed file escaped its own root')
        checked(item,event)
    index=indexed(path.parent/'index.jsonl')
    if set(index)!=set(prefixes):raise ValueError('Native index has missing/extra own trajectories')
    for tid,row in index.items():
        prefix=prefixes[tid]
        if (row['qid']!=str(prefix['qid']) or row['dimension']!=3584 or
                row['question_length']!=len(prefix['initial_prompt_token_ids']) or
                row['policy_length']!=len(prefix['policy_token_ids']) or row['policy_positions']!=prefix['policy_position_ids']):
            raise ValueError('Native question/policy index differs from own saved slots')
        if (prefix['feature_file']!=str(path) or prefix['feature_layer']!=26 or
                prefix['feature_dtype']!='bfloat16_bits_uint16' or prefix['future_information_included'] is not False):
            raise ValueError('Prefix/native feature locator or semantic dtype differs')
    return descriptor,dict(manifest=descriptor,payload_files=manifest['files'],index_rows=len(index))


def event_collect(base,config,index,plan,question_rows,seed_for):
    event=base/('event-A' if index==0 else 'event-B')
    plan_by_qid={group['qid']:group for group in plan['planned_G10']}
    question_by_qid={row['qid']:row for row in question_rows}
    manifest_path=event/'inputs/manifest.json';manifest=read(manifest_path)
    entries={item['path']:item for item in manifest['batches']}
    if len(entries)!=len(manifest['batches']):raise ValueError('Input batch descriptor repeats')
    labels,features,audits,seeds_by_qid={},{},[],{}
    contract=None;seen_questions=set()
    for shard in range(4):
        root=event/'sampling'/('shard-%02d'%shard);generic=read(root/'complete.json')
        if generic['run_id']!=config['run_id'] or generic['model_revision']!=REVISION:
            raise ValueError('Generic completion belongs to a different actor/event')
        own_inputs=sorted((event/'inputs'/('shard-%02d'%shard)).glob('batch-*.json'))
        expected_receipts={str(root/'batches'/p.stem/'complete.json') for p in own_inputs}
        if set(generic['batches'])!=expected_receipts or len(generic['batches'])!=len(expected_receipts):
            raise ValueError('Completed batches differ from every declared input')
        shard_qids=set()
        for input_path in own_inputs:
            checked(entries[str(input_path)],event)
            inputs=read(input_path)['rows'];qids={row['qid'] for row in inputs}
            if len(qids)!=len(inputs) or seen_questions.intersection(qids):raise ValueError('Question repeated between batches/shards')
            shard_qids.update(qids);seen_questions.update(qids)
            for row in inputs:
                if row!=question_by_qid.get(row['qid']):raise ValueError('Input differs from the frozen original question')
            dest=root/'batches'/input_path.stem;batch_path=dest/'complete.json';batch=read(batch_path)
            if (batch['run_id']!=config['run_id'] or batch['input_sha256']!=proof(input_path)['sha256'] or
                    batch['questions']!=len(inputs) or batch['trajectories']!=5*len(inputs) or batch['actor_optimizer_updates']!=0):
                raise ValueError('Genuine batch input/count/actor-update receipt differs')
            prefix,full=batch['prefix'],batch['full'];paths={}
            for phase,keys in ((prefix,('states','prefix','actor_hidden')),(full,('full','own_em'))):
                if (phase['run_id']!=config['run_id'] or phase['actor_version']!=MODEL_ACTOR or
                        phase['sampling_attempt_id']!=config['sampling_attempt_id'] or phase['questions']!=len(inputs) or
                        phase['trajectories']!=5*len(inputs)):
                    raise ValueError('Phase belongs to the wrong actor/event/seed namespace')
                current=tuple(phase[k] for k in ('sampling_contract','action_boundary','interaction_protocol','actual_compute_dtype'))
                if contract is None:contract=current
                if current!=contract:raise ValueError('Producer sampling/interaction/dtype protocol differs')
                for key in keys:paths[key]=checked(phase[key],event)
            outer=dest/'outers/outer-000001'
            prefix_complete=read(outer/'prefix/complete.json');full_complete=read(outer/'full/complete.json')
            if (prefix_complete['ranks']!=[prefix] or prefix_complete['scan_sha256']!=proof(outer/'prefix/scan.json')['sha256'] or
                    any(full_complete[k]!=v for k,v in full.items()) or full_complete['labels']!=5*len(inputs) or
                    full_complete['actor_optimizer_updates']!=0):
                raise ValueError('Generic phase completion differs from its sealed batch')
            prefixes=indexed(paths['prefix']);fulls=indexed(paths['full']);ems=indexed(paths['own_em']);states=indexed(paths['states'])
            expected={tid for qid in qids for tid in plan_by_qid[qid]['event_trajectory_ids'][index]}
            if not (set(prefixes)==set(fulls)==set(ems)==set(states)==expected):
                raise ValueError('Missing/extra prefix/full/state/EM trajectory or wrong precommitted ID')
            for tid,pre in prefixes.items():
                qid=pre['qid'];f=fulls[tid];em=ems[tid];state=states[tid];row=question_by_qid[qid]
                normalize_prefix(pre,expected_tid=tid,expected_qid=qid,require_digest=True)
                response=pre['response_index']
                if (type(response) is not int or not 0<=response<5 or
                        plan_by_qid[qid]['event_trajectory_ids'][index][response]!=tid or f['response_index']!=response):
                    raise ValueError('Event-local response index no longer matches its frozen ID')
                if (pre['run_id']!=config['run_id'] or f['run_id']!=config['run_id'] or f['policy_snapshot_id']!=MODEL_ACTOR or
                        pre['split']!='recipe_selection_validation' or f['split']!='recipe_selection_validation' or f['qid']!=qid or em['qid']!=qid or
                        pre['question_hash']!=row['question_hash'] or f['question_hash']!=row['question_hash'] or
                        pre['initial_prompt_token_ids']!=row['initial_prompt_token_ids'] or f['initial_prompt_token_ids']!=row['initial_prompt_token_ids']):
                    raise ValueError('Own question/prompt/actor/validation binding differs')
                if (pre['prefix_sha256']!=f['prefix_sha256'] or pre['prefix_sha256']!=em['prefix_sha256'] or
                        em['prefix_scan_sha256']!=f['prefix_scan_sha256'] or f['prefix_scan_sha256']!=prefix_complete['scan_sha256'] or
                        state['prefix_sha256']!=pre['prefix_sha256'] or state['sampling_attempt_id']!=config['sampling_attempt_id']):
                    raise ValueError('Full response or own EM is not a continuation of its own retained prefix')
                y=f['strict_em']
                if type(y) is not int or y not in (0,1) or type(em['strict_em']) is not int or em['strict_em']!=y:
                    raise ValueError('Actual own strict EM labels disagree or are not binary integers')
                if len(f['response_token_ids'])!=len(f['response_info_mask']) or not f['behavior_turns'] or not f['sampling_journal']:
                    raise ValueError('Full response lacks actual generation/retrieval journal')
                for chunk in f['sampling_journal']:
                    expected_seed=seed_for(config['run_id'],1,tid,chunk['logical_turn_index'],chunk['native_chunk_index'],config['seed'],sampling_attempt_id=config['sampling_attempt_id'])
                    if chunk['seed']!=expected_seed or chunk['sampling']['seed']!=expected_seed:
                        raise ValueError('Actual native seed differs from its independent event derivation')
                    seeds_by_qid.setdefault(qid,set()).add(expected_seed)
                if tid in labels:raise ValueError('Response repeats between completed batches')
                labels[tid]=y
            descriptor,feature_audit=native_metadata(prefix['actor_hidden'],prefixes,event)
            features[descriptor['path']]=descriptor
            audits.append(dict(batch=proof(batch_path),input=entries[str(input_path)],
                phase_payloads={key:prefix[key] if key in prefix else full[key] for key in ('states','prefix','actor_hidden','full','own_em')},
                prefix_completion=proof(outer/'prefix/complete.json'),full_completion=proof(outer/'full/complete.json'),native=feature_audit))
        if shard_qids!={row['qid'] for row in question_rows[shard::4]}:
            raise ValueError('Shard question partition differs from the original order')
    if seen_questions!=set(question_by_qid) or len(labels)!=5*len(question_rows):
        raise ValueError('Event did not actually complete every frozen question')
    return dict(labels=labels,features=list(features.values()),audit=audits,seeds=seeds_by_qid,contract=contract)


def merge(base,synthetic_fixture=False,expected_questions=2048):
    base=Path(base).resolve()
    if not synthetic_fixture and (not str(base).startswith('/data/user/') or expected_questions!=2048):
        raise ValueError('Actual validation artifacts must remain remote with exactly2048 questions')
    if expected_questions<4:raise ValueError('Four declared shards require at least four questions')
    require_all_complete(base,expected_questions)
    material,plan,question_rows,freeze=frozen_plan(base,expected_questions)
    configs=[];seed_functions=[]
    for index,descriptor in enumerate(material['event_configs']):
        path=checked(descriptor,base);config=read(path)
        if path!=base/('event-A' if index==0 else 'event-B')/'config.json':raise ValueError('Wrong config descriptor order')
        _,seed_for=check_config(base,config,index,plan,freeze['source_plan'])
        configs.append(config);seed_functions.append(seed_for)
    if len(configs)!=2:raise ValueError('Exactly two actual generating configs required')
    bank=base/'bank';bank.mkdir(exist_ok=True)
    with (bank/'merge.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(bank/'plan-freeze.json',freeze)
        # This is the first own-EM read: exact plan and fixedG5 were already
        # verified, copied and fsynced without ever observing these labels.
        events=[event_collect(base,c,i,plan,question_rows,seed_functions[i]) for i,c in enumerate(configs)]
        if events[0]['contract']!=events[1]['contract']:raise ValueError('Two events used different generation protocols')
        for qid in freeze['ordered_qids']:
            if events[0]['seeds'][qid]&events[1]['seeds'][qid]:
                raise ValueError('Same-question native seed collision across the two fresh events')
        g10=[];labels={}
        for event in events:
            if labels.keys()&event['labels'].keys():raise ValueError('A/B trajectory IDs overlap')
            labels.update(event['labels'])
        for group in plan['planned_G10']:
            own=dict(group);own['labels']=[labels[tid] for tid in group['trajectory_ids']]
            own.update(generation_run_ids=plan['run_ids'],sampling_attempt_ids=plan['sampling_attempt_ids'],
                role='validation',untouched_final=False)
            g10.append(own)
        fixed=[]
        for group in freeze['fixed_G5']['groups']:
            fixed.append(dict(group,labels=[labels[tid] for tid in group['trajectory_ids']]))
        feature_refs={name:event['features'] for name,event in zip(('event-A','event-B'),events)}
        # The production runner's native_reader consumes list[str], while the
        # independent per-event SHA bindings remain available in a separate
        # audit. Do not hand a mapping or descriptor list to that reader.
        feature_paths=list(dict.fromkeys(item['path'] for own in feature_refs.values() for item in own))
        payloads={'groups.json':dict(validation=g10),
            'features.json':feature_paths,
            'feature-sources-by-event.json':feature_refs,
            'fixed-G5-selection.json':freeze['fixed_G5'],
            'fixed-G5.json':dict(validation=fixed,precommitted_plan=freeze['source_plan'],
                fixed_G5_summary=freeze['fixed_G5']['summary'],selection_used_EM=False),
            'source-audit.json':dict(event_configs=material['event_configs'],
                events={name:event['audit'] for name,event in zip(('event-A','event-B'),events)},
                all_source_payload_SHA_freshly_verified=True,model_or_tensor_loaded=False,
                retained_features_losslessly_reconstructed=False)}
        completion=bank/'complete.json'
        if completion.exists():
            accepted=read(completion)
            if (accepted['status']!=STATUS or accepted['untouched_final'] is not False or
                    accepted['questions']!=expected_questions or accepted['trajectories']!=10*expected_questions or
                    accepted['role']!='validation' or accepted['source_commit']!=PRODUCER_COMMIT or
                    accepted['model_revision']!=REVISION or accepted['actor_optimizer_updates']!=0 or
                    accepted['synthetic_EM'] is not False or accepted['fixed_G5_was_chosen_before_own_EM'] is not True or
                    accepted['fitting_on_validation_allowed'] is not False or accepted['tuning_allowed'] is not True or
                    set(accepted['outputs'])!=set(payloads)):
                raise ValueError('Existing bank has wrong acceptance/validation role')
            for name,value in payloads.items():
                checked(accepted['outputs'][name],bank)
                if read(bank/name)!=value:raise ValueError('Existing accepted bank differs from fresh actual source join: '+name)
            checked(accepted['plan_freeze'],bank)
            expected_consumer=dict(groups=proof(bank/'groups.json'),group_split='validation',
                features=proof(bank/'features.json'),features_format='flat_list_of_native_complete_paths',
                fixed_G5_plan=proof(bank/'fixed-G5-selection.json'),selection_labels_consulted=False)
            if accepted['consumer_inputs']!=expected_consumer:
                raise ValueError('Existing accepted runner inputs differ from frozen consumer files')
            return dict(accepted,existing_complete_freshly_verified=True,complete=proof(completion))
        for name,value in payloads.items():immutable(bank/name,value)
        histogram=Counter(sum(group['labels']) for group in g10)
        accepted=dict(status=STATUS,role='validation',untouched_final=False,tuning_allowed=True,
            fitting_on_validation_allowed=False,questions=len(g10),trajectories=10*len(g10),
            real_fresh_events=2,real_trajectories_per_event=5*len(g10),actor_optimizer_updates=0,
            model_revision=REVISION,source_commit=configs[0]['source_commit'],
            plan_freeze=proof(bank/'plan-freeze.json'),outputs={name:proof(bank/name) for name in payloads},
            consumer_inputs=dict(groups=proof(bank/'groups.json'),group_split='validation',
                features=proof(bank/'features.json'),features_format='flat_list_of_native_complete_paths',
                fixed_G5_plan=proof(bank/'fixed-G5-selection.json'),selection_labels_consulted=False),
            feature_source_SHA_audit=proof(bank/'feature-sources-by-event.json'),
            fixed_G5_EM_evaluation_audit=proof(bank/'fixed-G5.json'),
            fixed_G5_was_chosen_before_own_EM=True,synthetic_EM=False,
            K10_histogram=[histogram[k] for k in range(11)],
            native_feature_manifests={name:len(event['features']) for name,event in zip(('event-A','event-B'),events)},
            merger_code=proof(__file__),completed_epoch_time=time.time())
        immutable(completion,accepted)
        return dict(accepted,existing_complete_freshly_verified=False,complete=proof(completion))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    result=merge(args.base)
    # Only small manifests/counts leave this process, never groups or raw data.
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

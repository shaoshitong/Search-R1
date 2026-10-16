"""Independently verify real child calibration before fresh production admission.

No threshold is chosen here. Explicit frozen health criteria are required.
Children keep their real run/job identities; their writes never become parent
writes. No model, optimizer, process or service is changed by this module.
"""
import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from reproduction.common8.policy_calibration_plan import (
    DISCOVERY_ROLE, CONFIRMATION_ROLE, canonical_bytes, document_sha256,
    load_plan, _identity, _strict_json,
)
from reproduction.common8.verify_data import digest


class AdmissionError(ValueError):
    pass


def _need(condition, message):
    if not condition: raise AdmissionError(message)


def _same(value, expected, message):
    _need(canonical_bytes(value)==canonical_bytes(expected),message)


def _number(value, name):
    _need(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid health criterion: '+name)
    return value


def evaluate_health(rule, discovery, confirmation):
    """Apply caller-frozen absolute criteria to both paired fixed512 measurements."""
    _need(set(rule)=={'document','sha256'} and document_sha256(rule['document'])==rule['sha256'],
          'Health rule bytes/hash differ')
    doc=rule['document'];criteria=doc['criteria']
    _need(doc['role']=='policy_safety_health_rule' and isinstance(doc['provenance'],dict) and bool(doc['provenance']),
          'Missing explicit frozen health provenance')
    _need(set(criteria)<= {'method','allowed_drop','minimum_before','minimum_after'} and
          criteria['method']=='paired_fixed512_absolute_allowed_drop','Unsupported explicit health rule')
    drops=criteria['allowed_drop']
    required={'weighted_em','weighted_f1','valid_action_rate'}
    supported=required|{'executed_search_question_rate','executed_search_calls_per_question'}
    _need(isinstance(drops,dict) and required<=set(drops)<=supported,'EM/F1/valid-action criteria are mandatory')
    for name,value in drops.items(): _number(value,name)
    for field in ('minimum_before','minimum_after'):
        values=criteria.get(field,{})
        _need(isinstance(values,dict) and set(values)<=supported,'Unknown minimum criterion')
        for name,value in values.items(): _number(value,field+'/'+name)
    decisions={}
    for label,segment in (('discovery',discovery),('confirmation',confirmation)):
        before,after=segment['evaluation_before'],segment['evaluation_after']
        _need(before['question_ids']==after['question_ids'] and before['data_sources']==after['data_sources'] and
              before['question_count']==after['question_count']==512,'Health measurements are not paired fixed512')
        observed={}
        for name,limit in drops.items():
            a,z=_number(before[name],name),_number(after[name],name)
            _need(a-z<=limit,'Frozen health allowed drop exceeded: '+label+'/'+name)
            observed[name]=dict(before=a,after=z,allowed_drop=limit,observed_drop=a-z)
        for field,measurement in (('minimum_before',before),('minimum_after',after)):
            for name,minimum in criteria.get(field,{}).items():
                _need(_number(measurement[name],name)>=minimum,'Frozen health minimum failed: '+label+'/'+field+'/'+name)
        decisions[label]=observed
    return dict(rule_sha256=rule['sha256'],method=criteria['method'],accepted=True,
                decisions=decisions,action_definition=discovery['evaluation_before']['valid_action_rate_definition'])


def _read(base, relative, receipts=None, prefix=''):
    relative=Path(relative)
    _need(not relative.is_absolute() and '..' not in relative.parts,'Unbound receipt path')
    path=Path(base)/relative
    _need(not path.is_symlink(),'Receipt must not be a symlink')
    content=path.read_bytes()
    if receipts is not None: receipts[prefix+relative.as_posix()]=content
    return _strict_json(content)


def _smoke(base,run,plan,allocation,receipts=None,prefix=''):
    from reproduction.common8.model_smoke import aggregate
    from reproduction.common8.validation_identity import expected
    _,_,ids=expected(base,'stage')
    for phase in ('initial','coexistence'):
        name='logs/model-smoke-'+phase+'/complete.json';proof=_read(base,name,receipts,prefix)
        _need(proof['accepted'] is True and proof['run_id']==run['run_id'] and proof['slurm_job_id']==plan['slurm_job_id'] and
              proof['phase']==phase and type(proof['actual_grpo_updates']) is int and proof['actual_grpo_updates']==0,
              'Not the actual fresh initial/coexistence model gate')
        ranks=[]
        for rank in range(8):
            relative='logs/model-smoke-'+phase+f'/rank-{rank}.json'
            row=_read(base,relative,receipts,prefix)
            _need(digest(base/relative)==proof['rank_file_sha256'][str(rank)] and
                  type(row['rank']) is int and row['rank']==rank and row['initial_parameters_finite'] is True and
                  type(row['actual_optimizer_writes']) is int and row['actual_optimizer_writes']==0,
                  'Initial actual rank finite/fresh proof differs')
            ranks.append(row)
        _need(ranks==proof['ranks'] and aggregate(ranks,run['run_id'],plan['slurm_job_id'],phase,ids,
              allocation['actual_cuda_runtime_uuids'])==plan['resolved_config_sha256'],'Actual rank probability/config proof differs')
        if phase=='initial':
            _need(digest(base/name)==plan['initial_probability_gate']['proof_sha256'],'Frozen initial probability bytes differ')


def _paired_evaluations(base,run,plan,sha,segment,receipts,prefix):
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    _,frame,ids=expected(base,'validation')
    for label,step in (('before',0),('after',1)):
        relative='logs/policy-calibration-evaluation-'+label+'.json';row=_read(base,relative,receipts,prefix)
        _need(digest(base/relative)==segment['evaluation_sha256'][label] and row==segment['evaluation_'+label],
              'Actual paired evaluation bytes differ')
        _need(row['run_id']==run['run_id'] and row['parent_production_run_id']==plan['parent_production_run_id'] and
              row['slurm_job_id']==plan['slurm_job_id'] and row['plan_sha256']==sha and type(row['step']) is int and
              row['step']==step and row['phase_role']==plan['role'] and
              row['validation_sha256']==run['input_sha256']['validation.parquet'],'Evaluation identity differs')
        calculated=aggregate(row['rows'],ids,frame['data_source'].tolist())
        for key,value in calculated.items(): _same(row[key],value,'Evaluation arithmetic/order differs: '+key)


def _training_inputs(base,run,sha,segment,receipts,prefix):
    documents={}
    for name,suffix in (('parents','inputs'),('groups','groups')):
        relative='logs/policy-calibration-training-'+suffix+'.json';row=_read(base,relative,receipts,prefix)
        _need(digest(base/relative)==segment['training_input_proof_sha256'][name] and
              row==segment['actual_training_'+suffix] and row['run_id']==run['run_id'] and row['plan_sha256']==sha and
              type(row['actual_trajectory_count']) is int and row['actual_trajectory_count']==2560 and
              row['production_admission'] is False,'Actual training input/group proof differs')
        documents[name]=row
    parents,groups=documents['parents'],documents['groups']
    _need(type(parents['actual_parent_count']) is int and parents['actual_parent_count']==512 and
          type(parents['n_agent']) is int and parents['n_agent']==5 and len(parents['parents'])==512 and
          parents['train_sha256']==run['input_sha256']['train.parquet'],'Not a complete canonical512 G5 outer')
    for row in parents['parents']:
        _need(row['data_source'] in ('nq','hotpotqa') and set(row['tensor_sha256'])=={'input_ids','attention_mask','position_ids'},
              'Actual training parent identity missing')
        for value in list(row['tensor_sha256'].values())+[row['upstream_group_sha256']]:
            _need(isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value),'Invalid parent tensor/group SHA')
    expected=[row['upstream_group_sha256'] for row in parents['parents'] for _ in range(5)]
    _need(parents['repeated_group_sha256']==expected and groups['upstream_group_multiset_unchanged'] is True and
          Counter(groups['balanced_group_sha256'])==Counter(expected),'Original GRPO grouping changed')


def _cleanup(base,run,receipts,prefix):
    from reproduction.a800.owned_processes import same,identity
    status=_read(base,'logs/pipeline-status.json',receipts,prefix)
    cleanup=_read(base,'logs/cleanup-status.json',receipts,prefix)
    ledger=_read(base,'logs/process-ledger.json',receipts,prefix)
    _need(status['run_id']==cleanup['run_id']==ledger['run_id']==run['run_id'] and cleanup['cleaned'] is True and
          status['stage']=='FINISHED_POLICY_CALIBRATION_MEASUREMENTS_AND_OWNED_CLEANUP','Child did not finish owned cleanup')
    _need(isinstance(ledger['jobs'],dict) and {'trainer','retriever','probe-server'}<=set(ledger['jobs']),
          'Missing actual owned component ledger')
    for job in ledger['jobs'].values():
        _need(job['state']=='STOPPED','Owned child component is not stopped')
        for process in job['ledger'].values():
            _need(not same(process) or identity(process['pid'])['state']=='Z','Owned child process remains live')
    for name in ('retrieval-staging-claim.json','corpus-cache-claim.json'):
        claim=_read(base,'logs/'+name,receipts,prefix)
        root=Path('/dev/shm')/('search-g1-common8-'+run['run_id']+'-'+run['policy_calibration_slurm_job_id'])
        if name=='corpus-cache-claim.json': root=root/'datasets-cache'
        _need(claim['run_id']==run['run_id'] and claim['phase']=='CLEANED' and
              claim['slurm_job_id']==run['policy_calibration_slurm_job_id'] and claim['uid']==os.getuid() and
              Path(claim['root'])==root,'Owned temporary data cleanup incomplete')
        _need(not root.exists() and not root.is_symlink(),'Owned temporary root still exists')


def _checkpoint(base,run,segment,receipts,prefix):
    import torch
    from safetensors import safe_open
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    checkpoint=Path(run['checkpoint_dir'])/'actor/global_step_1'
    complete=_read(checkpoint,'complete.json')
    _same(complete,segment['checkpoint'],'Actual fresh1 complete proof differs')
    _need(complete['run_id']==run['run_id'] and type(complete['step']) is int and complete['step']==1 and
          complete['finite_all_saved_tensors'] is True,'Not the finite child fresh1 checkpoint')
    count=validate_saved_checkpoint(checkpoint,Path(run['initial_model']),run['started_unix'])
    _need(count==complete['validated_tensor_count'],'Actual canonical tensor count differs')
    for name,value in complete['files_sha256'].items(): _need(digest(checkpoint/name)==value,'Actual checkpoint bytes changed')
    receipts[prefix+'checkpoint-complete.json']=(checkpoint/'complete.json').read_bytes()
    initial=Path(run['initial_model']);old=_read(initial,'model.safetensors.index.json')['weight_map']
    new=_read(checkpoint,'model.safetensors.index.json')['weight_map'];changed=[]
    for key in old:
        with safe_open(str(initial/old[key]),framework='pt',device='cpu') as file: a=file.get_tensor(key)
        with safe_open(str(checkpoint/new[key]),framework='pt',device='cpu') as file: z=file.get_tensor(key)
        if not torch.equal(a,z): changed.append(key)
        del a,z
    _need(bool(changed) and changed==segment['changed_parameters'],'No actual canonical parameter delta or delta declaration differs')


def verify_child(base,parent,role,receipts,prefix):
    """Independent historical verifier; no live services are restarted."""
    from reproduction.common8.freeze_calibration_plan import _mathematics
    from reproduction.common8.calibration_runtime import verify_optimizer_states
    from reproduction.common8.safety_calibration_evidence import verify_segment
    base=Path(base).resolve();run=_read(base,'logs/run-manifest.json',receipts,prefix)
    _need(run['policy_calibration_role']==role and run['parent_production_run_id']==parent['run_id'],'Wrong child role or parent')
    _same(_identity(run),_identity(parent),'Child source/model/data/math identity differs')
    parent_bytes=(base/'preparation/parent-run-manifest.json').read_bytes()
    original_parent=_strict_json(parent_bytes)
    _need(hashlib.sha256(parent_bytes).hexdigest()==run['policy_calibration_parent_manifest_sha256'] and
          original_parent['run_id']==parent['run_id'],'Child original parent manifest differs')
    _same(_identity(original_parent),_identity(parent),'Original parent scientific identity changed')
    receipts[prefix+'preparation/parent-run-manifest.json']=parent_bytes
    plan,sha=load_plan(base,run,expected_slurm_job_id=run['policy_calibration_slurm_job_id'])
    _read(base,'logs/policy-safety-calibration-plan.json',receipts,prefix)
    for folder,mapping in (('repo',run['source_sha256']),('data',run['input_sha256'])):
        for name,value in mapping.items(): _need(digest(base/folder/name)==value,'Child bound source/data bytes differ')
    for name,value in run['initial_model_sha256'].items():
        _need(digest(Path(run['initial_model'])/name)==value,'Original official model bytes differ')
    config=_read(base,'logs/resolved-training-config.json',receipts,prefix)
    _need(config['run_id']==run['run_id'] and config['slurm_job_id']==plan['slurm_job_id'] and
          document_sha256(config['resolved_config'])==config['resolved_config_sha256']==plan['resolved_config_sha256'],
          'Actual child resolved configuration differs')
    _mathematics(config['resolved_config'],base,run)
    allocation=_read(base,'logs/native-allocation-admission.json',receipts,prefix)
    _need(allocation['accepted'] is True and allocation['run_id']==run['run_id'] and
          allocation['slurm_job_id']==plan['slurm_job_id'] and allocation['actual_job_state']=='RUNNING' and
          allocation['foreign_compute_applications']==[] and allocation['node']=='h100-8' and
          len(set(allocation['actual_cuda_runtime_uuids']))==8 and set(allocation['actual_allocated_indices'])==set(range(8)),
          'Actual native child allocation differs')
    staging=_read(base,'logs/retrieval-staging-status.json',receipts,prefix)
    _need(staging['run_id']==run['run_id'] and staging['slurm_job_id']==plan['slurm_job_id'] and
          staging['resources']['actual_allocation_bytes']>=512*2**30,'Actual child did not have native512GiB')
    _smoke(base,run,plan,allocation,receipts,prefix)
    segment=_read(base,'logs/policy-calibration-segment.json',receipts,prefix)
    _need(segment['role']==role+'_evidence' and segment['origin']=='actual_native_policy_segment' and
          segment['run_id']==run['run_id'] and segment['parent_production_run_id']==parent['run_id'] and
          segment['slurm_job_id']==plan['slurm_job_id'] and segment['plan_sha256']==sha and
          segment['production_admission'] is False and segment['healthy_checkpoint_accepted'] is False,
          'Not an actual child measurements segment')
    for key,value in (('completed_outer_updates',1),('optimizer_writes_per_rank',20)):
        _need(type(segment[key]) is int and segment[key]==value,'Child actual budget incomplete')
    for key in ('identity','initial_reward_fingerprint','resolved_config_sha256'):
        _same(segment[key],plan[key],'Child segment frozen identity differs')
    logs={}
    for rank in range(8):
        name=f'logs/optimizer-rank-{rank}.jsonl';content=(base/name).read_bytes()
        _need(hashlib.sha256(content).hexdigest()==segment['worker_log_sha256'][str(rank)],'Actual worker ledger changed')
        receipts[prefix+name]=content
        logs[rank]=[_strict_json(line) for line in content.splitlines()]
    structural=verify_segment(logs,dict(schema_version=1,role=role,run_id=run['run_id'],calibration_sha256=sha,
                                      outer_count=1,initial_optimizer_writes=0))
    _same(structural,segment['structural'],'Actual monitor/write reconstruction differs')
    states={}
    for rank in range(8):
        relative=f'logs/policy-calibration-optimizer-rank-{rank}.json';states[rank]=_read(base,relative,receipts,prefix)
        _need(digest(base/relative)==segment['optimizer_state_proof_sha256'][str(rank)],'Actual Adam summary changed')
    _same(verify_optimizer_states(states,run,plan,sha,allocation),segment['actual_optimizer_states'],'Actual Adam summaries differ')
    births=segment['actual_worker_births_at_segment_close']
    _need(isinstance(births,list) and len(births)==8,'Missing actual segment-close births')
    for rank,birth in enumerate(births):
        _need(birth['pid']==states[rank]['pid'] and birth['start_ticks']==states[rank]['start_ticks'] and
              type(birth['sid']) is int and birth['sid']>0 and type(birth['uid']) is int and birth['uid']==os.getuid() and
              isinstance(birth['state'],str) and len(birth['state'])==1 and birth['state']!='Z','Actual closing worker birth differs')
    _paired_evaluations(base,run,plan,sha,segment,receipts,prefix)
    _training_inputs(base,run,sha,segment,receipts,prefix)
    updates=(base/'logs/actual-updates.jsonl').read_bytes();receipts[prefix+'logs/actual-updates.jsonl']=updates
    actual=[_strict_json(line) for line in updates.splitlines()]
    _same(actual,segment['actual_updates'],'Actual driver update ledger differs')
    _need(len(actual)==1 and actual[0]['run_id']==run['run_id'] and type(actual[0]['step']) is int and actual[0]['step']==1 and
          actual[0]['probe_version']==run['initial_probe_version'] and actual[0]['metrics']['actor/optimizer_steps_this_update']==20,
          'Not exactly one complete actual update')
    for value in actual[0]['metrics'].values():
        _need(type(value) in (int,float) and math.isfinite(value),'Nonfinite or nonnumeric update metric')
    live=_read(base,'artifacts/live/manifest.json',receipts,prefix)
    accepted=_read(base,'logs/initial-probes-accepted.json',receipts,prefix)
    _need(accepted['accepted'] is True and accepted['run_id']==run['run_id'] and accepted['version']==live['version']==run['initial_probe_version'] and
          accepted['live_manifest_sha256']==digest(base/'artifacts/live/manifest.json') and live['round']==0 and
          live['encoder_sha256']==plan['initial_reward_fingerprint']['encoder_sha256'] and
          live['head_sha256']==plan['initial_reward_fingerprint']['head_sha256'],'Initial admitted reward identity changed')
    for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
        _need(digest(Path(live[key]))==live['head_sha256'][name],'Actual frozen child head bytes changed')
    _checkpoint(base,run,segment,receipts,prefix)
    _cleanup(base,run,receipts,prefix)
    return run,plan,segment


def _receipt(base,relative,expected):
    path=(Path(base)/relative).resolve()
    _need(Path(base).resolve() in path.parents and not (Path(base)/relative).is_symlink(),'Unbound admission receipt')
    content=path.read_bytes()
    _need(hashlib.sha256(content).hexdigest()==expected,'Admission receipt changed: '+relative)
    return _strict_json(content)


def _adoption_lineage(run,job_id,receipt,live,source_run,source_plan,source_segment,
                      source_live,source_hashes,archive_sha256,live_sha256):
    """Recheck the real CPU adoption producer's immutable identity, never HTTP."""
    token=receipt['version'].removeprefix(run['run_id']+':0:')
    _need(receipt['schema_version']==1 and type(receipt['schema_version']) is int and
          receipt['role']=='initial_probe_adoption' and receipt['adopted'] is True and
          receipt['run_id']==run['run_id'] and receipt['slurm_job_id']==job_id and
          receipt['parent_production_run_id']==source_plan['parent_production_run_id'] and
          Path(receipt['discovery_base']).resolve()==Path(run['policy_calibration_discovery_base']).resolve() and
          receipt['production_admission'] is False and receipt['healthy_checkpoint_accepted'] is False and
          receipt['child_http_admission_required'] is True and receipt['live_manifest_sha256']==live_sha256,
          'Actual adoption receipt identity/actual live bytes differ')
    _need(isinstance(token,str) and len(token)==32 and all(ch in '0123456789abcdef' for ch in token) and
          receipt['version']==run['run_id']+':0:'+token==live['version']==run['initial_probe_version'] and
          type(live['round']) is int and live['round']==0 and live['run_id']==run['run_id'],
          'Adopted version is not this new run0 UUID')
    expected=dict(run_id=source_run['run_id'],slurm_job_id=source_plan['slurm_job_id'],version=source_live['version'],
                  archive_sha256=archive_sha256,accepted_manifest_sha256=source_hashes['logs/initial-probes-accepted.json'],
                  plan_sha256=document_sha256(source_plan),segment_sha256=source_hashes['logs/policy-calibration-segment.json'])
    _same(receipt['origin'],expected,'Adoption original discovery lineage differs')
    _same(live['adoption_origin'],expected,'Live adopted origin differs')
    _same(receipt['initial_reward_fingerprint'],source_plan['initial_reward_fingerprint'],'Adoption reward math differs')
    _same(live['head_sha256'],source_live['head_sha256'],'Adoption raw head SHA differs')
    _same(live['encoder_sha256'],source_live['encoder_sha256'],'Adoption encoder SHA differs')
    _need(source_segment['run_id']==source_run['run_id'] and
          source_segment['plan_sha256']==document_sha256(source_plan),'Adoption source segment/plan differs')
    for name,sha in source_hashes.items():
        _need(receipt['source_file_sha256'][name]==sha,'Adoption original source byte SHA differs: '+name)


def _archive_adoption(base,artifact,run,job_id,receipt_relative,live_relative):
    def read(relative):return _receipt(base,relative,artifact['receipts_sha256'][relative])
    prefix='logs/policy-safety-admission-receipts/discovery/'
    source_run=read(prefix+'logs/run-manifest.json');source_plan=read(prefix+'logs/policy-safety-calibration-plan.json')
    source_segment=read(prefix+'logs/policy-calibration-segment.json');source_live=read(prefix+'artifacts/live/manifest.json')
    source_names=('logs/run-manifest.json','logs/policy-safety-calibration-plan.json','logs/policy-calibration-segment.json',
                  'logs/initial-probes-accepted.json','artifacts/live/manifest.json')
    hashes={name:artifact['receipts_sha256'][prefix+name] for name in source_names}
    archive=artifact['discovery_original_version_archive']
    _same(read(archive),source_live,'Archived discovery original version differs')
    hashes[archive.removeprefix(prefix)]=artifact['receipts_sha256'][archive]
    source_base=Path(artifact['children']['discovery']['base'])
    for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
        hashes[Path(source_live[key]).relative_to(source_base).as_posix()]=source_live['head_sha256'][name]
    _adoption_lineage(run,job_id,read(receipt_relative),read(live_relative),source_run,source_plan,source_segment,
                      source_live,hashes,artifact['receipts_sha256'][archive],artifact['receipts_sha256'][live_relative])


def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _archived_segment(base,artifact,descriptor,plan,segment):
    """Reconstruct small immutable execution receipts without reopening models."""
    from reproduction.common8.safety_calibration_evidence import verify_segment
    from reproduction.common8.calibration_runtime import verify_optimizer_states
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.freeze_calibration_plan import _mathematics
    from reproduction.common8.model_smoke import aggregate as model_aggregate
    prefix=descriptor['plan'].removesuffix('logs/policy-safety-calibration-plan.json')
    def read(name):
        relative=prefix+name
        return _receipt(base,relative,artifact['receipts_sha256'][relative])
    run=read('logs/run-manifest.json');allocation=read('logs/native-allocation-admission.json')
    _same(_identity(run),artifact['production_identity'],'Archived child manifest official identity differs')
    _need(run['run_id']==segment['run_id'] and run['parent_production_run_id']==artifact['run_id'] and
          run['policy_calibration_role']==plan['role'] and run['policy_calibration_plan_sha256']==document_sha256(plan),
          'Archived child manifest/plan differs')
    original_parent=read('preparation/parent-run-manifest.json')
    _need(original_parent['run_id']==artifact['run_id'] and
          artifact['receipts_sha256'][prefix+'preparation/parent-run-manifest.json']==run['policy_calibration_parent_manifest_sha256'],
          'Archived original parent bytes differ')
    _same(_identity(original_parent),artifact['production_identity'],'Archived original parent identity differs')
    config=read('logs/resolved-training-config.json')
    _need(document_sha256(config['resolved_config'])==config['resolved_config_sha256']==plan['resolved_config_sha256'],
          'Archived child actual config differs')
    _mathematics(config['resolved_config'],Path(descriptor['base']),run)
    for phase in ('initial','coexistence'):
        proof=read('logs/model-smoke-'+phase+'/complete.json');rows=[]
        for rank in range(8):
            relative='logs/model-smoke-'+phase+f'/rank-{rank}.json';row=read(relative)
            _need(artifact['receipts_sha256'][prefix+relative]==proof['rank_file_sha256'][str(rank)] and
                  row['initial_parameters_finite'] is True,'Archived child actual rank bytes/finiteness differ')
            rows.append(row)
        _need(rows==proof['ranks'] and model_aggregate(rows,run['run_id'],plan['slurm_job_id'],phase,
              [qid for row in rows for qid in row['question_ids']],allocation['actual_cuda_runtime_uuids'])==plan['resolved_config_sha256'],
              'Archived child actual probability/precision gate differs')
    logs={};states={}
    for rank in range(8):
        relative=prefix+f'logs/optimizer-rank-{rank}.jsonl'
        content=(Path(base)/relative).read_bytes()
        _need(hashlib.sha256(content).hexdigest()==segment['worker_log_sha256'][str(rank)]==artifact['receipts_sha256'][relative],
              'Archived actual optimizer ledger SHA differs')
        logs[rank]=[_strict_json(line) for line in content.splitlines()]
        states[rank]=read(f'logs/policy-calibration-optimizer-rank-{rank}.json')
        _need(artifact['receipts_sha256'][prefix+f'logs/policy-calibration-optimizer-rank-{rank}.json']==
              segment['optimizer_state_proof_sha256'][str(rank)],'Archived Adam summary SHA differs')
    structural=verify_segment(logs,dict(schema_version=1,role=plan['role'],run_id=run['run_id'],
        calibration_sha256=document_sha256(plan),outer_count=1,initial_optimizer_writes=0))
    _same(structural,segment['structural'],'Archived actual monitor/write schedule differs')
    _same(verify_optimizer_states(states,run,plan,document_sha256(plan),allocation),segment['actual_optimizer_states'],
          'Archived actual Adam state summary differs')
    _training_inputs(Path(base)/prefix,run,document_sha256(plan),segment,{},'')
    updates_relative=prefix+'logs/actual-updates.jsonl'
    updates_bytes=(Path(base)/updates_relative).read_bytes()
    updates=[_strict_json(line) for line in updates_bytes.splitlines()]
    _same(updates,segment['actual_updates'],'Archived driver writes differ')
    _need(len(updates)==1 and updates[0]['run_id']==run['run_id'] and type(updates[0]['step']) is int and
          updates[0]['step']==1 and updates[0]['probe_version']==run['initial_probe_version'] and
          type(updates[0]['metrics']['actor/optimizer_steps_this_update']) in (int,float) and
          updates[0]['metrics']['actor/optimizer_steps_this_update']==20,'Archived driver budget differs')
    for value in updates[0]['metrics'].values():
        _need(type(value) in (int,float) and math.isfinite(value),'Archived nonfinite driver metric')
    live=read('artifacts/live/manifest.json');accepted=read('logs/initial-probes-accepted.json')
    _need(accepted['accepted'] is True and accepted['run_id']==run['run_id'] and
          accepted['version']==live['version']==run['initial_probe_version'] and type(live['round']) is int and live['round']==0 and
          accepted['live_manifest_sha256']==artifact['receipts_sha256'][prefix+'artifacts/live/manifest.json'],
          'Archived initial fitted probe admission differs')
    _same(live['encoder_sha256'],plan['initial_reward_fingerprint']['encoder_sha256'],'Archived encoder differs')
    _same(live['head_sha256'],plan['initial_reward_fingerprint']['head_sha256'],'Archived fitted heads differ')
    before,after=segment['evaluation_before'],segment['evaluation_after']
    for label,row in (('before',before),('after',after)):
        _same(read('logs/policy-calibration-evaluation-'+label+'.json'),row,'Archived paired512 evaluation differs')
        calculated=aggregate(row['rows'],before['question_ids'],before['data_sources'])
        for key,value in calculated.items(): _same(row[key],value,'Archived512 arithmetic differs: '+key)
    _need(before['validation_sha256']==after['validation_sha256']==run['input_sha256']['validation.parquet'] and
          before['question_ids']==after['question_ids'] and before['data_sources']==after['data_sources'],
          'Archived child fixed512 pairing differs')
    status=read('logs/pipeline-status.json');cleanup=read('logs/cleanup-status.json');ledger=read('logs/process-ledger.json')
    _need(status['run_id']==cleanup['run_id']==ledger['run_id']==run['run_id'] and cleanup['cleaned'] is True and
          status['stage']=='FINISHED_POLICY_CALIBRATION_MEASUREMENTS_AND_OWNED_CLEANUP' and
          {'trainer','retriever','probe-server'}<=set(ledger['jobs']) and
          all(job['state']=='STOPPED' for job in ledger['jobs'].values()),'Archived cleanup completion differs')
    for name in ('retrieval-staging-claim.json','corpus-cache-claim.json'):
        claim=read('logs/'+name)
        root=Path('/dev/shm')/('search-g1-common8-'+run['run_id']+'-'+plan['slurm_job_id'])
        if name=='corpus-cache-claim.json':root=root/'datasets-cache'
        _need(claim['run_id']==run['run_id'] and claim['phase']=='CLEANED' and claim['slurm_job_id']==plan['slurm_job_id'] and
              type(claim['uid']) is int and claim['uid']==os.getuid() and Path(claim['root'])==root,
              'Archived owned cleanup identity differs')
    complete=read('checkpoint-complete.json')
    _need(complete==segment['checkpoint'] and complete['finite_all_saved_tensors'] is True and
          complete['run_id']==run['run_id'] and type(complete['step']) is int and complete['step']==1 and
          bool(segment['changed_parameters']),'Archived canonical finite fresh1/delta proof differs')


def _verify_parent_probability(base,run,probability,receipts):
    from reproduction.common8.topology import world_size
    from reproduction.common8.precision_contract import sampling_journal_only
    size=world_size(run);journal_only=sampling_journal_only(run)
    for phase in ('initial','coexistence'):
        proof=_receipt(base,probability[phase+'_proof'],probability[phase+'_proof_sha256'])
        _need(proof['accepted'] is True and proof['run_id']==run['run_id'] and proof['phase']==phase and
              len(proof['ranks'])==size and type(proof['actual_grpo_updates']) is int and proof['actual_grpo_updates']==0,
              'Parent original actual probability receipt differs')
        from reproduction.common8.model_smoke import aggregate as model_aggregate
        rows=[]
        for rank,row in enumerate(proof['ranks']):
            relative='logs/policy-safety-admission-receipts/parent/model-smoke-'+phase+f'-rank-{rank}.json'
            original=_receipt(base,relative,receipts[relative])
            _same(original,row,'Parent original rank receipt differs')
            _need(receipts[relative]==proof['rank_file_sha256'][str(rank)],'Parent original rank SHA differs')
            rows.append(original)
            _need(type(row['rank']) is int and row['rank']==rank and type(row['actual_optimizer_writes']) is int and
                  row['actual_optimizer_writes']==0 and row['initial_parameters_finite'] is True and
                  row['parity']['tolerance']==(None if journal_only else .05) and all(type(row['parity'][key]) in (int,float) and
                  math.isfinite(row['parity'][key]) and 0<=row['parity'][key] and (journal_only or row['parity'][key]<=.05) for key in
                  ('actor_reference_max_abs','actor_vllm_max_abs')),'Parent original rank parity/freshness differs')
        _need(model_aggregate(rows,run['run_id'],probability['slurm_job_id'],phase,
              probability['question_ids'],probability['actual_cuda_runtime_uuids'],run=run)==probability['resolved_config_sha256'],
              'Parent actual probability/config/precision receipt differs')


def verify_admission(base,artifact,run):
    """Read-only frozen receipt verification; safe after claim and every outer.

    This does not rerun a GPU, HTTP endpoint, cleanup or the produce-only fresh
    namespace checks. Historical child identities remain their original IDs.
    """
    _need(type(artifact['schema_version']) is int and artifact['schema_version']==2 and
          artifact['accepted'] is True and artifact['role']=='healthy_pretraining_policy_calibration',
          'Not independent healthy production admission schema2')
    _need(run.get('policy_calibration_role','production')=='production' and
          artifact['run_id']==artifact['parent_production_run_id']==run['run_id'],'Wrong production parent')
    for key in ('initial_model_sha256','source_sha256','input_sha256'):
        _same(artifact[key],run[key],'Production scientific identity differs: '+key)
    _same(artifact['production_identity'],_identity(run),'Production official/math identity differs')
    _need(artifact['discovery_run_id']!=artifact['confirmation_run_id'] and
          run['run_id'] not in (artifact['discovery_run_id'],artifact['confirmation_run_id']) and
          artifact['discovery_job_id']!=artifact['confirmation_job_id'],'Child roles are not independent')
    receipts=artifact['receipts_sha256']
    _need(isinstance(receipts,dict) and bool(receipts),'No immutable admission receipts')
    for relative,sha in receipts.items():
        path=(Path(base)/relative).resolve()
        _need(Path(base).resolve() in path.parents and not (Path(base)/relative).is_symlink() and digest(path)==sha,
              'Immutable receipt path/hash differs')
    probability=_receipt(base,artifact['probability_proof'],artifact['probability_proof_sha256'])
    summary=_receipt(base,artifact['short_segment_proof'],artifact['short_segment_proof_sha256'])
    _need(probability['accepted'] is True and probability['role']=='production_initial_probability_admission_summary' and
          probability['run_id']==run['run_id'] and probability['phase']=='initial' and
          type(probability['actual_optimizer_writes']) is int and probability['actual_optimizer_writes']==0 and
          type(probability['ranks']) is list and all(type(rank) is int for rank in probability['ranks']) and
          probability['ranks']==list(range(8)),'Not actual parent fresh0 probability summary')
    _verify_parent_probability(base,run,probability,receipts)
    _need(summary['accepted'] is True and summary['role']=='healthy_independent_confirmation_summary' and
          summary['run_id']==artifact['confirmation_run_id'] and summary['parent_production_run_id']==run['run_id'] and
          summary['discovery_run_id']==artifact['discovery_run_id'] and
          summary['slurm_job_id']==artifact['confirmation_job_id'] and
          summary['discovery_job_id']==artifact['discovery_job_id'] and
          type(summary['actual_optimizer_writes_per_rank']) is int and summary['actual_optimizer_writes_per_rank']==20 and
          type(summary['monitors_per_rank']) is int and summary['monitors_per_rank']==40 and
          summary['all_parameters_and_adam_finite'] is True,'Not an independent healthy confirmation summary')
    child_documents={}
    for label,role in (('discovery',DISCOVERY_ROLE),('confirmation',CONFIRMATION_ROLE)):
        descriptor=artifact['children'][label]
        segment=_receipt(base,descriptor['segment'],descriptor['segment_sha256'])
        plan=_receipt(base,descriptor['plan'],descriptor['plan_sha256'])
        _need(segment['run_id']==artifact[label+'_run_id'] and segment['slurm_job_id']==artifact[label+'_job_id'] and
              segment['role']==role+'_evidence' and plan['role']==role and plan['run_id']==segment['run_id'] and
              plan['parent_production_run_id']==run['run_id'] and document_sha256(plan)==segment['plan_sha256'],
              'Archived child plan/segment identity differs')
        _same(segment['identity'],artifact['production_identity'],'Archived child official/math identity differs')
        _same(plan['initial_reward_fingerprint'],artifact['initial_reward_fingerprint'],'Archived frozen reward identity differs')
        _need(Path(descriptor['base']).resolve()==Path(run['policy_calibration_'+label+'_base']).resolve(),
              'Archived child source descriptor differs')
        _archived_segment(base,artifact,descriptor,plan,segment)
        child_documents[label]=(plan,segment)
    from reproduction.common8.policy_calibration_plan import verify_plan
    discovery_plan,discovery=child_documents['discovery']
    confirmation_plan,confirmation=child_documents['confirmation']
    _archive_adoption(base,artifact,run,probability['slurm_job_id'],artifact['parent_adoption_proof'],artifact['parent_live_proof'])
    confirmation_prefix='logs/policy-safety-admission-receipts/confirmation/'
    confirmation_run=_receipt(base,confirmation_prefix+'logs/run-manifest.json',receipts[confirmation_prefix+'logs/run-manifest.json'])
    _archive_adoption(base,artifact,confirmation_run,confirmation_plan['slurm_job_id'],
                      confirmation_prefix+'logs/initial-probe-adoption.json',confirmation_prefix+'artifacts/live/manifest.json')
    verify_plan(discovery_plan,run,expected_plan_sha256=document_sha256(discovery_plan))
    discovery_descriptor=artifact['children']['discovery']
    discovery_bytes=(Path(base)/discovery_descriptor['segment']).read_bytes()
    verify_plan(confirmation_plan,run,expected_plan_sha256=document_sha256(confirmation_plan),
                discovery_plan=discovery_plan,discovery_evidence_bytes=discovery_bytes)
    _same(confirmation_plan['health_rule'],artifact['health_rule'],'Frozen health rule differs')
    _same(confirmation_plan['discovery']['threshold_selection'],artifact['threshold_selection'],'Threshold provenance differs')
    _need(confirmation_plan['discovery']['evidence_sha256']==artifact['children']['discovery']['segment_sha256'],
          'Confirmation did not freeze actual discovery bytes')
    _same(confirmation_plan['candidate_thresholds'],artifact['candidate_thresholds'],'Frozen threshold candidates differ')
    thresholds=artifact['candidate_thresholds']
    _need(thresholds['dual_clip']==artifact['dual_clip']==3. and thresholds['epsilon']==.2 and
          thresholds['target_old_policy_kl']==artifact['target_old_policy_kl']==.01 and
          thresholds['kl_log_bound']==artifact['kl_log_bound']==30.,'Original candidate guards changed')
    tail=thresholds['negative_tail_limit'];_number(tail,'negative_tail_limit')
    _need(tail<=1 and tail==artifact['negative_tail_limit'] and summary['negative_tail_limit']==tail and
          artifact['actual_probability_gate_accepted'] is True and artifact['actual_native_short_segment_accepted'] is True,
          'Missing real calibrated tail/segment acceptance')
    for label,segment in (('discovery',discovery),('confirmation',confirmation)):
        for row in segment['structural']['global_monitors']:
            _need(row['sampled_old_policy_kl']<=.01,'Archived actual segment violated KL brake')
            if label=='confirmation':
                _need(row['negative_tail_weight_fraction']<=tail,'Archived confirmation violated tail brake')
    _same(evaluate_health(artifact['health_rule'],discovery,confirmation),artifact['health_decision'],
          'Independent frozen health decision differs')
    _same(summary['health_decision'],artifact['health_decision'],'Summary health decision differs')
    _need(type(artifact['parent_actual_optimizer_writes']) is int and artifact['parent_actual_optimizer_writes']==0,
          'Child writes were attributed to production')
    return tail,document_sha256(artifact)


def produce(base,discovery_base,confirmation_base,environ=None):
    """Validate everything before exclusively publishing reviewable admission."""
    from reproduction.common8.freeze_calibration_plan import (
        _unstarted,_resolved_document,_mathematics,_actual_prerequisites,_reward_fingerprint,
    )
    from reproduction.common8.calibration_runtime import write_once
    environ=os.environ if environ is None else environ
    base=Path(base).resolve();discovery_base=Path(discovery_base).resolve();confirmation_base=Path(confirmation_base).resolve()
    _need(len({base,discovery_base,confirmation_base})==3,'Parent and children namespaces must be independent')
    run=_read(base,'logs/run-manifest.json')
    _need(run.get('policy_calibration_role','production')=='production' and run.get('policy_safety_required') is True,
          'Not fresh required-safety production')
    _need(Path(run['policy_calibration_discovery_base']).resolve()==discovery_base and
          Path(run['policy_calibration_confirmation_base']).resolve()==confirmation_base,'Unbound child source descriptor')
    _unstarted(base,run);_identity(run)
    target=base/'logs/policy-safety-calibration.json'
    _need(not target.exists(),'Production calibrated admission already exists')
    document=_resolved_document(base,run);_mathematics(document,base,run)
    config_sha=document_sha256(document);job_id=environ['SLURM_JOB_ID']
    _actual_prerequisites(base,run,document,config_sha,job_id)
    fingerprint=_reward_fingerprint(base,run,environ)
    receipts={}
    discovery_run,discovery_plan,discovery=verify_child(discovery_base,run,DISCOVERY_ROLE,receipts,'discovery/')
    confirmation_run,confirmation_plan,confirmation=verify_child(confirmation_base,run,CONFIRMATION_ROLE,receipts,'confirmation/')
    _need(len({run['run_id'],discovery_run['run_id'],confirmation_run['run_id']})==3 and
          discovery_plan['slurm_job_id']!=confirmation_plan['slurm_job_id'],'Confirmation was not a separate fresh child')
    _same(discovery_plan['initial_reward_fingerprint'],fingerprint,'Parent did not adopt the calibrated original reward identity')
    _same(confirmation_plan['initial_reward_fingerprint'],fingerprint,'Confirmation reward identity differs')
    _need(confirmation_plan['discovery']['run_id']==discovery_run['run_id'] and
          confirmation_plan['discovery']['plan_sha256']==document_sha256(discovery_plan) and
          confirmation_plan['discovery']['evidence_sha256']==digest(discovery_base/'logs/policy-calibration-segment.json'),
          'Confirmation did not freeze this actual discovery')
    thresholds=confirmation_plan['candidate_thresholds'];tail=thresholds['negative_tail_limit']
    _number(tail,'negative_tail_limit');_need(tail<=1,'Invalid calibrated tail')
    for label,segment in (('discovery',discovery),('confirmation',confirmation)):
        for row in segment['structural']['global_monitors']:
            _need(row['sampled_old_policy_kl']<=thresholds['target_old_policy_kl'],'Actual segment violated the KL brake')
            if label=='confirmation': _need(row['negative_tail_weight_fraction']<=tail,'Actual confirmation violated the tail brake')
    source_live=_read(discovery_base,'artifacts/live/manifest.json')
    archive_path=Path(source_live['commit']).parent.parent/'version-manifest.json'
    archive_relative=archive_path.resolve().relative_to(discovery_base).as_posix()
    archive_raw=archive_path.read_bytes()
    _same(_strict_json(archive_raw),source_live,'Discovery original version archive differs')
    receipts['discovery/'+archive_relative]=archive_raw
    source_names=('logs/run-manifest.json','logs/policy-safety-calibration-plan.json','logs/policy-calibration-segment.json',
                  'logs/initial-probes-accepted.json','artifacts/live/manifest.json')
    source_hashes={name:hashlib.sha256(receipts['discovery/'+name]).hexdigest() for name in source_names}
    source_hashes[archive_relative]=hashlib.sha256(archive_raw).hexdigest()
    for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
        source_hashes[Path(source_live[key]).relative_to(discovery_base).as_posix()]=source_live['head_sha256'][name]
    for target,target_run,target_job,prefix in ((base,run,job_id,'parent/'),
            (confirmation_base,confirmation_run,confirmation_plan['slurm_job_id'],'confirmation/')):
        adoption=_read(target,'logs/initial-probe-adoption.json',receipts,prefix)
        live=_read(target,'artifacts/live/manifest.json',receipts,prefix)
        _adoption_lineage(target_run,target_job,adoption,live,discovery_run,discovery_plan,discovery,
                          source_live,source_hashes,hashlib.sha256(archive_raw).hexdigest(),
                          hashlib.sha256(receipts[prefix+'artifacts/live/manifest.json']).hexdigest())
        for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
            _need(digest(Path(live[key]))==digest(Path(source_live[key]))==source_live['head_sha256'][name],
                  'Actual original/adopted raw head bytes differ')
    health=evaluate_health(confirmation_plan['health_rule'],discovery,confirmation)
    # All expensive physical/file checks and rule decisions precede any writes.
    receipt_root=base/'logs/policy-safety-admission-receipts'
    _need(not receipt_root.exists(),'Admission receipt namespace already attempted')
    write_once(base/'logs/policy-safety-admission-claim.json',dict(run_id=run['run_id'],
        discovery_run_id=discovery_run['run_id'],confirmation_run_id=confirmation_run['run_id'],slurm_job_id=job_id))
    receipt_root.mkdir(mode=0o700)
    receipt_hashes={}
    for name,content in receipts.items():
        path=receipt_root/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with path.open('xb') as stream:
            os.chmod(path,0o600);stream.write(content);stream.flush();os.fsync(stream.fileno())
        relative=path.relative_to(base).as_posix();receipt_hashes[relative]=hashlib.sha256(content).hexdigest()
    def parent_receipt(phase):
        relative='logs/policy-safety-admission-receipts/parent/model-smoke-'+phase+'-complete.json'
        path=base/relative;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        content=(base/'logs'/('model-smoke-'+phase)/'complete.json').read_bytes()
        with path.open('xb') as stream:
            os.chmod(path,0o600);stream.write(content);stream.flush();os.fsync(stream.fileno())
        receipt_hashes[relative]=hashlib.sha256(content).hexdigest()
        for rank in range(8):
            rank_relative='logs/policy-safety-admission-receipts/parent/model-smoke-'+phase+f'-rank-{rank}.json'
            rank_content=(base/'logs'/('model-smoke-'+phase)/f'rank-{rank}.json').read_bytes()
            with (base/rank_relative).open('xb') as stream:
                os.chmod(base/rank_relative,0o600);stream.write(rank_content);stream.flush();os.fsync(stream.fileno())
            receipt_hashes[rank_relative]=hashlib.sha256(rank_content).hexdigest()
        return relative,receipt_hashes[relative]
    initial,initial_sha=parent_receipt('initial');coexistence,coexistence_sha=parent_receipt('coexistence')
    allocation=_read(base,'logs/native-allocation-admission.json')
    initial_original=_read(base,'logs/model-smoke-initial/complete.json')
    for directory in sorted({(base/name).parent for name in receipt_hashes},key=lambda path:len(path.parts),reverse=True):
        _sync_directory(directory)
    _sync_directory(receipt_root);_sync_directory(receipt_root.parent)
    probability=dict(schema_version=1,role='production_initial_probability_admission_summary',accepted=True,
        run_id=run['run_id'],slurm_job_id=job_id,phase='initial',actual_optimizer_writes=0,ranks=list(range(8)),
        question_ids=[qid for row in initial_original['ranks'] for qid in row['question_ids']],
        actual_cuda_runtime_uuids=allocation['actual_cuda_runtime_uuids'],
        initial_proof=initial,initial_proof_sha256=initial_sha,coexistence_proof=coexistence,coexistence_proof_sha256=coexistence_sha,
        source_sha256=run['source_sha256'],resolved_config_sha256=config_sha,
        scope='actual parent initial/coexistence gates; no child writes attributed to parent')
    summary=dict(schema_version=1,role='healthy_independent_confirmation_summary',accepted=True,
        run_id=confirmation_run['run_id'],parent_production_run_id=run['run_id'],
        discovery_run_id=discovery_run['run_id'],discovery_job_id=discovery_plan['slurm_job_id'],
        slurm_job_id=confirmation_plan['slurm_job_id'],actual_optimizer_writes_per_rank=20,monitors_per_rank=40,
        all_parameters_and_adam_finite=True,negative_tail_limit=tail,target_old_policy_kl=thresholds['target_old_policy_kl'],
        health_decision=health,scope='independent verified child calibration; not parent optimizer writes')
    probability_path='logs/policy-safety-production-initial-probability.json';summary_path='logs/policy-safety-confirmation-summary.json'
    write_once(base/probability_path,probability);write_once(base/summary_path,summary)
    children={}
    for label in ('discovery','confirmation'):
        prefix='logs/policy-safety-admission-receipts/'+label+'/'
        children[label]=dict(segment=prefix+'logs/policy-calibration-segment.json',
            segment_sha256=receipt_hashes[prefix+'logs/policy-calibration-segment.json'],
            plan=prefix+'logs/policy-safety-calibration-plan.json',plan_sha256=receipt_hashes[prefix+'logs/policy-safety-calibration-plan.json'],
            base=str(discovery_base if label=='discovery' else confirmation_base))
    artifact=dict(schema_version=2,role='healthy_pretraining_policy_calibration',accepted=True,
        run_id=run['run_id'],parent_production_run_id=run['run_id'],parent_actual_optimizer_writes=0,
        discovery_run_id=discovery_run['run_id'],confirmation_run_id=confirmation_run['run_id'],
        discovery_job_id=discovery_plan['slurm_job_id'],confirmation_job_id=confirmation_plan['slurm_job_id'],
        production_identity=_identity(run),initial_model_sha256=run['initial_model_sha256'],source_sha256=run['source_sha256'],input_sha256=run['input_sha256'],
        initial_reward_fingerprint=fingerprint,candidate_thresholds=thresholds,dual_clip=thresholds['dual_clip'],
        target_old_policy_kl=thresholds['target_old_policy_kl'],kl_log_bound=thresholds['kl_log_bound'],negative_tail_limit=tail,
        probability_proof=probability_path,probability_proof_sha256=digest(base/probability_path),
        short_segment_proof=summary_path,short_segment_proof_sha256=digest(base/summary_path),
        actual_probability_gate_accepted=True,actual_native_short_segment_accepted=True,
        health_rule=confirmation_plan['health_rule'],threshold_selection=confirmation_plan['discovery']['threshold_selection'],
        health_decision=health,children=children,receipts_sha256=receipt_hashes,
        parent_adoption_proof='logs/policy-safety-admission-receipts/parent/logs/initial-probe-adoption.json',
        parent_live_proof='logs/policy-safety-admission-receipts/parent/artifacts/live/manifest.json',
        discovery_original_version_archive='logs/policy-safety-admission-receipts/discovery/'+archive_relative,
        semantics='verified independent child calibration bound to fresh production; original run/job identities retained')
    verify_admission(base,artifact,run)
    # Do not permit another attempt to claim parent training during receipt creation.
    _unstarted(base,run)
    write_once(target,artifact)
    return artifact


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--discovery-base',type=Path,required=True);parser.add_argument('--confirmation-base',type=Path,required=True)
    arguments=parser.parse_args();result=produce(arguments.base,arguments.discovery_base,arguments.confirmation_base)
    print(json.dumps(dict(run_id=result['run_id'],accepted=result['accepted'],artifact_sha256=document_sha256(result))))

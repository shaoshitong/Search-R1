#!/usr/bin/env python3
"""Future final effective200 trace-retention audit; source preparation is not acceptance.

Adapted from actual tree171_trace_retention.py (30579B/ac8b0fca4b4a02275a2953605b2c10604e9fe6a3da4897739e86c038a07b1fb5).
Future actual manifest must prove train200/native200 and bind all200 original local
rollout metadata refs. Original dev200 control JSON must be actual COMPLETE. FULL200 is optional;
full train metric may remain explicitly pending behind slow51713. No ROOTFULL200, final-score completion or actual201 required.
Exactly one compressed retained18 route call, nice19/own idle IO/no GPU, no remote
writes. This source elects one fresh full200 SHA/structural read (about4.7GB stays
remote), also compares fresh1..171 hashes to the unchanged prior171 audit.
Ordinary JSON cannot reconstruct full token/action/raw-ledger vectors or tokenwise
weights/advantages; original writer evidence and current SHA checks stay distinct.
No retry, overwrite of old171 evidence, PT/HF read or training/evaluation operation.
"""
import argparse,base64,datetime,hashlib,importlib.util,json,os,re,shlex,subprocess,time,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/tree-nested-branch-handoff-20261013'
MIRRORS=[ROOT/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
NAME='TREE_TRACE_RETENTION_THROUGH200_ACTUAL'
SOURCE=ROOT/'work/tree-speed-repair-20261008/nested-native5-recovery-source'
ROUTE=ROOT/'work/tree-speed-repair-20261008/ssh_retained18.py'
ROUTE_SHA='8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'
CORE=ROOT/'work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py'
CORE_SHA='676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
COMMITS=['0befbaf99fc44942105831b3fbec94e4bf1b8ccb','ba9f6627ec9a4e9749830d7b6acca53f6feec6b4','778b2b43dad90fc0c1c5d95d77c2326286011b73']
CONFIGS=['2a317628c497bba932698bf6d7bf64ca3503233d32080a389a36e1f69b362243','22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59','355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3']
MODEL='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
INPUTS={}

REMOTE = r'''
import os,json,time,hashlib,math,shutil,subprocess,re
from pathlib import Path
from collections import Counter,defaultdict
os.nice(19)
io_priority_result=None
if shutil.which('ionice'):
    ip=subprocess.run(['ionice','-c','3','-p',str(os.getpid())],capture_output=True,text=True)
    io_priority_result=dict(returncode=ip.returncode,stderr=ip.stderr[-200:])
PINS = __PINS__
CONTROL200 = __CONTROL200__
IDENTITIES = __IDENTITIES__
RANK_WRITER_SHAS = __RANK_WRITER_SHAS__
START=time.time(); CPU=time.process_time()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def stream(p,consume):
    before=p.stat();digest=hashlib.sha256();head={};section=None;chunk=None;keys=set();n=0
    with p.open('rb') as f:
        for raw in f:
            n+=len(raw);digest.update(raw);line=raw.decode('utf-8')
            if chunk is not None:
                chunk.append(line)
                if line.startswith('    }'):
                    value=json.loads(''.join(chunk).rstrip().rstrip(','));consume(section,value);chunk=None
            elif line.startswith('    {') and section:
                chunk=[line]
            elif line.startswith('  "'):
                key,value=line.strip().split(':',1);key=json.loads(key);keys.add(key);value=value.strip().rstrip(',')
                section=key if value=='[' else None
                if value not in ('[','{'):
                    try:head[key]=json.loads(value)
                    except json.JSONDecodeError:pass
            elif line.startswith('  ]'):section=None
    after=p.stat()
    if chunk is not None:raise ValueError('Incomplete indented JSON record')
    return dict(bytes=n,sha256=digest.hexdigest(),mtime_ns=after.st_mtime_ns,stable=before.st_size==after.st_size==n and before.st_mtime_ns==after.st_mtime_ns,header=head,top_fields=sorted(keys))

required='stage branch_event_id parent_trajectory_id child_trajectory_id ancestor_trajectory_ids actual_prefix_trajectory_id action_start_id action_depth sampled_reward parent_reward original_root_reward accepted training_leaf reason stop_reason fallback initial_state_attempts backoff_actions b2_prefix_origin inherited_prefix_sha256 inherited_raw_ledger_sha256 inherited_response_tokens inherited_observation_tokens remaining_turns reward_difference_tolerance'.split()
results=[];totalbytes=0;allcandidate=0;allleaf=0;allquestion=0;allfail=Counter();allmissing=Counter();schemas={};errors=[]
for pin in PINS:
    p=Path(pin['run'])/'output/tree-audit'/pin['audit'];q=p.with_name(p.stem+'_policy_batch.json')
    if p.is_symlink() or q.is_symlink():raise ValueError('Original own audit/policy JSON must not be a symlink')
    try:
        expected_budget=pin['bytes']+4_000_000
        if totalbytes+expected_budget>5_500_000_000:raise ValueError('Explicit audit JSON IO budget reached')
        if p.stat().st_size>32_000_000 or q.stat().st_size>4_000_000:raise ValueError('Audit size exceeds fixed retention inventory budget')
        leaves={};branches={};accepted={};candidate_ids=set();fail=Counter();missing=Counter();stages=Counter();reasons=Counter();fallback=Counter();depths=Counter();prefixorigins=Counter();fields=defaultdict(set);rowcounts=Counter();questiongroups={};candidatecount=0
        def consume(section,v):
            nonlocal_dummy=None
            fields[section].update(v);rowcounts[section]+=1
            if section=='branches':
                h=v.get('child_trajectory_hash')
                if h in branches:fail['duplicate_branch_child']+=1
                branches[h]=v
            elif section=='leaves':
                if v['trajectory_hash'] in leaves:fail['duplicate_training_leaf']+=1
                leaves[v['trajectory_hash']]=v
            elif section=='question_groups':
                if v['question_hash'] in questiongroups:fail['duplicate_question_group']+=1
                questiongroups[v['question_hash']]=v
            elif section=='candidate_attempts':
                for k in required:
                    if k not in v:missing[k]+=1
                h=fp(v['child_trajectory_id'])
                if h in candidate_ids:fail['duplicate_candidate_child']+=1
                candidate_ids.add(h);stages[v.get('stage')]+=1;reasons[v.get('reason')]+=1;depths[str(v.get('stage'))+':'+str(v.get('action_depth'))]+=1
                fallback[str(v.get('stage'))]+=int(bool(v.get('fallback')))
                if v.get('stage')=='B2':prefixorigins[str(v.get('b2_prefix_origin'))]+=1
                if v.get('accepted')!=v.get('training_leaf') or type(v.get('accepted')) is not bool or type(v.get('fallback')) is not bool:fail['accepted_training_leaf_flag']+=1
                if v.get('stage') not in ('A1','B1','B2') or not isinstance(v.get('reason'),str) or not v.get('reason') or not isinstance(v.get('stop_reason'),str) or not v.get('stop_reason'):fail['candidate_stage_reason_stop_record']+=1
                if v.get('accepted'):accepted[h]=v
                elif h in leaves:fail['rejected_candidate_in_training_leaves']+=1
                vals=[v.get(k) for k in ['sampled_reward','parent_reward','original_root_reward','reward_difference_tolerance']]
                if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals):fail['missing_or_nonfinite_candidate_reward']+=1
        meta=stream(p,consume);totalbytes+=meta['bytes']
        if not meta['stable']:fail['unstable_rollout_read']+=1
        if meta['bytes']!=pin['bytes']:fail['historical_rollout_bytes']+=1
        if pin.get('sha256') and meta['sha256']!=pin['sha256']:fail['historical_rollout_sha256']+=1
        if meta['header'].get('global_step')!=pin['outer']:fail['outer_identity']+=1
        if meta['header'].get('status')!='ROLLOUT_DIAGNOSTIC_COMPLETE':fail['rollout_status']+=1
        grouped=defaultdict(dict)
        for h,v in leaves.items():
            m=v.get('branch_sampling',{});stage=m.get('stage');label=stage if stage else ('A' if v.get('root_ordinal')==0 else 'B')
            g=grouped[v['question_hash']]
            if label in g:fail['duplicate_question_leaf_label']+=1
            g[label]=(h,v)
            if v.get('relative_leaf_mass')!=0.2:fail['recorded_leaf_relative_mass']+=1
            if v.get('root_leaf_quota')!=(2 if label in ('A','A1') else 3):fail['recorded_root_leaf_quota']+=1
            if label in ('A','B'):
                if m.get('stop_reason')!='original_root' or not isinstance(m.get('sampled_reward'),(int,float)):fail['original_root_reward_record']+=1
            else:
                c=accepted.get(h);b=branches.get(h)
                if c is None or c!=m:fail['accepted_candidate_leaf_metadata_binding']+=1
                if b is None:fail['accepted_branch_record_missing']+=1;continue
                if b.get('branch_sampling')!=m or not b.get('training_leaf'):fail['accepted_branch_metadata_binding']+=1
                if b.get('parent_fields')!=b.get('child_fields') or not b.get('inherited'):fail['snapshot_inheritance_hash_binding']+=1
                parent=b.get('parent_fields',{})
                if parent.get('rolling')!=m.get('inherited_prefix_sha256') or parent.get('raw_ledger')!=m.get('inherited_raw_ledger_sha256'):fail['prefix_ledger_hash_binding']+=1
                if b.get('event_parent_trajectory_hash')!=fp(m.get('parent_trajectory_id')):fail['event_parent_binding']+=1
                if b.get('parent_trajectory_hash')!=fp(m.get('actual_prefix_trajectory_id')):fail['snapshot_origin_binding']+=1
                if b.get('parent_node_hash')!=fp(m.get('action_start_id')):fail['action_start_binding']+=1
                if b.get('tree_hash')!=v.get('tree_hash') or b.get('source')!=v.get('source'):fail['branch_leaf_tree_source_binding']+=1
        for question,g in grouped.items():
            if set(g)!=set(['A','A1','B','B1','B2']):fail['five_question_leaf_labels']+=1;continue
            if g['A'][1]['tree_hash']!=g['A1'][1]['tree_hash'] or len({g[x][1]['tree_hash'] for x in ['B','B1','B2']})!=1 or g['A'][1]['tree_hash']==g['B'][1]['tree_hash']:fail['root_tree_group_binding']+=1
            for child,parent in [('A1','A'),('B1','B'),('B2','B1')]:
                cm=g[child][1]['branch_sampling']
                if fp(cm['parent_trajectory_id'])!=g[parent][0]:fail['actual_'+parent+'_to_'+child+'_parent_binding']+=1
                ancestors=[fp(x) for x in cm.get('ancestor_trajectory_ids',[])]
                expected=[g['B'][0],g['B1'][0]] if child=='B2' else [g[parent][0]]
                if ancestors!=expected:fail['ancestor_chain_binding']+=1
            qg=questiongroups.get(question,{})
            if qg.get('roots')!=2 or qg.get('leaves')!=5:fail['question_group_record_binding']+=1
        if len(accepted)!=1536 or len(leaves)!=2560 or len(grouped)!=512 or len(branches)!=1536 or rowcounts['leaves']!=2560 or rowcounts['branches']!=1536 or rowcounts['question_groups']!=512:fail['expected_record_counts']+=1
        if len(accepted)+rowcounts['candidate_attempts']-len(accepted)!=len(candidate_ids):fail['candidate_count_identity']+=1
        if set(accepted)-set(leaves):fail['accepted_candidate_missing_training_leaf']+=1
        policyrows={};policyfields=set();policyfail=Counter()
        def consume_policy(section,v):
            if section=='rows':
                policyfields.update(v);h=v['trajectory_hash']
                if h in policyrows:policyfail['duplicate_policy_row']+=1
                policyrows[h]=v
                previous=leaves.get(h)
                if previous is None or v.get('tree_hash')!=previous.get('tree_hash') or v.get('question_hash')!=previous.get('question_hash'):policyfail['policy_leaf_binding']+=1
                tokens=v.get('effective_tokens')
                if type(tokens) is not int or tokens<0:policyfail['effective_token_count']+=1
                expected=1. if isinstance(tokens,int) and tokens>0 else 0.
                if v.get('empty_leaf')!=(expected==0.):policyfail['empty_leaf_mass_binding']+=1
                if not math.isclose(v.get('policy_weight_sum',-1),expected,rel_tol=1e-6,abs_tol=1e-6):policyfail['policy_weight_mass']+=1
        pm=stream(q,consume_policy);totalbytes+=pm['bytes']
        if not pm['stable']:policyfail['unstable_policy_read']+=1
        if pm['header'].get('status')!='POLICY_BATCH_DIAGNOSTIC_COMPLETE':policyfail['policy_status']+=1
        if pm['header'].get('rollout_file_sha256')!=meta['sha256']:policyfail['policy_original_rollout_sha256_binding']+=1
        if set(policyrows)!=set(leaves):policyfail['policy_leaf_set_binding']+=1
        fail.update(policyfail);allfail.update(fail);allmissing.update(missing)
        for k,v in fields.items():schemas.setdefault(k,set()).update(v)
        schemas.setdefault('policy_rows',set()).update(policyfields)
        allcandidate+=rowcounts['candidate_attempts'];allleaf+=len(leaves);allquestion+=len(grouped)
        results.append(dict(outer=pin['outer'],run_commit=pin['commit'],audit=pin['audit'],audit_bytes=meta['bytes'],audit_sha256=meta['sha256'],original_sha256_known=bool(pin.get('sha256')),original_sha256_matches=None if not pin.get('sha256') else meta['sha256']==pin['sha256'],stable=meta['stable'],candidate_count=rowcounts['candidate_attempts'],accepted=len(accepted),rejected=rowcounts['candidate_attempts']-len(accepted),questions=len(grouped),leaves=len(leaves),branches=len(branches),stages=dict(stages),reasons=dict(reasons),fallback=dict(fallback),action_depth_counts=dict(depths),b2_prefix_origin_counts=dict(prefixorigins),candidate_required_field_missing=dict(missing),binding_discrepancies=dict(fail),policy_bytes=pm['bytes'],policy_sha256=pm['sha256'],policy_rows=len(policyrows),policy_rollout_hash_matches=pm['header'].get('rollout_file_sha256')==meta['sha256'],raw_token_vectors_present=any(k in fields['leaves'] for k in ['response','response_token_ids','raw_ledger','prompt_token_ids','action_token_ids']),rollout_top_fields=meta['top_fields'],policy_top_fields=pm['top_fields']))
    except Exception as exc:errors.append(dict(outer=pin['outer'],error_type=type(exc).__name__,error=str(exc)[:240]))

# Strict admitted rank JSON inventory. Profiles and failure tensors are never read.
rank_errors=[];rank_rows=[];rank_totals={str(r):0 for r in range(8)};rank_counts={};rank_bytes=0;zero_writes=0;excluded_names=0;historical_zero_samples=[]
for commit,identity in IDENTITIES.items():
    lo,hi=identity['effective_outers'];directory=Path(identity['run'])/'output/tree-rank-audit';admitted={};count=0
    try:
        for path in sorted(directory.iterdir()):
            match=re.fullmatch(r'step-(\d{6})(?:-attempt-(\d{6}))?-rank-(\d{2})\.json',path.name)
            if match is None:
                if path.suffix=='.json':excluded_names+=1
                continue
            outer,rank=int(match[1]),int(match[3])
            if not lo<=outer<=hi:continue
            if path.is_symlink() or not path.is_file():raise ValueError('Original own rank JSON path differs')
            with path.open('rb') as stream:
                before=os.fstat(stream.fileno())
                if not 0<before.st_size<=262144:raise ValueError('Small rank JSON budget')
                raw=stream.read();after=os.fstat(stream.fileno())
            rank_bytes+=len(raw)
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Rank JSON changed during read')
            value=json.loads(raw)
            if value.get('kind')!='tree_actor_rank' or value.get('outer_step')!=outer or value.get('rank')!=rank or value.get('world_size')!=8:raise ValueError('Rank record kind/outer/rank/world differs')
            if value.get('code_commit')!=commit or value.get('config_sha256')!=identity['config_sha256'] or value.get('data_identity')!=identity['data_identity'] or value.get('run_id')!=Path(identity['run']).name:raise ValueError('Rank source/config/data/run identity differs')
            if value.get('auditor_source_sha256')!=RANK_WRITER_SHAS[commit]:raise ValueError('Original rank writer SHA differs')
            if match[2] is not None and value.get('attempt_id')!=int(match[2]):raise ValueError('Rank filename attempt differs')
            calls=value.get('successful_optimizer_calls')
            if type(calls) is not int or not 0<=calls<=20:raise ValueError('Actual successful rank writes differ')
            if calls==0:
                zero_writes+=1
                continue
            if (outer,rank) in admitted:raise ValueError('Ambiguous positive-write records for one effective outer/rank')
            sample=value.get('local_shard_sample_delta') or {}
            historical_zero_sample=(commit=='0befbaf99fc44942105831b3fbec94e4bf1b8ccb' and outer==1 and sample.get('nonzero')==0)
            if value.get('finite_grad_norms')!=calls or not 1<=value.get('nonzero_grad_norms',0)<=calls or not isinstance(sample.get('count'),int) or sample['count']<=0 or sample.get('finite')!=sample['count'] or not (historical_zero_sample or 1<=sample.get('nonzero',0)<=sample['count']):raise ValueError('Rank finite/nonzero sample incomplete')
            if historical_zero_sample:historical_zero_samples.append(dict(outer=outer,rank=rank,source_commit=commit,file=path.name,sha256=hashlib.sha256(raw).hexdigest(),sample=sample,sampled_parameter_change_nonzero_proven=False,successful_optimizer_calls=calls,finite_grad_norms=value['finite_grad_norms'],nonzero_grad_norms=value['nonzero_grad_norms']))
            processed=value.get('opportunities_processed');skipped=value.get('opportunities_skipped',value.get('skipped_optimizer_calls'))
            if processed is not None and processed!=20:raise ValueError('Explicit rank processed opportunities differ')
            if skipped is not None and (type(skipped) is not int or not 0<=skipped<=19 or calls+skipped!=20):raise ValueError('Explicit rank skipped opportunities differ')
            admitted[(outer,rank)]=dict(rank=rank,file=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),calls=calls,processed=processed,skipped=skipped)
        for outer in range(lo,hi+1):
            values=[admitted[(outer,r)] for r in range(8)]
            for row in values:rank_totals[str(row['rank'])]+=row['calls']
            rank_rows.append(dict(outer=outer,rank_file_set_sha256=fp(values),actual_writes=[row['calls'] for row in values],processed=[row['processed'] for row in values],skipped=[row['skipped'] for row in values]))
        count=len(admitted)
        if count!=(hi-lo+1)*8:raise ValueError('Effective admitted rank count differs')
    except Exception as exc:rank_errors.append(dict(commit=commit,error_type=type(exc).__name__,error=str(exc)[:200]))
    rank_counts[commit]=count
control200_observation={}
try:
    control_path=Path(CONTROL200['path'])
    if control_path.is_symlink() or control_path.parent!=Path(CONTROL200['run'])/'output/validation-cost':raise ValueError('Original dev200 control path differs')
    with control_path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if not 0<before.st_size<=100000:raise ValueError('Small dev200 control budget')
        control_raw=stream.read();after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Dev200 control changed during read')
    control_value=json.loads(control_raw);control_digest=hashlib.sha256(control_raw).hexdigest();totalbytes+=len(control_raw)
    if len(control_raw)!=CONTROL200['bytes'] or control_digest!=CONTROL200['sha256'] or control_value.get('status')!='VALIDATION_COST_COMPLETE' or control_value.get('actual_completed_outer')!=200 or control_value.get('ledger_sha256')!=CONTROL200['ledger_sha256']:raise ValueError('Original COMPLETE dev200 control bytes/ledger differ')
    control200_observation=dict(bytes=len(control_raw),sha256=control_digest,stable=True,matches_actual_input_pin=True,actual_completed_outer=200)
except Exception as exc:errors.append(dict(outer=200,kind='original_dev200_control',error_type=type(exc).__name__,error=str(exc)[:200]))
END=time.time();totalbytes+=rank_bytes
if totalbytes>5500000000:raise ValueError('Complete own JSON read budget exceeded')
summary=dict(schema='tree_trace_retention_remote_stream_v2',started_epoch=START,ended_epoch=END,collector_wall_seconds=END-START,collector_process_cpu_seconds=time.process_time()-CPU,explicit_JSON_bytes_read=totalbytes,rank_JSON_bytes_read=rank_bytes,remote_nice=os.getpriority(os.PRIO_PROCESS,0),remote_IO_priority='ionice idle class3 on this observer only',io_priority_result=io_priority_result,no_remote_files_written=True,no_tensors_or_training_imports=True,no_question_answer_or_full_trace_returned=True,effective_outers_requested=200,completed_record_inventories=len(results),candidate_records=allcandidate,leaf_records=allleaf,question_records=allquestion,binding_discrepancies=dict(allfail),candidate_required_field_missing=dict(allmissing),schema_fields={k:sorted(v) for k,v in schemas.items()},records=results,errors=errors,rank_inventory=dict(effective_rank_records=sum(rank_counts.values()),counts_by_commit=rank_counts,actual_effective_successful_calls_per_rank=rank_totals,outer_summaries=sorted(rank_rows,key=lambda row:row['outer']),zero_write_records_not_credited=zero_writes,historical_zero_sample_records=historical_zero_samples,historical_zero_sample_scope='Only original0bef outer1; no sampled parameter-change claim. All other effective outers still require nonzero sample.',nonrank_JSON_names_excluded=excluded_names,profiles_counted_as_rank_records=False,errors=rank_errors,scope='Exact tree_actor_rank JSON filenames, positive writes and unique effective outer/rank; no profile or PT read. Per-outer file-set digests bind all eight original rank JSON byte/SHA refs without transferring their payloads.'))
top_schemas=[]
for row in summary['records']:
    pair={key:row.pop(key) for key in ('rollout_top_fields','policy_top_fields')}
    if pair not in top_schemas:top_schemas.append(pair)
    row['top_schema_fieldset']=top_schemas.index(pair)
summary['top_schema_fieldsets']=top_schemas
summary['dev200_control']=control200_observation
encoded=json.dumps(summary,separators=(',',':'),allow_nan=False)
if len(encoded.encode())>315000:raise ValueError('Complete scalar remote summary exceeds small metadata budget; never truncate coverage')
print(encoded)
'''

def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(value):return (json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def locate(value):
    p=Path(value)
    assert p.parts and '..' not in p.parts
    p=(p if p.is_absolute() else ROOT/p if p.parts[0] in ('outputs','work') else BASE/p).resolve(strict=True)
    assert p.is_relative_to(ROOT) and p.suffix in ('.json','.py') and not p.name.startswith('ROOT_FULL200')
    return p
def bound(ref,expected=None,mirror=None):
    assert isinstance(ref,dict) and type(ref.get('bytes')) is int and 0<ref['bytes']<4000000 and digest(ref.get('sha256'))
    p=locate(ref.get('path',ref.get('file')))
    if expected is not None:assert p==locate(expected)
    raw=p.read_bytes();assert len(raw)==ref['bytes'] and sha(raw)==ref['sha256'],str(p)
    mirrored=p.is_relative_to(BASE) and not p.name.startswith(('NATIVE5_CONTINUE_POLL_','NATIVE5_CONTINUE_RESOURCE_')) if mirror is None else mirror
    if mirrored:
        for m in MIRRORS:assert (m/p.relative_to(BASE)).read_bytes()==raw,str(p)
    record=dict(path=str(p.relative_to(ROOT)),bytes=len(raw),sha256=sha(raw))
    assert str(p) not in INPUTS or INPUTS[str(p)]==(raw,record,mirrored)
    INPUTS[str(p)]=(raw,record,mirrored)
    return json.loads(raw) if p.suffix=='.json' else raw
def pin(p):
    p=locate(p);assert str(p) in INPUTS
    return INPUTS[str(p)][1]
def static_source(p,size,digest_value):return bound(dict(path=str(p),bytes=size,sha256=digest_value),mirror=False)
def check_inputs():
    for name,(raw,record,mirrored) in INPUTS.items():
        p=Path(name);assert p.read_bytes()==raw and sha(raw)==record['sha256'],name
        if mirrored:
            for m in MIRRORS:assert (m/p.relative_to(BASE)).read_bytes()==raw,name
def publish(name,raw):
    assert len(raw)<=358400 and name.startswith(NAME) and '/' not in name
    paths=[folder/name for folder in [BASE]+MIRRORS]
    assert all(folder.is_dir() for folder in [BASE]+MIRRORS)
    assert all(not path.exists() for path in paths),name
    for path in paths:
        with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    observations=[]
    for number in range(2):
        for path in paths:assert path.read_bytes()==raw,str(path)
        check_inputs()
        observations.append(dict(pass_number=number+1,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),copies=3,actual_bytes_SHA_equal=True))
        if number==0:time.sleep(1)
    return dict(file=name,bytes=len(raw),sha256=sha(raw),two_actual_readbacks=observations)
def main():
    if not __debug__:raise RuntimeError('Input gates cannot be disabled with -O')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pins',required=True,type=Path);parser.add_argument('--pins-bytes',required=True,type=int);parser.add_argument('--pins-sha256',required=True)
    args=parser.parse_args()
    manifest=bound(dict(path=str(args.pins),bytes=args.pins_bytes,sha256=args.pins_sha256))
    bound(manifest['source_command'],expected=Path(__file__).resolve(),mirror=False)
    assert manifest['schema']=='tree200_trace_retention_input_pins_v1' and manifest['status']=='ACTUAL_TRAIN200_NATIVE200_TRACE_INPUTS_READY' and manifest['outer']==200
    required={'native200','original_poll','original_resource','lineage_identity','dev200_control'}
    assert set(manifest['source_pins']) in (required,required|{'full200'})
    static_source(ROUTE,4411,ROUTE_SHA);static_source(CORE,39098,CORE_SHA)
    old_source=ROOT/'work/tree-speed-repair-20261008/tree171_trace_retention.py'
    static_source(old_source,30579,'ac8b0fca4b4a02275a2953605b2c10604e9fe6a3da4897739e86c038a07b1fb5')
    prior=bound(dict(path=str(BASE/'TREE_TRACE_RETENTION_THROUGH171_ACTUAL.json'),bytes=242392,sha256='1a394d6e657c66ff4ad2bcf9b1d7723ba49467d99139e0f5b504e87d921c1696'))
    assert prior['status']=='STRUCTURAL_CANDIDATE_RECORD_RETENTION_VERIFIED_THROUGH171' and prior['remote_ssh_exit_status']==0
    assert not prior['remote']['errors'] and not prior['remote']['binding_discrepancies'] and not prior['remote']['candidate_required_field_missing']
    docs={key:bound(ref) for key,ref in manifest['source_pins'].items()}
    native,poll,resource,compact,control=[docs[key] for key in ('native200','original_poll','original_resource','lineage_identity','dev200_control')]
    full=docs.get('full200')
    assert locate(manifest['source_pins']['native200']['path'])==BASE/'NATIVE_HISTORY_200_METADATA_READBACK.json'
    for key in ('original_poll','original_resource'):
        observed=locate(manifest['source_pins'][key]['path'])
        assert observed.parent==BASE and observed.name.startswith('NATIVE5_CONTINUE_'+('POLL_' if key=='original_poll' else 'RESOURCE_'))
    identity=native['identity'];assert identity['model_identity']==MODEL and identity['data_identity']==DATA
    spec=importlib.util.spec_from_file_location('tree200_actual_core_metadata_only',CORE);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    assert poll['source_commit']==COMMITS[2] and poll['progress']['actual_completed_outer_from_attempt_results']==200 and not poll['progress'].get('errors')
    assert poll['unit']==core.UNIT and poll['job_id']==core.JOB and poll['node']=='h100-10'
    ps,rs,active=core.validate_unit(poll,resource,200);core.validate_restore(poll)
    metric=core.validate_metric(poll,200);ranks=core.validate_ranks(poll,metric,200)
    record,guard,completed,totals=core.validate_native(native,poll['progress'],200)
    assert record['driver']['next_step']==201
    if metric is None:assert {row['attempt_id'] for row in ranks}=={guard['training_attempts']}
    final_phase=core.final_phase(poll,resource,active)
    run='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+COMMITS[2]
    assert poll['run']==run
    source=manifest['dev200_control_source'];control_ref=pin(manifest['source_pins']['dev200_control']['path'])
    assert source['stable_read'] is True and source['bytes']==control_ref['bytes'] and source['sha256']==control_ref['sha256']
    actual_dev=next(row for row in poll['development'] if row['outer']==200)
    assert Path(actual_dev['path']).name==actual_dev['path']
    assert source['path']==run+'/output/validation-cost/'+actual_dev['path']
    alias='/dataset/shaoshitong/search-r1-rl/tree-grpo/';retained='/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/'
    assert source['path'].startswith(alias) and source['resolved_path']==retained+source['path'][len(alias):]
    assert control['code_commit']==COMMITS[2] and control['config_sha256']==CONFIGS[2] and control['model_identity']==MODEL and control['data_identity']==DATA
    assert control['run'] in (run+'/output',str(Path(source['resolved_path']).parent.parent))
    assert control['job_id']=='not-slurm;systemd:'+COMMITS[2] and control['training_protocol']=='tree_fresh200_five_leaf_v1' and control['physical_gpus']==8
    assert control['status']==actual_dev['status']=='VALIDATION_COST_COMPLETE' and control['actual_completed_outer']==200 and control['phase']==actual_dev['phase']=='periodic'
    assert control['evaluated_rows']==control['dataset_rows']==actual_dev['rows']==512
    assert control['metrics']==actual_dev['metrics'] and control['ledger_sha256']==actual_dev['ledger_sha256'] and digest(control['ledger_sha256'])
    assert control['metrics']['val/qa_count/all']==512 and control['metrics']['val/qa_count/nq']==control['metrics']['val/qa_count/hotpotqa']==256
    control200=dict(run=run,path=source['path'],bytes=control_ref['bytes'],sha256=control_ref['sha256'],ledger_sha256=control['ledger_sha256'])
    if full is not None:
        assert locate(manifest['source_pins']['full200']['path'])==BASE/'FULL_OUTER200_PROGRESS_RECEIPT.json'
        for key in ('poll','resource','native_readback'):
            saved=manifest['source_pins']['native200' if key=='native_readback' else 'original_'+key];original=full['input_bindings'][key]
            assert locate(saved['path'])==locate(original['path']) and saved['bytes']==original['bytes'] and saved['sha256']==original['sha256']
        assert full['status'] in ('ACTUAL_TRAIN200_UPDATES_NATIVE_COMPLETE_FULL_METRIC_PENDING_FINAL_PHASE_OBSERVED','ACTUAL_FULL_OUTER200_NATIVE200_COMPLETE_FINAL_PHASE_OBSERVED')
        assert full['completed_outer']==200 and full['actual_next_outer_entered'] is None and full['native_driver_next_step']==201
        assert full['local_control_script_sha256']==CORE_SHA and full['source_commit']==COMMITS[2] and full['config_sha256']==CONFIGS[2]
        assert full['actual_training_pid']==503255 and full['actual_training_invocation_id']=='a313bda919834d24b4c5d491698d39ce'
        assert full['per_rank_successful_adam']==completed and full['per_rank_actual_total_adam']==totals
        assert full['last_rank_summaries']==ranks and full['iteration_metrics']==metric and full['final_phase_evidence']==final_phase
        assert full['actual_dev512']==actual_dev and full['native_manifest_sha256']==native['manifest_sha256'] and full['native_history_receipt_sha256']==native['receipt_sha256']
    training_gate=dict(gate_poll_observed_epoch=poll['epoch'],gate_instance_scope='Original SHA-bound poll/resource observations at their capture time, retained before any authorized slow-final handoff; no current live-process audit is inferred from these snapshots.',optional_FULL200=None if full is None else pin(manifest['source_pins']['full200']['path']),FULL200_required=False,native=pin(manifest['source_pins']['native200']['path']),original_poll=pin(manifest['source_pins']['original_poll']['path']),original_resource=pin(manifest['source_pins']['original_resource']['path']),full_train_metric_pending=metric is None,original_step200_train_metric_retained=metric is not None,actual_original_step200_metric=metric,actual_dev200_control=control_ref,actual_dev200_control_source=source,actual_dev200_metrics=control['metrics'],actual_dev200_inner_ledger_sha256=control['ledger_sha256'],actual_completed_outer=200,actual_total_adam_per_rank=totals,completed_successful_adam_per_rank=completed,partial_optimizer_calls_per_rank=guard['partial_optimizer_calls_per_rank'],native_saved_driver_next_step=201,actual_next_outer_entered=None,actual201_entry_required=False,final51713_completion_required=False,final_phase_evidence=final_phase)
    assert [Path(value).resolve() for value in compact['mirrors']]==MIRRORS
    identities={}
    for index,commit in enumerate(COMMITS):
        item=compact['source_identity_by_commit'][commit];ident=item['identity'];run=item['run_directory']
        expected='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+commit
        assert run==expected and ident['code_commit']==commit and ident['config_sha256']==CONFIGS[index] and ident['model_identity']==MODEL and ident['data_identity']==DATA
        identities[commit]=dict(run=run,config_sha256=CONFIGS[index],model_identity=MODEL,data_identity=DATA,effective_outers=[1,1] if index==0 else [2,5] if index==1 else [6,200])
    refs=manifest['rollout_metadata_pins'];assert isinstance(refs,list) and [ref['outer'] for ref in refs]==list(range(1,201))
    pins=[];local=[];fieldsets=[]
    for ref in refs:
        outer=ref['outer'];document=bound(ref);candidates=[row for row in document['rollouts'] if row['global_step']==outer];assert len(candidates)==1
        row=candidates[0];commit=COMMITS[0 if outer==1 else 1 if outer<=5 else 2];run=identities[commit]['run']
        assert document['run_directory']==run and type(row['audit_file_bytes']) is int and 0<row['audit_file_bytes']<=32000000
        audit=Path(row['audit_file']).name;match=re.fullmatch(r'step_(\d+)_rollout_(\d+)\.json',audit);assert match and int(match[1])==outer
        capture=None if outer in (1,2) else row.get('audit_file_sha256');assert outer in (1,2) or digest(capture)
        pins.append(dict(outer=outer,commit=commit,run=run,audit=audit,bytes=row['audit_file_bytes'],sha256=capture))
        fields=document.get('schema_observed',{}).get('candidate_fields',[])
        if fields not in fieldsets:fieldsets.append(fields)
        local.append(dict(outer=outer,**pin(ref['path']),candidate_schema_fieldset=fieldsets.index(fields),metadata_evidence_mode=document.get('evidence_mode','prior_compact_metadata')))
    frozen=[];rank_writer_shas={}
    for commit in COMMITS:
        for name in ('tree_audit.py','tree_nested_branch.py','controller.py','objectives.py','tree_rank_audit.py'):
            path='search_r1/baselines/'+name
            raw=subprocess.run(['git','-C',str(SOURCE),'show',commit+':'+path],check=True,capture_output=True).stdout
            frozen.append(dict(source_commit=commit,path=path,bytes=len(raw),sha256=sha(raw)))
            if name=='tree_rank_audit.py':rank_writer_shas[commit]=sha(raw)
    for original in prior['frozen_source_bindings']:assert original in frozen
    program=REMOTE.replace('__PINS__',repr(pins)).replace('__IDENTITIES__',repr(identities)).replace('__RANK_WRITER_SHAS__',repr(rank_writer_shas)).replace('__CONTROL200__',repr(control200))
    encoded=base64.b64encode(zlib.compress(program.encode(),9)).decode()
    command='env CUDA_VISIBLE_DEVICES= python3 -B -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
    spec=importlib.util.spec_from_file_location('tree200_existing_retained18_route',ROUTE);route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
    os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control';argv=route.route(command,'h100-10')
    transport=dict(program_utf8=len(program.encode()),target_command_utf8=len(command.encode()),management_command_utf8=len(argv[-1].encode()),all_local_argv_utf8=sum(len(value.encode())+1 for value in argv),encoding='One zlib9/base64 program+minimal pins copy; no uncompressed pin duplication, no stdin, no raw asset transfer.')
    assert transport['target_command_utf8']<=32768 and transport['management_command_utf8']<=65536 and transport['all_local_argv_utf8']<=80000,'Short transport bounds exceeded; no SSH is attempted'
    for suffix in ('.json','.md','_LOCAL_READBACK.json','_ATTEMPT.json'):assert all(not (folder/(NAME+suffix)).exists() for folder in [BASE]+MIRRORS)
    check_inputs();began=datetime.datetime.now(datetime.timezone.utc).isoformat()
    attempt=publish(NAME+'_ATTEMPT.json',enc(dict(schema='tree200_trace_retention_once_attempt_v1',started_utc=began,input_manifest=pin(args.pins),transport=transport,remote_program_sha256=sha(program.encode()),authorized_remote_calls=1,automatic_retry=False,training_operations=False)))
    response=None
    try:
        result=route.run(command,target='h100-10',timeout=600)
        response=publish(NAME+'_REMOTE_RESPONSE.json',enc(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,stdout_bytes=len(result.stdout.encode()),stdout_sha256=sha(result.stdout.encode()),stderr_bytes=len(result.stderr.encode()),stderr_sha256=sha(result.stderr.encode()),payload_scope='Complete original scalar-only remote response; no question_groups or raw trace payload.')))
        remote=json.loads(result.stdout) if result.returncode==0 else dict(status='CURRENT_REMOTE_RETENTION_UNKNOWN',returncode=result.returncode,stderr=result.stderr[-1200:])
        exit_code=result.returncode
    except (OSError,subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
        if response is None:
            partial_stdout=getattr(exc,'stdout',None) or '';partial_stderr=getattr(exc,'stderr',None) or ''
            if isinstance(partial_stdout,bytes):partial_stdout=partial_stdout.decode('utf-8')
            if isinstance(partial_stderr,bytes):partial_stderr=partial_stderr.decode('utf-8')
            response=publish(NAME+'_REMOTE_RESPONSE.json',enc(dict(returncode=None,error_type=type(exc).__name__,stdout=partial_stdout,stderr=partial_stderr,complete_response_received=False)))
        remote=dict(status='CURRENT_REMOTE_RETENTION_UNKNOWN',error_type=type(exc).__name__,error='Existing read-only retained-route call failed or timed out; no automatic retry.');exit_code=None
    rank=remote.get('rank_inventory') or {};rows=remote.get('records',[])
    success=(exit_code==0 and remote.get('completed_record_inventories')==200 and [row['outer'] for row in rows]==list(range(1,201)) and remote.get('question_records')==102400 and remote.get('leaf_records')==512000 and not remote.get('errors') and not remote.get('binding_discrepancies') and not remote.get('candidate_required_field_missing') and rank.get('effective_rank_records')==1600 and rank.get('counts_by_commit')=={COMMITS[0]:8,COMMITS[1]:32,COMMITS[2]:1560} and not rank.get('errors') and rank.get('actual_effective_successful_calls_per_rank')==completed and (remote.get('dev200_control') or {}).get('matches_actual_input_pin') is True and remote.get('remote_nice')==19 and (remote.get('io_priority_result') or {}).get('returncode')==0)
    oldrows=prior['remote']['records'];assert len(oldrows)==171
    old_hashes_match=bool(len(rows)>=171 and all((new['outer'],new['audit_sha256'],new['policy_sha256'])==(old['outer'],old['audit_sha256'],old['policy_sha256']) for new,old in zip(rows[:171],oldrows)))
    success=success and old_hashes_match
    coverage=list(range(1,201)) if success else []
    report=dict(schema='tree_trace_retention_through200_actual_v1',created_at_utc=began,status='STRUCTURAL_CANDIDATE_RECORD_RETENTION_VERIFIED_THROUGH200' if success else 'TREE200_TRACE_RETENTION_READBACK_FAILED_NO_FINAL_ACCEPTANCE',scope='Fresh full effective200 rollout/policy SHA and structural retention inventory plus admitted rank JSON metadata. This does not accept final51713 scores, PT/HF recovery or a complete content trace; no ROOTFULL200 dependency.',source_commit=COMMITS[2],source_config_sha256=CONFIGS[2],model_identity=MODEL,data_identity=DATA,actual_input_manifest=pin(args.pins),actual_training200_gate=training_gate,prior171_actual_audit=pin(BASE/'TREE_TRACE_RETENTION_THROUGH171_ACTUAL.json'),prior171_source=pin(old_source),prior171_evidence_preserved=True,fresh1_171_audit_and_policy_hashes_match_prior171=old_hashes_match,route=dict(helper=pin(ROUTE),management_control_path=os.environ['TREE_MGMT_CONTROL_PATH'],target='mgmt-host -> h100-10/10.33.0.18',remote_calls=1,retries=0,exit_status=exit_code,transport=transport,attempt=attempt,original_response=response),remote_program_sha256=sha(program.encode()),frozen_source_bindings=frozen,effective_lineage=[dict(source_commit=commit,outers=identities[commit]['effective_outers']) for commit in COMMITS],local_original_metadata_sources=local,historical_candidate_schema_fieldsets=fieldsets,remote=remote,coverage=dict(actual_stable_rollout_policy_and_parent_prefix_candidate_weight_bindings=coverage,current_capture_SHA_matches=list(range(3,201)) if success else [],prior_compact_capture_SHA_absent=[1,2],original_policy_writer_rollout_SHA_matches=coverage,full_token_action_vectors_or_tokenwise_weight_advantage_reconstruction=[]),weight_evidence_strength=prior['weight_evidence_strength'],historical_hash_evidence_strength='Selected metadata refs retain their actual evidence_mode; retrospective cost readback hashes are current readback pins, not invented historical capture-time SHA. Outer1/2 still lack original compact capture-time SHA. Original policy-writer/current rollout SHA equality and fresh1..171 equality with preserved prior171 are separately checked.',limits=prior['limits'][:4]+['This inventory reads ordinary JSON identity/structure/hash and scalar weights only; no complete token/action/raw-ledger vectors or per-token weights/advantages are reconstructed.','Rank model identity is inherited from the SHA-bound run/native/config gate; absent per-record fields are not invented. Profiles are explicitly excluded from effective rank counts.','Original0bef outer1 rank samples are finite but zero and do not prove sampled parameter change; actual successful Adam and finite/nonzero gradient records remain checked. Other outers retain the nonzero-sample gate.','Observer CPU/wall/JSON bytes are separate readback overhead and are not added to training or evaluation costs.'],cost_accounting=dict(observer_cpu_wall_bytes_only=True,added_to_training_or_service_cost=False),generator=dict(path=str(Path(__file__).resolve()),bytes=Path(__file__).stat().st_size,sha256=sha(Path(__file__).read_bytes())),training_operations_performed=False,PT_HF_tensor_or_model_read=False,remote_files_written=False,final51713_acceptance_claimed=False,ROOTFULL200_used_as_input=False,goal_complete=False,mirrors=list(map(str,MIRRORS)))
    check_inputs();raw=enc(report);assert len(raw)<=358400,'Complete audit exceeds small local summary budget; coverage is never truncated'
    md=('# Tree trace retention through effective200\n\nStatus: '+report['status']+'\n\nFresh full200 original rollout/policy SHA and structure inventory; expected102400 question instances and512000 selected leaves. Actual admitted rank records expected1560 child+40 inherited; profiles excluded. Actual Adam follows original successful/partial counters, without a4000 assumption. Full training metric may still be pending behind original slow51713; no actual201 or ROOTFULL200 dependency.\n\nOrdinary JSON preserves structural parent/prefix/action-start hashes, candidate acceptance/rejection/reasons/fallback and leaf .2/row-weight sums. It does not reconstruct full token/action/raw-ledger vectors or tokenwise weights/advantages. Outer1/2 capture-time SHA remains absent; policy-writer/current-hash evidence is separately recorded.\n\nObserver CPU/wall/bytes and all actual per-outer/hash results are in JSON; no training/evaluation cost is added and no PT/HF is read. Old171 evidence remains unchanged.\n').encode()
    products=[publish(NAME+'.json',raw),publish(NAME+'.md',md)]
    receipt=publish(NAME+'_LOCAL_READBACK.json',enc(dict(schema='tree_trace_retention200_local_readback_v1',products=products,input_files=len(INPUTS),input_byte_readbacks=2,prior171_preserved=True,success=success)))
    print(json.dumps(dict(status=report['status'],products=products,local_readback_receipt=receipt,actual200_structural_trace_retention_verified=success,remote_exit_status=exit_code,remote_calls=1,automatic_retry=False),ensure_ascii=False))
    if not success:raise SystemExit(1)
if __name__=='__main__':main()

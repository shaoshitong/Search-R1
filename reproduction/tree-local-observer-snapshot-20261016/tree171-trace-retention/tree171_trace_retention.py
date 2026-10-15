#!/usr/bin/env python3
"""Read-only retention inventory of the 171 explicit effective tree JSON pairs.

Never imports training code, loads tensors, reads text answers, starts services,
or writes remotely. One retained-route call; no retry or alternative route.
Remote JSON is processed one indented record at a time at idle IO priority.
Only small counts, schemas, identities, paths and hashes return to the Mac.
"""
import datetime
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/tree-nested-branch-handoff-20261013'
NAME = 'TREE_TRACE_RETENTION_THROUGH171_ACTUAL'
SOURCE = ROOT / 'work/tree-speed-repair-20261008/nested-native5-recovery-source'
ROUTE = ROOT / 'work/tree-speed-repair-20261008/ssh_retained18.py'
ROUTE_SHA = '8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'
COMMITS = ['0befbaf99fc44942105831b3fbec94e4bf1b8ccb', 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4', '778b2b43dad90fc0c1c5d95d77c2326286011b73']

REMOTE = r'''
import os,json,time,hashlib,math,shutil,subprocess
from pathlib import Path
from collections import Counter,defaultdict
os.nice(19)
io_priority_result=None
if shutil.which('ionice'):
    ip=subprocess.run(['ionice','-c','3','-p',str(os.getpid())],capture_output=True,text=True)
    io_priority_result=dict(returncode=ip.returncode,stderr=ip.stderr[-200:])
PINS = __PINS__
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
    try:
        expected_budget=pin['bytes']+4_000_000
        if totalbytes+expected_budget>5_500_000_000:raise ValueError('Explicit audit JSON IO budget reached')
        if p.stat().st_size>32_000_000 or q.stat().st_size>4_000_000:raise ValueError('Audit size exceeds fixed retention inventory budget')
        leaves={};branches={};accepted={};candidate_ids=set();fail=Counter();missing=Counter();stages=Counter();reasons=Counter();fallback=Counter();depths=Counter();prefixorigins=Counter();fields=defaultdict(set);rowcounts=Counter();questiongroups={};candidatecount=0
        def consume(section,v):
            nonlocal_dummy=None
            fields[section].update(v);rowcounts[section]+=1
            if section=='branches':
                h=v.get('child_trajectory_hash');branches[h]=v
            elif section=='leaves':leaves[v['trajectory_hash']]=v
            elif section=='question_groups':questiongroups[v['question_hash']]=v
            elif section=='candidate_attempts':
                for k in required:
                    if k not in v:missing[k]+=1
                h=fp(v['child_trajectory_id'])
                if h in candidate_ids:fail['duplicate_candidate_child']+=1
                candidate_ids.add(h);stages[v.get('stage')]+=1;reasons[v.get('reason')]+=1;depths[str(v.get('stage'))+':'+str(v.get('action_depth'))]+=1
                fallback[str(v.get('stage'))]+=int(bool(v.get('fallback')))
                if v.get('stage')=='B2':prefixorigins[str(v.get('b2_prefix_origin'))]+=1
                if v.get('accepted')!=v.get('training_leaf'):fail['accepted_training_leaf_flag']+=1
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
        if len(accepted)!=1536 or len(leaves)!=2560 or len(grouped)!=512 or len(branches)!=1536:fail['expected_record_counts']+=1
        if len(accepted)+rowcounts['candidate_attempts']-len(accepted)!=len(candidate_ids):fail['candidate_count_identity']+=1
        if set(accepted)-set(leaves):fail['accepted_candidate_missing_training_leaf']+=1
        policyrows={};policyfields=set();policyfail=Counter()
        def consume_policy(section,v):
            if section=='rows':
                policyfields.update(v);h=v['trajectory_hash'];policyrows[h]=v
                previous=leaves.get(h)
                if previous is None or v.get('tree_hash')!=previous.get('tree_hash') or v.get('question_hash')!=previous.get('question_hash'):policyfail['policy_leaf_binding']+=1
                expected=1. if v.get('effective_tokens',0)>0 else 0.
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
END=time.time()
print(json.dumps(dict(schema='tree_trace_retention_remote_stream_v1',started_epoch=START,ended_epoch=END,collector_wall_seconds=END-START,collector_process_cpu_seconds=time.process_time()-CPU,explicit_JSON_bytes_read=totalbytes,remote_nice=19,remote_IO_priority='ionice idle class3 when available',io_priority_result=io_priority_result,no_remote_files_written=True,no_tensors_or_training_imports=True,no_question_answer_or_full_trace_returned=True,effective_outers_requested=171,completed_record_inventories=len(results),candidate_records=allcandidate,leaf_records=allleaf,question_records=allquestion,binding_discrepancies=dict(allfail),candidate_required_field_missing=dict(allmissing),schema_fields={k:sorted(v) for k,v in schemas.items()},records=results,errors=errors),separators=(',',':'),allow_nan=False))
'''

def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-short-command-authorized',action='store_true',help='Root authorized second short-code SSH after original oversized mux failure; only replaces this task unfrozen drafts.')
    args=parser.parse_args()
    draft=BASE/(NAME+'.json')
    if draft.exists() and not args.retry_short_command_authorized:raise ValueError('An SSH attempt/draft already exists; no implicit retry permitted')
    preceding=json.loads(draft.read_text()) if draft.exists() else None
    if sha(ROUTE.read_bytes())!=ROUTE_SHA:raise ValueError('Retained route SHA changed')
    compact=json.loads((BASE/'PER_OUTER_METRICS_COMPACT.json').read_text())
    pins=[];local=[]
    for outer in range(1,172):
        name='PARENT0BEF_CANDIDATE_COST_COMPACT.json' if outer==1 else 'CANDIDATE_COST_COMPACT.json' if outer==2 else f'candidate_cost_backfill/CANDIDATE_OUTER{outer:04d}.json'
        raw=(BASE/name).read_bytes();d=json.loads(raw);r=next(v for v in d['rollouts'] if v['global_step']==outer)
        commit=COMMITS[0 if outer==1 else 1 if outer<=5 else 2]
        expectedrun=compact['source_identity_by_commit'][commit]['run_directory']
        if d['run_directory']!=expectedrun:raise ValueError(f'Effective run identity mismatch at {outer}')
        pins.append(dict(outer=outer,commit=commit,run=expectedrun,audit=Path(r['audit_file']).name,bytes=r['audit_file_bytes'],sha256=r.get('audit_file_sha256')))
        local.append(dict(outer=outer,path=name,bytes=len(raw),sha256=sha(raw),candidate_fields_observed=d.get('schema_observed',{}).get('candidate_fields',[])))
    frozen=[]
    for commit in COMMITS:
        for path in ['search_r1/baselines/tree_audit.py','search_r1/baselines/tree_nested_branch.py','search_r1/baselines/controller.py','search_r1/baselines/objectives.py']:
            raw=subprocess.run(['git','-C',str(SOURCE),'show',commit+':'+path],check=True,capture_output=True).stdout
            frozen.append(dict(source_commit=commit,path=path,bytes=len(raw),sha256=sha(raw)))
    program=REMOTE.replace('__PINS__',repr(pins))
    spec=importlib.util.spec_from_file_location('retained18',ROUTE);route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
    os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
    encoded=base64.b64encode(zlib.compress(program.encode(),9)).decode()
    command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
    began=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=route.run(command,target='h100-10',timeout=600)
    remote=json.loads(result.stdout) if result.returncode==0 else dict(status='CURRENT_REMOTE_RETENTION_UNKNOWN',returncode=result.returncode,stderr=result.stderr[-1200:])
    mirrors=[Path(v) for v in compact['mirrors']]
    report=dict(schema='tree_trace_retention_through171_actual_v1',created_at_utc=began,scope='Actual record-retention inventory, not code review, training tests, launch gate, independent tensor verification, or final200 acceptance.',training_must_continue=True,training_operations_performed=False,route=dict(path=str(ROUTE),sha256=ROUTE_SHA,management_control_path=os.environ['TREE_MGMT_CONTROL_PATH'],target='mgmt-host -> h100-10/10.33.0.18',remote_calls=1,retries=0),source_commit=COMMITS[2],frozen_source_bindings=frozen,effective_lineage=[dict(source_commit=COMMITS[0],outers=[1]),dict(source_commit=COMMITS[1],outers=[2,3,4,5]),dict(source_commit=COMMITS[2],outers=list(range(6,172)))],run_directories={k:v['run_directory'] for k,v in compact['source_identity_by_commit'].items()},local_original_metadata_sources=local,remote_program_sha256=sha(program.encode()),remote=remote,definitions=dict(actual_tree='Each question has A,A1 sharing root0; B,B1,B2 sharing root1. A1 event parent=A; B1 event parent=B; B2 event parent=B1. B2 actual snapshot may originate from inherited B prefix or new B1 suffix, separately recorded.',branch_position='action_start_id,action_depth,actual_prefix_trajectory_id,backoff_actions,remaining_turns; candidate event parent is distinct from actual snapshot owner.',shared_prefix='Persisted branch parent_fields/child_fields cryptographic signatures, inherited_prefix_sha256 and inherited_raw_ledger_sha256, lengths and inheritance result. This preserves structure and hash bindings, not full prefix token arrays.',candidate_records='Every candidate_attempts record retains sampled_reward,parent_reward,original_root_reward,accepted,training_leaf,reason,stop_reason,fallback,initial_state_attempts and backoff metadata, including rejected candidates.',weights='Rollout leaves record relative_leaf_mass=.2, quotas2/3 and root masses. policy_batch rows retain trajectory/tree/question hashes,effective_tokens and policy_weight_sum; writer originally checked per-effective-token1/T and exact response/mask/step bindings. Nonempty policy row mass1, empty row mass0. Token-wise weights and advantages are not persisted here.',publication='TreeAudit.publish fsyncs a temporary JSON and atomically links it to an immutable pathname without replacing prior evidence.'),limits=['Scalar inherited_prefix_verified_fraction1.0 and configured leaf weights do not substitute for full trace retention.','No original token/action text, complete response token vectors, token-wise policy weights, token-wise advantages or full behavior raw ledger are retained by the ordinary tree-audit JSON schema. Current state/checkpoints were not opened to infer their availability.','Outer1/2 historical candidate compact summaries lack original audit SHA; this observation can obtain current audit SHA and policy-to-rollout binding but cannot retroactively supply a historical capture-time SHA.','Reading persisted hash bindings does not replay model generation or independently prove semantic token equality.','This is through effective171 only;172/controller progress and final200 acceptance belong to the active observer.'],generator=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes())),mirrors=[str(v) for v in mirrors])
    report['route']['payload_bytes']=dict(program_utf8=len(program.encode()),target_command_utf8=len(command.encode()),management_command_utf8=len(route.route(command,'h100-10')[-1].encode()),transport='one zlib/base64 code copy; no model, data or raw rollout transfer')
    if preceding is not None:
        report['prior_failed_ssh_attempt']=dict(result=preceding['remote'],draft_bytes=draft.stat().st_size,draft_sha256=sha(draft.read_bytes()),program_utf8_bytes=70853,target_command_utf8_bytes=264648,management_command_utf8_bytes=611851,local_argv_utf8_bytes=612098,stdin_used=False,cause='Unknown. Oversized repeated exec payload is a possible mux boundary; this does not establish network or training failure.',root_authorized_short_code_retry=True)
        report['route'].update(remote_calls=2,retries=1)
    raw=(json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    rr=remote.get('records',[]);covered=[v['outer'] for v in rr if not v['binding_discrepancies'] and not v['candidate_required_field_missing']]
    md=(f"# Tree trace retention through effective outer171\n\n记录保留核对；不是代码 review、训练测试、启动门禁或最终200验收。训练未操作，必须继续。\n\n单次原18只读流式核对完成 {len(rr)}/171 个实际 rollout/policy JSON 对，绑定核对无缺口的 outer 为 {covered[0] if covered else 'unknown'}—{covered[-1] if covered else 'unknown'}（完整列表见JSON）。读取 {remote.get('explicit_JSON_bytes_read','unknown')} 字节；仅小摘要返回本地。逐候选记录总数 {remote.get('candidate_records','unknown')}；逐题记录 {remote.get('question_records','unknown')}；叶子记录 {remote.get('leaf_records','unknown')}。差异：{remote.get('binding_discrepancies','unknown')}；缺字段：{remote.get('candidate_required_field_missing','unknown')}；读取异常：{remote.get('errors',remote.get('status','unknown'))}。\n\n逐题 A→A1、B→B1→B2 的 event-parent/ancestor、B2 snapshot origin、分叉 action depth/start、共享前缀/ledger 哈希、逐候选 reward/接受或拒绝原因/回退信息均在远端 `RUN/output/tree-audit/step_N_rollout_R.json`。policy 绑定在同目录 `_policy_batch.json`，写入时已按原 writer 检查1/T，当前保留每叶有效token数与权重和。具体路径、字节、SHA和每outer结果见JSON；原有candidate compact摘要只保存聚合，不能替代这些实际远端记录。\n\n普通tree-audit保存的是身份、结构、哈希和计数，不含完整token/action内容、逐token权重/advantage或完整raw ledger。不能从 `inherited_prefix_verified_fraction=1.0` 或叶权重配置宣称完整内容trace已持久保存。outer1/2旧摘要没有capture-time audit SHA，本次SHA不能补成当时历史SHA。未读取checkpoint/tensor，也没有推断其中是否另含轨迹。\n\n冻结源码：778b2b43dad90fc0c1c5d95d77c2326286011b73；历史有效谱系outer1=0bef、outer2—5=ba9、outer6—171=778。\n").encode()
    receipts=[]
    for name,data in [(NAME+'.json',raw),(NAME+'.md',md)]:
        paths=[BASE/name]+[v/name for v in mirrors]
        if any(p.exists() for p in paths) and not args.retry_short_command_authorized:raise ValueError('Refuse replacing existing retention evidence')
        for p in paths:
            if not p.parent.is_dir():raise ValueError('Existing mirror absent')
            with p.open('wb' if args.retry_short_command_authorized else 'xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        reads=[]
        for observation in range(2):
            reads.append([dict(path=str(p),bytes=len(p.read_bytes()),sha256=sha(p.read_bytes()),byte_equal=p.read_bytes()==data) for p in paths])
        receipts.append(dict(file=name,bytes=len(data),sha256=sha(data),two_byte_readbacks=reads))
    receipt=(json.dumps(dict(schema='tree_trace_retention_local_readback_v1',products=receipts),ensure_ascii=False,separators=(',',':'))+'\n').encode()
    receiptname=NAME+'_LOCAL_READBACK.json'
    for d in [BASE]+mirrors:(d/receiptname).write_bytes(receipt)
    print(json.dumps(dict(products=[{k:v for k,v in r.items() if k!='two_byte_readbacks'} for r in receipts],readback_receipt=receiptname,remote_inventories=len(rr),binding_discrepancies=remote.get('binding_discrepancies'),missing=remote.get('candidate_required_field_missing'),errors=remote.get('errors'),remote_status=remote.get('status')),ensure_ascii=False))

def finalize_local_report():
    """Seal the successful observation locally; this path makes no SSH call."""
    p=BASE/(NAME+'.json');report=json.loads(p.read_bytes());remote=report['remote']
    if remote.get('completed_record_inventories')!=171 or remote.get('errors'):
        raise ValueError('No complete existing remote observation to finalize')
    fieldsets=[]
    for source in report['local_original_metadata_sources']:
        fields=source.pop('candidate_fields_observed')
        if fields not in fieldsets:fieldsets.append(fields)
        source['candidate_schema_fieldset']=fieldsets.index(fields)
    report['historical_candidate_schema_fieldsets']=fieldsets
    top_schemas=[]
    for row in remote['records']:
        pair={key:row.pop(key) for key in ['rollout_top_fields','policy_top_fields']}
        if pair not in top_schemas:top_schemas.append(pair)
        row['top_schema_fieldset']=top_schemas.index(pair)
    report['remote_top_schema_fieldsets']=top_schemas
    report['remote_ssh_exit_status']=0
    report['status']='STRUCTURAL_CANDIDATE_RECORD_RETENTION_VERIFIED_THROUGH171'
    report['coverage']={
        'actual_stable_rollout_and_policy_pairs':list(range(1,172)),
        'actual_each_question_A_A1_B_B1_B2_parent_ancestor_bindings':list(range(1,172)),
        'actual_every_candidate_reward_reason_rejection_fallback_required_fields':list(range(1,172)),
        'actual_branch_snapshot_prefix_and_raw_ledger_hash_bindings':list(range(1,172)),
        'actual_leaf_relative_mass_and_policy_row_weight_sums':list(range(1,172)),
        'current_audit_SHA_matches_prior_compact_capture_SHA':list(range(3,172)),
        'prior_compact_capture_SHA_absent':[1,2],
        'original_policy_writer_rollout_SHA_matches_current_audit_SHA':list(range(1,172)),
        'full_token_action_content_trace_in_ordinary_tree_audit_JSON':[],
        'independent_tokenwise_weight_advantage_or_generation_replay':[],
    }
    report['weight_evidence_strength']='Actual retained leaf relative mass=.2 and policy row weight sums verified for every row. Original writer checks uniform1/T and exact tensor hashes before publishing; this inventory does not read tokenwise tensors or independently re-execute that original check.'
    report['historical_hash_evidence_strength']='Prior compact capture-time audit SHA exists and matches for outer3..171. Outer1/2 have no such prior compact SHA; their original immutable policy writer recorded a rollout SHA, and current stream verified that binding. This is original-writer evidence, not a retroactively invented earlier independent capture.'
    report['limits'][2]=report['historical_hash_evidence_strength']
    report['finalization']=dict(local_only=True,additional_remote_calls=0,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['generator']['sha256']=sha(Path(__file__).read_bytes())
    report['source_frozen']=True
    jsondata=(json.dumps(report,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    last=remote['records'][-1]
    md=(
        '# Tree trace retention through effective outer171\n\n'
        '已实际核对171/171组远端rollout与policy JSON；训练未操作并继续。此为记录保留核对，不是代码review、训练测试、启动门禁或最终200验收。\n\n'
        f"effective1—171每轮512题、2560叶、1536条接受分叉；合计87552题次、437760叶、{remote['candidate_records']}逐候选记录。每题实际A→A1、B→B1→B2父子与ancestor链、B2 snapshot来源、分叉action start/depth、共享prefix/raw-ledger哈希、接受/拒绝与训练叶绑定均核对；required reward/reason/fallback字段缺失0，绑定差异0，读取异常0。\n\n"
        '结构与逐候选记录在 `RUN/output/tree-audit/step_N_rollout_R.json`；policy绑定在同目录 `_policy_batch.json`。JSON列出三个RUN、每outer文件名、字节和SHA。每叶relative mass=.2及policy row权重和已实际核对；原writer发布前检查1/T及response/mask/step绑定，当前保留的是哈希和权重和。\n\n'
        f"outer171 rollout：`step_171_rollout_166.json`，{last['audit_bytes']}B，SHA256 `{last['audit_sha256']}`；policy：{last['policy_bytes']}B，SHA256 `{last['policy_sha256']}`。outer3—171均匹配先前compact原SHA；outer1/2旧compact无capture SHA，但本次核对原policy writer的rollout SHA绑定全部相等。\n\n"
        '**保留范围限制：**普通tree-audit未保存完整token/action内容、完整raw ledger、逐token权重或advantages。其身份/结构/哈希记录不能冒称完整内容trace；scalar `inherited_prefix_verified_fraction=1.0` 或叶权重配置不能替代全trace。未读取tensor/checkpoint，也未推断它们是否另存内容。\n\n'
        f"实际读取JSON {remote['explicit_JSON_bytes_read']}B；CPU {remote['collector_process_cpu_seconds']:.3f}s；wall {remote['collector_wall_seconds']:.3f}s；nice19、自身idle I/O设置exit0。仅小聚合回传，无远端写入。首次mux255错误保留，不能归因网络/训练；exec参数由611851B压缩为20306B，仅一次获授权短code核对成功exit0，未再次重试。\n\n"
        '冻结训练源码778b2b43；有效谱系outer1=0bef、outer2—5=ba9、outer6—171=778。172与200终验由主线observer继续。\n'
    ).encode()
    directories=[BASE]+[Path(v) for v in report['mirrors']]
    products=[]
    for name,data in [(NAME+'.json',jsondata),(NAME+'.md',md)]:
        paths=[d/name for d in directories]
        for path in paths:
            with path.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        observations=[]
        for cycle in range(2):
            cycle_rows=[]
            for path in paths:
                got=path.read_bytes()
                if got!=data:raise ValueError('Final local byte readback mismatch')
                cycle_rows.append(dict(path=str(path),bytes=len(got),sha256=sha(got),byte_equal=True))
            observations.append(cycle_rows)
        products.append(dict(file=name,bytes=len(data),sha256=sha(data),two_byte_readbacks=observations))
    receiptdata=(json.dumps(dict(schema='tree_trace_retention_local_readback_v2',source_frozen=report['generator'],finalized_utc=report['finalization']['created_at_utc'],products=products),ensure_ascii=False,separators=(',',':'))+'\n').encode()
    for directory in directories:
        with (directory/(NAME+'_LOCAL_READBACK.json')).open('wb') as f:f.write(receiptdata);f.flush();os.fsync(f.fileno())
    for cycle in range(2):
        for directory in directories:
            if (directory/(NAME+'_LOCAL_READBACK.json')).read_bytes()!=receiptdata:raise ValueError('Receipt bytes mismatch')
    print(json.dumps(dict(source=report['generator'],products=[{k:v for k,v in r.items() if k!='two_byte_readbacks'} for r in products],receipt=dict(file=NAME+'_LOCAL_READBACK.json',bytes=len(receiptdata),sha256=sha(receiptdata)),three_locations_two_reads_byte_equal=True,remote_ssh_exit_status=0,additional_remote_calls=0),ensure_ascii=False))

if __name__=='__main__':
    if sys.argv[1:]==['--finalize-local-only']:finalize_local_report()
    else:main()

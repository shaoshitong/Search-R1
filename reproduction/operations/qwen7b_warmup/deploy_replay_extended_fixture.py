"""Prefix-only extended replay fixture; immutable Q1 code/model are REUSED.

Only prepare/status/explicit guarded submit are exposed. Prepare reads small
sealed batch/prefix/native-manifest metadata on HPC3, never full rollouts/EM or
model/feature tensors. The old replay job verifies actual model/native file SHA.
No remote action executes on import or --self-test. No old source/model is copied.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shlex
import subprocess

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
Q1=PARENT+'/replay-layer-qualification-20261021'
Q2=PARENT+'/replay-extended-qualification-20261021'
RUNTIME_COMMIT='18c46affb9c6c69330aa19eb1e00124e1f39d019'
CODE_ROOT=Q1+'/code-18c46affb9c6'
MODEL_VIEW=Q1+'/official-snapshot-view'
JOB_NAME='q7b-replay-extqual'
SELECTION_RULE=('first-declared-sealed-batch-of-first4-sorted-sealed-shards; '
                'complete-authentic-G5-only; anchors=most-multisegment,largest-later-context-gap,'
                'longest-prompt,shortest-prompt,short-terminal-policy,longest-causal-context; '
                'then-SHA256(replay-extended-v1:NUL:qid); capped24questions; no-full-or-EM-read')


def choose_questions(records,limit=24):
    """Pure label-blind selector over already normalized prefix metadata."""
    if type(limit) is not int or not 1<=limit<=24:raise ValueError('At most24 complete questions')
    grouped={};seen=set()
    for row in records:
        tid=row['trajectory_id'];qid=row['qid']
        if type(row['response_index']) is not int or not 0<=row['response_index']<5:
            raise ValueError('Authentic integer G5 response index required')
        if tid in seen:raise ValueError('Duplicate authentic trajectory identity')
        seen.add(tid);grouped.setdefault(qid,[]).append(row)
    candidates={}
    for qid,own in grouped.items():
        own=sorted(own,key=lambda r:r['response_index'])
        if [r['response_index'] for r in own]!=list(range(5)):
            raise ValueError('Every considered sealed question must contain own completeG5')
        if len({r['prompt_length'] for r in own})!=1:
            raise ValueError('Original prompt length changed within authentic G5')
        candidates[qid]=dict(qid=qid,rows=own,prompt_length=own[0]['prompt_length'],
             multisegment=sum(r['segment_count']>1 for r in own),
             later_context_gap=sum(r['later_context_gap_tokens']>0 for r in own),
             max_gap=max(r['later_context_gap_tokens'] for r in own),
             min_policy=min(r['policy_length'] for r in own),
             max_context=max(r['max_context_length'] for r in own))
    if not candidates:raise ValueError('No authentic complete G5 prefix questions')
    def hashed(qid):return hashlib.sha256(('replay-extended-v1\0'+qid).encode()).hexdigest()
    chosen=[];anchors=[]
    def add(pool,key,reason):
        if not pool:return
        qid=sorted(pool,key=lambda q:(key(candidates[q]),hashed(q)))[0]
        if qid not in chosen and len(chosen)<limit:chosen.append(qid);anchors.append(dict(qid=qid,reason=reason))
    add([q for q,v in candidates.items() if v['multisegment']],lambda v:-v['multisegment'],'most_multisegment_prefix_trajectories')
    add([q for q,v in candidates.items() if v['later_context_gap']],lambda v:-v['max_gap'],'largest_observed_later_causal_context_gap')
    add(list(candidates),lambda v:-v['prompt_length'],'longest_initial_prompt')
    add(list(candidates),lambda v:v['prompt_length'],'shortest_initial_prompt')
    add([q for q,v in candidates.items() if v['min_policy']<20],lambda v:v['min_policy'],'short_terminal_policy_prefix')
    add(list(candidates),lambda v:-v['max_context'],'longest_actual_causal_context')
    for qid in sorted(candidates,key=hashed):
        if len(chosen)>=limit:break
        if qid not in chosen:chosen.append(qid)
    selected=[row for qid in chosen for row in candidates[qid]['rows']]
    def coverage(rows):
        prompts=[r['prompt_length'] for r in rows]
        return dict(questions=len({r['qid'] for r in rows}),trajectories=len(rows),
             multisegment_trajectories=sum(r['segment_count']>1 for r in rows),
             later_turn_trajectories=sum(r['max_turn_index']>0 for r in rows),
             positive_later_context_gap_trajectories=sum(r['later_context_gap_tokens']>0 for r in rows),
             prompt_length_min=min(prompts),prompt_length_max=max(prompts),
             short_terminal_policy_trajectories=sum(r['policy_length']<20 for r in rows),
             max_actual_causal_context_tokens=max(r['max_context_length'] for r in rows))
    actual=coverage(selected)
    report=dict(selection_rule=SELECTION_RULE,label_blind=True,anchors=anchors,
                available=coverage(records),selected=actual,
                multisegment_present=actual['multisegment_trajectories']>0,
                additional_later_context_present=actual['positive_later_context_gap_trajectories']>0,
                semantic_tool_calls_confirmed=False,
                tool_coverage_claim='No semantic tool-call claim: only saved causal token/turn metadata inspected',
                full_rollouts_read=False,own_EM_read=False)
    return selected,report


def render_sbatch(base,selection_sha256,job_name=JOB_NAME):
    if str(base)!=Q2:raise ValueError('Use the separate declared Q2 directory')
    if len(selection_sha256)!=64 or any(c not in '0123456789abcdef' for c in selection_sha256):
        raise ValueError('Actual selection SHA256 required')
    if not job_name or any(not(c.isalnum() or c in '-_') for c in job_name):
        raise ValueError('Safe isolated job name required')
    script='''#!/usr/bin/env bash
#SBATCH --job-name=@@JOB@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/logs/replay-%j.log
set -euo pipefail
REPLAY_BASE=@@BASE_Q@@
REPLAY_CODE_ROOT=@@CODE_Q@@
REPLAY_MODEL_VIEW=@@MODEL_Q@@
exec 9>"$REPLAY_BASE/replay-run.lock"
flock -n 9
export QWEN7B_CODE_ROOT="$REPLAY_CODE_ROOT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 - "$REPLAY_BASE/source-publication.json" "$REPLAY_CODE_ROOT" <<'VERIFY_CODE'
import hashlib,json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text());root=pathlib.Path(sys.argv[2])
assert p['code_root']==str(root)
for name,item in p['files'].items():
 path=root/name;assert path.stat().st_size==item['bytes']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],name
VERIFY_CODE
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.replay_probe_layers --selection "$REPLAY_BASE/inputs/fixture-selection.json" --selection-sha256 @@SHA@@ --model "$REPLAY_MODEL_VIEW" --output "$REPLAY_BASE/qualification-output" --max-trajectories 120 --device cuda:0
'''
    for key,value in {'@@JOB@@':job_name,'@@BASE@@':Q2,'@@BASE_Q@@':shlex.quote(Q2),
                      '@@CODE_Q@@':shlex.quote(CODE_ROOT),'@@MODEL_Q@@':shlex.quote(MODEL_VIEW),
                      '@@SHA@@':selection_sha256}.items():script=script.replace(key,value)
    if '@@' in script:raise ValueError('Unrendered shell token')
    return script


COMMON=r'''
import pathlib,json,hashlib,os,fcntl,time,subprocess,gzip,sys,shlex
sys.dont_write_bytecode=True
parent=pathlib.Path(inputs['parent']);q1=pathlib.Path(inputs['q1']);b=pathlib.Path(inputs['q2'])
code=pathlib.Path(inputs['code_root']);view=pathlib.Path(inputs['model_view'])
def read(p):return json.loads(pathlib.Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def atomic(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(item):
 actual=fp(item['path']);assert actual['sha256']==item['sha256']
 if 'bytes' in item:assert actual['bytes']==item['bytes']
 return pathlib.Path(item['path'])
def command(args,timeout=25):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert r.returncode==0,(args,r.stderr[-2000:]);return r.stdout.strip()
def lock():
 h=(b/'controller.lock').open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def unsubmitted():
 assert not (b/'job.id').exists(),'Existing job identity; inspect instead of duplicating'
 assert not (b/'submission-intent.json').exists(),'Unknown/previous submission: reconcile before retry'
def q1_binding():
 pub=read(q1/'source-publication.json')
 assert pub['commit']==inputs['runtime_commit'] and pub['code_root']==str(code)
 for name,item in pub['files'].items():assert fp(code/name)==dict(item,path=str(code/name)),name
 prepared=read(q1/'preparation.json');selection_path=verify(prepared['selection']);selection=read(selection_path)
 assert prepared['code_root']==str(code) and prepared['model_view']==str(view)
 complete=read(q1/'qualification-output/complete.json')
 assert complete['status']=='BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE'
 assert complete['source_selection']==fp(selection_path) and complete['actor_optimizer_updates']==0 and complete['actor_frozen']
 assert complete['code']['sha256']==pub['files']['reproduction/operations/qwen7b_warmup/replay_probe_layers.py']['sha256']
 for name,item in selection['model_files'].items():
  assert complete['model_files'][name]['sha256']==item['sha256'] and complete['model_files'][name]['bytes']==item['bytes']
  path=view/name;assert path.stat().st_size==item['bytes']
  if not name.endswith('.safetensors'):assert sha(path)==item['sha256']
 return pub,prepared,selection,complete
def verify_runtime():
 pub=read(b/'source-publication.json');old,prepared,selection,complete=q1_binding()
 verify(pub['source_q1_publication']);verify(pub['source_q1_preparation']);verify(pub['source_q1_complete'])
 assert pub['commit']==old['commit'] and pub['code_root']==old['code_root'] and pub['files']==old['files']
 return pub,selection
'''

INIT=r'''
old,prepared,selection,complete=q1_binding()
b.mkdir(parents=True,exist_ok=True);guard=lock();unsubmitted()
for name in ['inputs','logs']:(b/name).mkdir(exist_ok=True)
pub=dict(commit=old['commit'],remote_sha=old['remote_sha'],code_root=str(code),files=old['files'],
 source_q1_publication=fp(q1/'source-publication.json'),source_q1_preparation=fp(q1/'preparation.json'),
 source_q1_complete=fp(q1/'qualification-output/complete.json'),model_view=str(view),
 deployment_helper=inputs['deployment_helper'],reuses_immutable_source_and_model=True,
 old_source_or_model_copied=False,old_source_or_model_modified=False)
path=b/'source-publication.json'
if path.exists():assert read(path)==pub,'Immutable Q2 source binding differs'
else:atomic(path,pub)
print(json.dumps(dict(status='Q2_IMMUTABLE_Q1_RUNTIME_BOUND',base=str(b),code_root=str(code),model_view=str(view))))
'''

PREPARE=r'''
guard=lock();unsubmitted();publication,q1_selection=verify_runtime()
sys.path.insert(0,str(code))
from reproduction.operations.qwen7b_warmup.replay_probe_layers import normalize_prefix,REVISION
config=read(parent/'config.json');shards=sorted((parent/'sampling').glob('shard-*/complete.json'))[:4]
assert len(shards)==4,'Need the first4 genuinely sealed sampling shards'
records=[];prefix_files=[];native_files=[];sources=[];seen_tids=set();seen_qids=set()
for shard_path in shards:
 shard=read(shard_path);assert shard['model_revision']==REVISION and shard['run_id']==config['run_id'] and shard['batches']
 batch_path=pathlib.Path(shard['batches'][0]);batch=read(batch_path)
 assert batch['model_revision']==REVISION and batch['run_id']==config['run_id'] and batch['actor_optimizer_updates']==0
 assert pathlib.Path(batch['batch']).name==batch['batch']
 input_path=parent/'inputs'/shard_path.parent.name/(batch['batch']+'.json');assert sha(input_path)==batch['input_sha256']
 original={str(row['qid']):row for row in read(input_path)['rows']}
 prefix=batch['prefix']['prefix'];native=batch['prefix']['actor_hidden'];verify(prefix);verify(native)
 native_meta=read(native['path'])
 assert native_meta['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE' and native_meta['feature_layer']==26 and native_meta['dimension']==3584 and native_meta['question_max_tokens']==273 and native_meta['prefix_tokens']==20
 local_qids=set()
 with gzip.open(prefix['path'],'rt') as handle:
  for line in handle:
   row=json.loads(line);tid=row['trajectory_id'];qid=str(row['qid']);assert tid not in seen_tids;seen_tids.add(tid)
   assert qid in original and row['initial_prompt_token_ids']==original[qid]['initial_prompt_token_ids'] and row['question_hash']==original[qid]['question_hash']
   own=normalize_prefix(row,tid,qid,True);assert own['source_native_manifest']==native['path'] and row['feature_layer']==26
   segments=own['segments'];gaps=[s['request_length']-len(previous['context_tokens']) for previous,s in zip(segments,segments[1:])]
   assert all(gap>=0 for gap in gaps)
   records.append(dict(trajectory_id=tid,qid=qid,response_index=row['response_index'],
       prompt_length=own['question_length'],policy_length=own['policy_length'],
       segment_count=len(segments),max_turn_index=max(s['turn_index'] for s in segments),
       later_context_gap_tokens=max(gaps) if gaps else 0,
       max_context_length=max(len(s['context_tokens']) for s in segments)))
   local_qids.add(qid)
 assert not local_qids&seen_qids,'Original question duplicated across source shards';seen_qids.update(local_qids)
 prefix_files.append({k:prefix[k] for k in ['path','bytes','sha256']});native_files.append({k:native[k] for k in ['path','bytes','sha256']})
 sources.append(dict(shard=fp(shard_path),first_sealed_batch=fp(batch_path),source_input=fp(input_path),prefix=prefix_files[-1],native_manifest=native_files[-1]))
selected,coverage=choose_questions(records,24)
assert len(selected)<=120 and len(selected)%5==0 and len({r['trajectory_id'] for r in selected})==len(selected)
selection=dict(schema_version=1,model_revision=REVISION,prefix_files=prefix_files,
 trajectories=[{k:r[k] for k in ['trajectory_id','qid']} for r in selected],
 native_layer26_manifests=native_files,model_files=q1_selection['model_files'],
 model_source_receipt=q1_selection['model_source_receipt'],label_blind=True,
 selection_rule=SELECTION_RULE,coverage=coverage,
 inherited_model_selection=fp(q1/'inputs/fixture-selection.json'))
path=b/'inputs/fixture-selection.json'
if path.exists():assert read(path)==selection,'Existing immutable extended fixture changed'
else:atomic(path,selection)
prepared=dict(status='BOUNDED_REPLAY_EXTENDED_FIXTURE_PREPARED',code_root=str(code),
 runtime_commit=inputs['runtime_commit'],model_view=str(view),selection=fp(path),
 trajectories=len(selected),questions=len(selected)//5,max_trajectories=120,
 source_publication=fp(b/'source-publication.json'),sources=sources,coverage=coverage,
 model_files_inherited_from=fp(q1/'inputs/fixture-selection.json'),
 model_shards_hashed_on_login=False,feature_tensor_files_read_on_login=False,
 full_rollouts_read=False,own_EM_read=False,gpu_qualification_claimed=False)
target=b/'preparation.json'
if target.exists():assert read(target)==prepared
else:atomic(target,prepared)
print(json.dumps(dict(status=prepared['status'],base=str(b),selection=prepared['selection'],
 questions=prepared['questions'],trajectories=prepared['trajectories'],coverage=coverage,
 immutable_Q1_code_and_model_reused=True,submitted=False)))
'''

SUBMIT=r'''
guard=lock();publication,_=verify_runtime()
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();assert job.isdigit()
 print(json.dumps(dict(status='EXISTING_Q2_JOB_PRESERVED_NO_RESUBMISSION',job_id=job,
    squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),
    sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode']))))
else:
 unsubmitted();prepared=read(b/'preparation.json');assert prepared['status']=='BOUNDED_REPLAY_EXTENDED_FIXTURE_PREPARED'
 verify(prepared['source_publication']);verify(prepared['selection'])
 assert 0<prepared['trajectories']<=120 and prepared['trajectories']==prepared['questions']*5
 assert not (b/'qualification-output').exists(),'Prior GPU output exists: preserve and inspect'
 assert not command(['squeue','-h','-u','mou176','-n',inputs['job_name'],'-o','%i|%T|%R']),'Unknown same-name job; inspect instead of submitting'
 script=render_sbatch(str(b),prepared['selection']['sha256'],inputs['job_name']);path=b/'replay-extended.sbatch'
 if path.exists():assert path.read_text()==script
 else:path.write_text(script)
 intent=dict(status='Q2_SUBMIT_INTENT',script=fp(path),job_name=inputs['job_name'],
   requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='00:30:00'),
   dependency=None,source_publication=prepared['source_publication'],selection=prepared['selection'],epoch_time=time.time())
 atomic(b/'submission-intent.json',intent)
 try:result=subprocess.run(['sbatch','--hold','--parsable',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(b/'submission-intent.json',intent);raise
 if result.returncode:
  intent.update(status='SBATCH_REJECTED',stderr=result.stderr[-2000:]);atomic(b/'submission-intent.json',intent);raise RuntimeError(result.stderr[-2000:])
 job=result.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch response: do not repeat'
 with (b/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 submission=dict(intent,status='Q2_JOB_HELD',job_id=job,code_root=str(code),model_view=str(view));atomic(b/'submission.json',submission)
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);submission.update(status='Q2_JOB_RELEASED',epoch_time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
 atomic(b/'submission.json',submission);print(json.dumps(submission))
'''

STATUS=r'''
result=dict(base=str(b),exists=b.exists(),observed_epoch=time.time())
if (b/'source-publication.json').exists():
 publication,_=verify_runtime();result['code_root']=publication['code_root'];result['runtime_commit']=publication['commit']
for name in ['preparation.json','submission-intent.json','submission.json']:
 if (b/name).exists():
  item=read(b/name);result[name]={k:item[k] for k in ['status','job_id','questions','trajectories','coverage','selection'] if k in item}
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();assert job.isdigit();result['job_id']=job
 result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%u|%j|%T|%P|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode,Elapsed'])
for name in ['progress.json','complete.json','native-layer26-comparison.json']:
 if (b/'qualification-output'/name).exists():result[name]=read(b/'qualification-output'/name)
print(json.dumps(result))
'''


def remote_source(inputs,action):
    functions=''
    if action=='prepare':functions=inspect.getsource(choose_questions)
    if action=='submit':functions=inspect.getsource(render_sbatch)
    prefix='inputs='+repr(inputs)+'\nSELECTION_RULE='+repr(SELECTION_RULE)+'\n'
    if action=='submit':prefix+='Q2='+repr(Q2)+'\nCODE_ROOT='+repr(CODE_ROOT)+'\nMODEL_VIEW='+repr(MODEL_VIEW)+'\nJOB_NAME='+repr(JOB_NAME)+'\n'
    return prefix+COMMON+'\n'+functions+'\n'+dict(init=INIT,prepare=PREPARE,submit=SUBMIT,status=STATUS)[action]


def self_test():
    script=render_sbatch(Q2,'a'*64)
    subprocess.run(['bash','-n'],input=script,text=True,check=True)
    assignments='\n'.join(line for line in script.splitlines() if line.startswith(('REPLAY_','export ')))
    subprocess.run(['bash','-euc',assignments+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(CODE_ROOT)],check=True)
    assert '--max-trajectories 120' in script and '--time=00:30:00' in script and '--dependency' not in script
    values=dict(parent=PARENT,q1=Q1,q2=Q2,code_root=CODE_ROOT,model_view=MODEL_VIEW,
                runtime_commit=RUNTIME_COMMIT,job_name=JOB_NAME,deployment_helper={})
    sizes={}
    for action in ['init','prepare','submit','status']:
        code=remote_source(values,action);compile(code,'remote-'+action,'exec')
        if len(code.encode())>=16000:raise AssertionError('Deliberate16KB transport bound exceeded: '+action)
        sizes[action]=len(code.encode())
    records=[]
    for i in range(30):
        for response in range(5):
            records.append(dict(qid='q%02d'%i,trajectory_id='q%02d-r%d'%(i,response),response_index=response,
                 prompt_length=100+i,policy_length=0 if i==4 else 20,
                 segment_count=2 if i==3 else 1,max_turn_index=1 if i==3 else 0,
                 later_context_gap_tokens=3 if i==3 else 0,max_context_length=140+i))
    selected,coverage=choose_questions(records)
    assert len(selected)==120 and len({r['qid'] for r in selected})==24
    assert {'q03','q04','q00','q29'}<={r['qid'] for r in selected}
    assert coverage['multisegment_present'] and not coverage['semantic_tool_calls_confirmed']
    class NoEMRead(dict):
        def __getitem__(self,key):
            if key in ('labels','strict_em','EM','full','own_em'):
                raise AssertionError('Label/full artifact consulted by fixture selection')
            return super().__getitem__(key)
    protected=[NoEMRead(row,strict_em=1,labels=[1]*5) for row in records]
    protected_selected,protected_coverage=choose_questions(protected)
    assert [r['trajectory_id'] for r in protected_selected]==[r['trajectory_id'] for r in selected]
    assert protected_coverage==coverage
    single=[dict(r,segment_count=1,max_turn_index=0,later_context_gap_tokens=0) for r in records]
    assert not choose_questions(single)[1]['multisegment_present']
    result=dict(status='LOCAL_EXTENDED_REPLAY_DEPLOYMENT_CHECKS_PASSED',safe_shell=True,
                code_SHA_preflight_rendered=True,selection_complete_G5=True,
                remote_command_bytes=sizes,remote_actions_executed=False)
    print(json.dumps(result));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-root',type=Path)
    parser.add_argument('--action',choices=['prepare','status','submit'],default='status')
    parser.add_argument('--submit-authorized',action='store_true');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.control_root is None:parser.error('--control-root required for remote actions')
    if args.action=='submit' and not args.submit_authorized:parser.error('Separate submit requires --submit-authorized')
    repo=Path(__file__).resolve().parents[3];helper={}
    if args.action!='status':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
        rel=str(Path(__file__).resolve().relative_to(repo));raw=Path(__file__).read_bytes()
        if commit!=remote or raw!=subprocess.check_output(['git','show',commit+':'+rel],cwd=repo):
            raise RuntimeError('Publish this exact helper before remote prepare/submit')
        helper=dict(commit=commit,path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    values=dict(parent=PARENT,q1=Q1,q2=Q2,code_root=CODE_ROOT,model_view=MODEL_VIEW,
                runtime_commit=RUNTIME_COMMIT,job_name=JOB_NAME,deployment_helper=helper)
    spec=importlib.util.spec_from_file_location('extended_replay_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    results=[]
    for action in (['init','prepare'] if args.action=='prepare' else [args.action]):
        source=remote_source(values,action)
        if len(source.encode())>=16000:raise RuntimeError('Remote metadata command exceeds deliberate16KB bound')
        result=control.run('python3 -c '+shlex.quote(source),account='mou176',source_address=address,timeout=55)
        rows=[json.loads(line) for line in result.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if result.returncode or not rows:raise RuntimeError('Q2 action unconfirmed; inspect before retry: '+result.stderr.decode(errors='replace')[-1500:])
        results.extend(rows);print(json.dumps(rows[-1]),flush=True)
    out=args.control_root/'outputs/qwen7b-warmup-20261019'/('replay-extended-'+args.action+'.json')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results[-1],indent=2)+'\n')


if __name__=='__main__':main()

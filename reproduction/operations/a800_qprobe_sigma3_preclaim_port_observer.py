"""One bounded read of the pinned sigma3 preclaim exit and port ownership.

This diagnostic does not bind, connect, signal, evaluate, initialize CUDA, read
model/tensor/QA files, retry a launch, or write any remote file. Publish this
exact operator first, then pass its actual private-publication receipt.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'qprobe-a800-sigma3-loss'
HANDLE=ROOT/'preparation/qprobe-a800-sigma3-loss-deployment.json'
READY=ROOT/'preparation/qprobe-sigma3-reviewed-prepared-switch-ready.json'
REVIEW=ROOT/'preparation/qprobe-sigma3-independent-release-review.json'
LAUNCH=ROOT/'preparation/qprobe-a800-sigma3-loss-deployment.launch-1792047208319905000.json'
INSPECT=ROOT/'preparation/qprobe-a800-sigma3-loss-deployment.inspect-1792047415277211000.json'
PINS={str(HANDLE):'d2fba0db2bd7671bc2c26316032ab34ba7134f608c73595c03469acfb4484220',
      str(READY):'9ddddf73b00f62701e1dca9c25ea826c35ada90543bbd8d2857e067f8dbdff88',
      str(REVIEW):'6b974d81912679360872bcaefd625c254694ea0f26e99bc5acfce7446cc7ac3e',
      str(LAUNCH):'843428a512c263a15b9b2cffac1ce335f41f03dd0628595b3efffdc389508dba',
      str(INSPECT):'c7de16c7817284ec997c02227dff0b7c74eac3dbf8b8e2609ddc69eac1e722d7'}
SOURCE='9ec12a3b0ca919be0f083927439d4f2fdef42e48'
MAP_SHA='2efbe2e4a7a5280567c415a6796b6a059f462ec9d2d580ae39623317b931e5ff'
RUN='c5289001497e458485b98030e71d5f36'
EXECUTION='aaecb213f394427ca308b25fa23cea60'
BASE='/data/shaoshitong/Search-G1-a800-fresh200-b8eb3037d2704046952e66c91e38c81d'
PID,TICKS=4107391,612679933
SUPPORT='reproduction/operations/a800_qprobe_sigma3_switch_stop.py'
SUPPORT_COMMIT='a521995b09d35d70dc27a6c8d0c5835e634c7e60'
SUPPORT_SHA='0a5734aa1358a18601f89b46f2ed7112c137467d622f6346af16660f2423adae'


def source_support():
    path=OPS/SUPPORT;raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SUPPORT_SHA
    assert subprocess.check_output(['git','-C',str(OPS),'show',SUPPORT_COMMIT+':'+SUPPORT],timeout=10)==raw
    spec=importlib.util.spec_from_file_location('pinned_sigma3_preclaim_source_support',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


REMOTE_SUPPORT=r'''
import hashlib,json,os,re,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['GIT_OPTIONAL_LOCKS']='0'
p=json.load(sys.stdin)
deadline=time.monotonic()+50
def check():
 if time.monotonic()>=deadline:raise TimeoutError('bounded diagnostic deadline')
def process(pid):
 check();root=Path('/proc')/str(pid);raw=(root/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=root.stat().st_uid,state=fields[0])
def bounded_proc(path,limit):
 check()
 with Path(path).open('rb') as stream:raw=stream.read(limit+1)
 assert len(raw)<=limit
 return raw
STATES={'01':'ESTABLISHED','02':'SYN_SENT','03':'SYN_RECV','04':'FIN_WAIT1','05':'FIN_WAIT2',
        '06':'TIME_WAIT','07':'CLOSE','08':'CLOSE_WAIT','09':'LAST_ACK','0A':'LISTEN','0B':'CLOSING','0C':'NEW_SYN_RECV'}
def tcp_rows(raw,family):
 result=[]
 for line in raw.decode('ascii').splitlines()[1:]:
  fields=line.split();assert len(fields)>=10
  address,port=fields[1].rsplit(':',1);port=int(port,16)
  if port not in (18010,18012):continue
  state=fields[3].upper();inode=int(fields[9]);uid=int(fields[7])
  assert state in STATES and inode>=0 and uid>=0
  loopback=address=='0100007F' if family=='tcp' else address in ('00000000000000000000000001000000','0000000000000000FFFF00000100007F')
  binding='loopback' if loopback else 'all_interfaces' if set(address)=={'0'} else 'other_interface'
  result.append(dict(port=port,family=family,state=STATES[state],state_hex=state,socket_inode=inode,socket_uid=uid,binding_scope=binding))
  assert len(result)<=256
 return result
def cwd_scope(pid):
 try:value=os.readlink('/proc/'+str(pid)+'/cwd')
 except (FileNotFoundError,ProcessLookupError,PermissionError):return dict(cwd_scope='unavailable')
 parts=Path(value).parts
 namespaces=[part for part in parts if re.fullmatch(r'Search-G1[-A-Za-z0-9_.]{0,100}',part)]
 if namespaces:return dict(cwd_scope='search_g1_namespace',cwd_namespace=namespaces[-1])
 name=Path(value).name
 return dict(cwd_scope='basename_only',cwd_basename=name if re.fullmatch(r'[A-Za-z0-9_.-]{1,96}',name) else 'redacted')
def port_summary(entries):
 result={}
 for port in (18010,18012):
  rows=[row for row in entries if row['port']==port]
  listeners=sum(row['state']=='LISTEN' for row in rows);waiting=sum(row['state']=='TIME_WAIT' for row in rows)
  status='listener_present' if listeners else 'time_wait_only' if rows and waiting==len(rows) else 'other_tcp_state_present' if rows else 'no_current_tcp_entry'
  result[str(port)]=dict(status=status,entry_count=len(rows),listener_count=listeners,time_wait_count=waiting,
   bindability_proven=False,entries=rows)
 return result
'''

REMOTE_READ=r'''
check();base=Path(p['base']);repo=base/'repo';intent=p['intent']
assert sys.executable=='/data/search-g1/env/bin/python'
assert str(base)==p['base'] and base==base.resolve() and base.stat().st_uid==os.getuid()
assert document(base/'preparation/qprobe-intent.json')==intent
run=document(base/'logs/run-manifest.json');launch=document(base/'logs/deployment-launch-receipt.json')
assert launch==p['launch']
assert (run['run_id'],run['execution_id'],run['source_commit'])==(p['run'],p['execution'],p['source'])
assert run['source_sha256']==intent['source_sha256'] and len(run['source_sha256'])==481
assert (launch['run_id'],launch['execution_id'],launch['source_commit'],launch['pid'],launch['start_ticks'],launch['sid'])==(p['run'],p['execution'],p['source'],p['pid'],p['ticks'],p['pid'])
assert run['completed_updates']==0 and run['target_updates']==200 and run['training_world_size']==4
assert run.get('full_state_restore') is None and run['question_probe']['loss_mode']=='sigma3_std_aux6_v1'
assert (run['question_probe']['mean_loss_weight'],run['question_probe']['std_loss_weight'],run['question_probe']['count_aux_loss_weight'])==(0.0,3.0,0.1)
verify_source(repo,p['source'],run['source_sha256'],481)
check()
try:actual=process(p['pid'])
except (FileNotFoundError,ProcessLookupError):actual=None
birth_matches=actual is not None and (actual['start_ticks'],actual['sid'],actual['uid'])==(p['ticks'],p['pid'],os.getuid())
assert not birth_matches or actual['state'] in ('Z','X'), 'Original pipeline still live'
absence={}
for name in ('pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json','resolved-training-config.json',
             'actual-updates.jsonl','actual-attempts.jsonl','native-allocation-admission.json','policy-safety-calibration.json'):
 path=base/'logs'/name;absence[name]=not path.exists() and not path.is_symlink()
assert all(absence.values())
rank_files=[path.name for pattern in ('optimizer-rank-*','scheduler-rank-*','worker-rank-*') for path in (base/'logs').glob(pattern)]
assert not rank_files
assert not Path(run['train_log']).exists() and not Path(run['train_log']).is_symlink()
checkpoint_root=base/'checkpoints';assert checkpoint_root==checkpoint_root.resolve() and checkpoint_root.is_dir()
checkpoint_files=[]
for path in checkpoint_root.rglob('*'):
 check();assert not path.is_symlink()
 if path.is_file():checkpoint_files.append(str(path.relative_to(checkpoint_root)))
 assert len(checkpoint_files)<=16
assert not checkpoint_files
log_path=Path(launch['stdout_log']);assert log_path==base/'logs/deployment-pipeline.log'
raw=sealed(log_path,8*1024);assert len(raw)==p['initial_log_bytes']
text=raw.decode('utf-8',errors='replace')
frames=[dict(file=Path(match.group(1)).name,line=int(match.group(2))) for match in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+),',text,re.MULTILINE)][-12:]
errors=list(re.finditer(r'^([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception)):(?:\s*\[Errno\s+([0-9]+)\])?',text,re.MULTILINE))
assert errors and any(row['file']=='run_pipeline.py' and row['line']==170 for row in frames)
last=errors[-1];error=dict(type=last.group(1).rsplit('.',1)[-1],errno=int(last.group(2)) if last.group(2) else None,frames=frames)
entries=[];proc_tables=[];scanned=denied=vanished=links=0
ports_observed=error['type']=='OSError'
if ports_observed:
 for family in ('tcp','tcp6'):
  raw_table=bounded_proc('/proc/net/'+family,2*1024**2);entries+=tcp_rows(raw_table,family)
  proc_tables.append(dict(family=family,bytes=len(raw_table),sha256=hashlib.sha256(raw_table).hexdigest()))
 inodes={row['socket_inode'] for row in entries if row['socket_inode']>0};owners={inode:[] for inode in inodes}
 for root in (Path('/proc').iterdir() if inodes else ()):
  check()
  if not root.name.isdigit():continue
  scanned+=1;assert scanned<=65536
  pid=int(root.name)
  try:
   before=process(pid);matches={}
   for fd in (root/'fd').iterdir():
    check();links+=1;assert links<=262144
    try:target=os.readlink(fd)
    except (FileNotFoundError,ProcessLookupError):continue
    match=re.fullmatch(r'socket:\[([0-9]+)\]',target)
    if match and int(match.group(1)) in inodes:
     inode=int(match.group(1));matches[inode]=matches.get(inode,0)+1
   if not matches:continue
   scope=cwd_scope(pid);after=process(pid)
   if any(before[key]!=after[key] for key in ('pid','start_ticks','sid','uid')):vanished+=1;continue
   for inode,count in matches.items():
    owners[inode].append(dict(after,matching_fd_count=count,**scope));assert len(owners[inode])<=128
  except PermissionError:denied+=1
  except (FileNotFoundError,ProcessLookupError):vanished+=1
 for row in entries:
  row['owners']=owners.get(row['socket_inode'],[])
  row['owner_status']='no_process_for_kernel_time_wait' if row['socket_inode']==0 and row['state']=='TIME_WAIT' else 'matched' if row['owners'] else 'unresolved'
assert not any(name=='torch' or name.startswith(('torch.','transformers.','vllm.')) for name in sys.modules)
result=dict(role='a800_qprobe_sigma3_actual_preclaim_port_observation',run_id=p['run'],execution_id=p['execution'],source_commit=p['source'],
 base=str(base),original_pipeline=dict(pid=p['pid'],start_ticks=p['ticks'],sid=p['pid'],original_birth_live=False,
 original_birth_present=birth_matches,current_pid_identity=actual),
 preclaim_absence=absence,rank_log_files=0,checkpoint_files=0,training_log_exists=False,
 startup_log=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),exception=error,raw_text_returned=False),
 ports=port_summary(entries) if ports_observed else dict(status='not_observed_non_socket_exception'),proc_tables=proc_tables,owner_scan=dict(performed=ports_observed and bool(entries),processes=scanned,fd_links=links,
 permission_denied_processes=denied,raced_or_exited_processes=vanished),
 no_remote_mutation=True,no_signal_bind_or_connect=True,no_tensor_qa_or_model_read=True,no_cuda_initialization=True,
 checked_unix=time.time())
out=json.dumps(result,sort_keys=True,allow_nan=False);assert len(out.encode())<=120*1024
print('SIGMA3_PORT_JSON:'+out,flush=True)
'''


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def local_inputs(support):
    values={}
    for name,digest in PINS.items():
        path=Path(name);assert support.sha(path)==digest;values[name]=support.document(path)
    intent=values[str(HANDLE)];ready=values[str(READY)];review=values[str(REVIEW)]
    assert intent['source_commit']==SOURCE and len(intent['source_sha256'])==481 and canonical_sha(intent['source_sha256'])==MAP_SHA
    assert intent['base']==BASE and intent.get('saved_checkpoint_restore') is None
    assert (ready['new_run_id'],ready['new_execution_id'],ready['new_base'],ready['new_source_commit'])==(RUN,EXECUTION,BASE,SOURCE)
    assert ready['files']['new_handle']==dict(path=str(HANDLE),sha256=PINS[str(HANDLE)])
    assert ready['files']['new_review']==dict(path=str(REVIEW),sha256=PINS[str(REVIEW)])
    assert review['approved'] is True and review['independent'] is True and review['parent_context_inherited'] is False
    assert review['source_commit']==SOURCE and review['source_sha256']==intent['source_sha256'] and review.get('blocking_findings',[])==[]
    assert all(value is True for value in review['checks'].values())
    launch=values[str(LAUNCH)];assert launch['action']=='launch' and launch['exit_code']==0 and len(launch['observations'])==1
    first=launch['observations'][0]
    assert first['stage']=='PREPARED_PIPELINE_LAUNCHED'
    first={key:value for key,value in first.items() if key!='stage'}
    assert (first['run_id'],first['execution_id'],first['source_commit'],first['pid'],first['start_ticks'],first['sid'])==(RUN,EXECUTION,SOURCE,PID,TICKS,PID)
    inspected=values[str(INSPECT)];assert inspected['action']=='inspect' and inspected['exit_code']==0 and inspected['observations']
    for row in inspected['observations']:
        assert (row['run_id'],row['execution_id'],row['source_commit'],row['pid'],row['start_ticks'])==(RUN,EXECUTION,SOURCE,PID,TICKS)
        assert row['pipeline_alive'] is False and row['pipeline_claimed'] is False and row['trainer_claimed'] is False
        assert row['stdout_log_bytes']==910 and row['actual_updates_ledger']['exists'] is False
    support.verify_source(RUNTIME,SOURCE,intent['source_sha256'],481)
    return dict(intent=intent,launch=first,initial_log_bytes=910,run=RUN,execution=EXECUTION,source=SOURCE,base=BASE,pid=PID,ticks=TICKS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-publication',type=Path,required=True)
    args=parser.parse_args();support=source_support()
    operator_commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=10).strip()
    path=Path(__file__).resolve();relative=str(path.relative_to(OPS));raw=support.sealed(path)
    assert subprocess.check_output(['git','-C',str(OPS),'show',operator_commit+':'+relative],timeout=10)==raw, 'Publish this exact observer first'
    publication=support.document(args.operator_publication)
    assert publication['source_commit']==publication['remote_commit']==operator_commit
    assert publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    selected=[row for row in publication['files'] if row['path']==relative]
    assert len(selected)==1 and selected[0]['sha256']==hashlib.sha256(raw).hexdigest()
    payload=local_inputs(support)
    remote=REMOTE_SUPPORT+support.SOURCE_CHECK+REMOTE_READ;ast.parse(remote)
    spec=importlib.util.spec_from_file_location('pinned_sigma3_port_existing_route',RUNTIME/'reproduction/a800_common/deploy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert support.sha(Path(module.__file__))==payload['intent']['source_sha256']['reproduction/a800_common/deploy.py']
    route=module.ssh_command(remote);assert route[-1].startswith('python3 -c ')
    route[-1]=shlex.quote('/data/search-g1/env/bin/python')+' -c '+shlex.quote(remote)
    assert local_inputs(support)==payload
    stem='qprobe-sigma3-preclaim-port-read-once-v2';claim=ROOT/'preparation'/(stem+'.json');output=claim.with_name(claim.stem+'.result.json')
    request=dict(role='a800_qprobe_sigma3_preclaim_port_one_read_claim',run_id=RUN,execution_id=EXECUTION,source_commit=SOURCE,
        original_pipeline_birth=dict(pid=PID,start_ticks=TICKS,sid=PID),operator_commit=operator_commit,operator_sha256=hashlib.sha256(raw).hexdigest(),
        operator_publication_sha256=support.sha(args.operator_publication),remote_code_sha256=hashlib.sha256(remote.encode()).hexdigest(),
        pinned_evidence_sha256=PINS,no_automatic_retry=True,no_remote_mutation=True,claimed_unix=time.time())
    support.once(claim,request)
    with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
        try:
            completed=subprocess.run(route,input=json.dumps(payload,sort_keys=True,allow_nan=False).encode(),stdout=stdout,stderr=stderr,timeout=60)
            status=dict(exit_code=completed.returncode,status='READ_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            status=dict(exit_code=None,status='TRANSPORT_UNKNOWN_NO_RETRY',failure_type=type(failure).__name__)
        status.update(stdout_bytes=stdout.tell(),stderr_bytes=stderr.tell());stdout.seek(0);returned=stdout.read(128*1024+1)
        stderr.seek(0);error=stderr.read(64*1024)
    status['bounded_stderr_sha256']=hashlib.sha256(error).hexdigest();diagnostic=error.decode(errors='replace')
    status['remote_exception_frames']=[dict(file=Path(match.group(1)).name[:96],line=int(match.group(2)))
        for match in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+),',diagnostic,re.MULTILINE)][-8:]
    types=re.findall(r'^([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception))(?::|$)',diagnostic,re.MULTILINE)
    status['remote_exception_type']=types[-1].rsplit('.',1)[-1][:96] if types else None
    results=[]
    if status['exit_code']==0 and len(returned)<=128*1024:
        try:
            results=[json.loads(line.split(':',1)[1]) for line in returned.decode().splitlines() if line.startswith('SIGMA3_PORT_JSON:')]
            assert len(results)==1 and (results[0]['run_id'],results[0]['execution_id'],results[0]['source_commit'])==(RUN,EXECUTION,SOURCE)
            assert results[0]['no_remote_mutation'] is True and results[0]['no_signal_bind_or_connect'] is True
        except (AssertionError,ValueError,UnicodeDecodeError,KeyError,TypeError) as failure:
            results=[];status.update(status='OUTPUT_REJECTED_NO_RETRY',failure_type=type(failure).__name__)
    elif len(returned)>128*1024:status['status']='OUTPUT_LIMIT_EXCEEDED_NO_RETRY'
    receipt=dict(request,**status,results=results,observed_unix=time.time());support.once(output,receipt)
    print(json.dumps(dict(receipt=str(output),**status,results=results,no_automatic_retry=True),allow_nan=False))
    return 0 if status['exit_code']==0 and len(results)==1 else 1


if __name__=='__main__':raise SystemExit(main())

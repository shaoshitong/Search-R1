#!/usr/bin/env python3
"""Once-only read-only COMPLETE fast-final delta. Source preparation does not run it.

Future: python3 -B work/tree-speed-repair-20261008/final_fast_terminal_delta.py collect
 --workspace "$WS" --pins "$ACTUAL_PINS" --pins-bytes "$ACTUAL_BYTES" --pins-sha256 "$ACTUAL_SHA"
Local recovery from the exact durable response uses phase recover plus
 --response-bytes "$ACTUAL_RESPONSE_BYTES" --response-sha256 "$ACTUAL_RESPONSE_SHA".
No traincost/variance/curve collection, evaluation or cleanup operation is performed.
"""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import shlex
import time
import uuid

UNIT='sst-tree-nested-final-eval-tp1-778b2b43-20261016.service'
INV='abd6a826997945da932cf09047806431'
EVAL='a8ab17f36f1bcf64f19193522aaed84fb2749863'
CODE='439a58113f8e12e557121ef3b7b241ebd740376c240e1cd40b27be2083c5e9b4'
BASE='outputs/tree-nested-branch-handoff-20261013'
STEM='FINAL200_FAST_TERMINAL_DELTA'
HELPER='work/tree-speed-repair-20261008/ssh_retained18.py'
CONTROL='/tmp/tree-mgmt-3fw9a1ab/control'
MAX_RESPONSE=4*1024*1024
LOCAL={};NEW={}

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
def strict(raw):
 def unique(pairs):
  out={}
  for key,value in pairs:need(key not in out,'Duplicate JSON key');out[key]=value
  return out
 return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def locate(name):
 p=Path(name);need('..' not in p.parts,'No parent path traversal')
 p=p if p.is_absolute() else W/p if p.parts[0] in ('outputs','work') else B/p
 need(not p.is_symlink(),'Ordinary local file');p=p.resolve(strict=True);need(p.is_relative_to(W),'Workspace local proof');return p
def pin(p):
 p=locate(p);r=p.read_bytes();return dict(path=str(p.relative_to(W)),bytes=len(r),sha256=sha(r))
def read(q):
 p=locate(q['path']);r=p.read_bytes();need((len(r),sha(r))==(q['bytes'],q['sha256']),'Local source binding changed: '+p.name)
 name=str(p);need(name not in LOCAL or LOCAL[name]==r,'Input changed between reads');LOCAL[name]=r;return r
def bound(q):return strict(read(q))
def publish(name,raw):
 for root in [B]+M:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():need(p.read_bytes()==raw,'Immutable existing artifact differs: '+name);continue
  tmp=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
  with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  try:
   try:os.link(tmp,p)
   except FileExistsError:need(p.read_bytes()==raw,'Concurrent immutable artifact differs')
  finally:tmp.unlink()
 NEW[name]=raw;return dict(path=str((B/name).relative_to(W)),bytes=len(raw),sha256=sha(raw))
def checks():
 observations=[]
 for n in (1,2):
  for p,raw in LOCAL.items():need(Path(p).read_bytes()==raw,'Local source changed')
  for name,raw in NEW.items():
   for root in [B]+M:need((root/name).read_bytes()==raw,'New mirrored bytes changed')
  observations.append(dict(pass_number=n,utc=datetime.now(timezone.utc).isoformat(),bound_local_files=len(LOCAL),generated_three_copy_files=len(NEW),generated_primary_bytes=sum(map(len,NEW.values()))))
 return observations

def inputs(args):
 request=bound(dict(path=str(args.pins),bytes=args.pins_bytes,sha256=args.pins_sha256))
 need(request['schema']=='tree_fast_final_terminal_delta_input_v1' and request['status']=='ACTUAL_COMPLETE_ROOT_AUTHORIZED_READ_READY','Future actual COMPLETE authorization only')
 need(request['unit']==UNIT and request['invocation_id']==INV and request['original_main_pid']==3409502 and request['evaluation_source_commit']==EVAL and request['code_manifest_sha256']==CODE,'Exact original fast instance')
 need(locate(request['source_command']['path'])==Path(__file__).resolve(),'Executing frozen source pin');read(request['source_command'])
 remote=request['remote_source'];need(locate(remote['path'])==Path(__file__).with_name('final_fast_terminal_delta_remote.py').resolve(),'Adjacent frozen remote source');read(remote)
 helper=read(dict(path=HELPER,bytes=4411,sha256='8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'))
 need(helper,'Existing original retained18 helper')
 proofs=request['local_proofs'];expected={'archive':'archive-proof.json','assets':'asset-manifest.json','original_terminal':'original-terminal.json','handoff':'handoff-authorized.json','dispatch':'dispatch-once-tp1-20261016.json'}
 values={}
 for key,name in expected.items():
  need(locate(proofs[key]['path'])==B/'final200-fast-handoff-proofs'/name,'Exact existing handoff proof');values[key]=bound(proofs[key])
 dispatch=values['dispatch'];need(dispatch['unit']==UNIT and dispatch['eval_wrapper_source_commit']==EVAL and dispatch['code_manifest_sha256']==CODE,'Actual dispatch identity')
 ready=bound(request['root_ready_proof'])
 def complete(v):
  if isinstance(v,dict):
   if v.get('unit')==UNIT and v.get('status')=='COMPLETE' and v.get('worker_exit_code')==0 and v.get('evaluated_count')==51713:return True
   return any(complete(x) for x in v.values())
  if isinstance(v,list):return any(complete(x) for x in v)
  return False
 need(complete(ready),'Root actual ready proof must retain original COMPLETE run-result; no queued/future readiness')
 cpu=bound(proofs['CPU_HF_pre_stop']);trace=bound(proofs['trace_pre_stop']);cpu_state=cpu['summary']['systemd'];trace_state=trace['systemd']
 need(cpu['summary']['unit']=='sst-tree-nativehf200-cpu-caac340-778b2b43.service' and cpu_state['InvocationID']=='510e864008f548fcbefcc7143d7f5976','Actual CPU HF pre-stop source')
 need(trace['unit']=='sst-tree-trace200-durable-repair001.service' and trace_state['InvocationID']=='94fd3b56080743aa884d19e9fefb40e6','Actual durable trace pre-stop source')
 for state in (cpu_state,trace_state):need(state['MainPID']=='0' and state['Result']=='success' and state['ExecMainStatus']=='0','Already-exited accepted audit process required')
 cleanup=bound(proofs['CPU_audit_cleanup']);need(cleanup['returncode']==0 and cleanup['remote']['status']=='COMPLETE','ROOT completed CPU audit cleanup proof')
 need(locate(proofs['original_service_terminal']['path'])==B/'FINAL200_ORIGINAL_FINAL_TERMINAL_READBACK_1.json','Exact retained original778 terminal proof')
 original_terminal_state=bound(proofs['original_service_terminal'])['systemd'];need(original_terminal_state['InvocationID']=='a313bda919834d24b4c5d491698d39ce' and original_terminal_state['MainPID']=='0' and original_terminal_state['ExecMainStatus']=='1','Exact retained user-stopped original778 terminal')
 root_states=[]
 def states(value):
  if isinstance(value,dict):
   if value.get('InvocationID')==INV:root_states.append({k:v for k,v in value.items() if isinstance(v,(str,int,float,bool)) or v is None})
   for child in value.values():states(child)
  elif isinstance(value,list):
   for child in value:states(child)
  elif isinstance(value,str) and 'InvocationID='+INV in value:
   state=dict(line.split('=',1) for line in value.splitlines() if '=' in line)
   if state.get('InvocationID')==INV:root_states.append(state)
 states(ready);need(len(root_states)<=16,'Bounded own root terminal/asof state observations')
 original=values['original_terminal'];paths={Path(q['path']).name:q for q in original['partial_final_and_terminal_cost_inputs_retained']}
 originals=dict(run_result=original['original_run_result'],termination_cost=paths['termination-cost.json'],partial_final_control=paths['validation-506917-40.json'])
 return request,dict(gpu_uuids=dispatch['gpu_uuids'],asset_manifest_sha256=values['handoff']['asset_manifest']['sha256'],handoff_sha256=dispatch['handoff_sha256'],original_final_only_inputs=originals,human_authority_local_proof=proofs['handoff'],auxiliary_cleanup_remote_path='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73/output/final-eval-fast/completed-cpu-audit-cleanup-20261016.json',auxiliary_cleanup_expected_object=cleanup['remote'],auxiliary_cleanup_local_proof=proofs['CPU_audit_cleanup'],root_preserved_unit_states=root_states,original_systemd_asof=original_terminal_state,original_systemd_local_proof=proofs['original_service_terminal'])

def saved_response(size,digest):
 name=STEM+'_RESPONSE.txt';raw=(B/name).read_bytes();need((len(raw),sha(raw))==(size,digest),'Actual full response bytes/SHA required')
 line,tail=raw.split(b'\n',1);h=strict(line);a=h['stdout_bytes'];b=h['stderr_bytes'];need(len(tail)==a+b,'Complete framed stdout/stderr')
 stdout,stderr=tail[:a],tail[a:];need(sha(stdout)==h['stdout_sha256'] and sha(stderr)==h['stderr_sha256'],'Response stream SHA')
 return h,stdout,stderr

def retain_error(phase,error):
 value=dict(schema='tree_fast_final_terminal_delta_error_v1',phase=phase,utc=datetime.now(timezone.utc).isoformat(),error=repr(error),automatic_retry=False,remote_replay_allowed=False)
 p=B/(STEM+'_RESPONSE.txt')
 if p.exists():value['durable_response']=pin(p)
 try:publish(STEM+'_READ_ERROR_'+uuid.uuid4().hex+'.json',encode(value))
 except Exception:pass

def parts(value):
 result=dict(value)
 for key in list(result):
  data=result[key]
  if len(encode(data))<=80000:continue
  need(isinstance(data,(dict,list)),'Large scalar evidence must be a complete container')
  sequence=list(data.items()) if isinstance(data,dict) else list(enumerate(data));chunks=[];current={} if isinstance(data,dict) else []
  for k,v in sequence:
   candidate=dict(current,**{k:v}) if isinstance(data,dict) else current+[v]
   if len(encode(dict(schema='fast_final_complete_field_part_v1',field=key,value=candidate)))>100000 and current:
    chunks.append(current);current={k:v} if isinstance(data,dict) else [v]
   else:current=candidate
  if current:chunks.append(current)
  refs=[]
  for i,chunk in enumerate(chunks):
   raw=encode(dict(schema='fast_final_complete_field_part_v1',field=key,value=chunk));need(len(raw)<=100000,'One complete metadata item exceeds cap; no truncation')
   refs.append(publish(f'final200_fast_terminal_delta/{key}_PART{i:03d}.json',raw))
  result[key]=dict(storage='complete_original_field_parts',container_type='mapping' if isinstance(data,dict) else 'list',original_entries=len(data),parts=refs)
 need(len(encode(result))<=100000,'Complete terminal descriptor exceeds cap');return result

def finish(args,request,header,stdout,stderr):
 need(header['returncode']==0 and header['actual_remote_calls']==1 and header['input_manifest']==pin(args.pins),'Original successful response required')
 need(header['source_command']==request['source_command'] and header['remote_source']==request['remote_source'],'Actual source producer binding')
 result=strict(stdout);need(result['status']=='ACTUAL_COMPLETE_FINAL51713_SCALAR_COST_CLEANUP_READBACK' and result['evaluated_count']==51713 and result['committed_chunk_count']==809,'Actual complete final delta')
 result.update(input_manifest=pin(args.pins),generator=pin(Path(__file__)),remote_generator=request['remote_source'],transport=header['transport'],local_observer=dict(wall_seconds=header['local_wall_seconds'],process_CPU_seconds=header['local_process_CPU_seconds'],stdout_bytes=header['stdout_bytes'],stderr_bytes=header['stderr_bytes']),actual_read_only_remote_calls=1,source1ed_calls=0,traincost_variance_curve_recollected=False)
 product=publish(STEM+'.json',encode(parts(result)));observations=checks()
 readback=B/(STEM+'_READBACK.json')
 if readback.exists():
  raw=readback.read_bytes();need(strict(raw)['product']==product,'Existing readback product differs');publish(readback.name,raw)
 else:publish(readback.name,encode(dict(schema='tree_fast_final_terminal_delta_readback_v1',product=product,observations=observations,input_manifest=pin(args.pins),local_bound_proofs=[pin(p) for p in LOCAL],no_other_tasks_operated=True,goal_complete=False)))
 checks();print(json.dumps(dict(product=product,readback=pin(readback),actual_remote_calls_this_collection=1,cleanup_observed=result['own_cleanup_observed_pass'],auxiliary_root_stop_observed=result['auxiliary_audit_units_root_stop_observed'],goal_complete=False)))

def main():
 global W,B,M
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['collect','recover']);parser.add_argument('--workspace',required=True,type=Path);parser.add_argument('--pins',required=True,type=Path);parser.add_argument('--pins-bytes',required=True,type=int);parser.add_argument('--pins-sha256',required=True);parser.add_argument('--response-bytes',type=int);parser.add_argument('--response-sha256');args=parser.parse_args();need(__debug__,'Do not disable identity gates')
 W=args.workspace.resolve(strict=True);B=W/BASE;M=[W/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
 request,remote_pins=inputs(args)
 if args.phase=='recover':
  try:
   header,stdout,stderr=saved_response(args.response_bytes,args.response_sha256);publish(STEM+'_RESPONSE.txt',(B/(STEM+'_RESPONSE.txt')).read_bytes());finish(args,request,header,stdout,stderr)
  except Exception as error:retain_error('recover-local-only',error);raise
  return
 need(os.environ.get('TREE_MGMT_CONTROL_PATH')==CONTROL,'Explicit original retained18 ControlPath required')
 remote_source=read(request['remote_source']).decode();program=remote_source+'\nPINS='+repr(remote_pins)+"\n_RESULT=collect(PINS)\n_RAW=json.dumps(_RESULT,ensure_ascii=False,separators=(',',':'),allow_nan=False)\nrequire(len(_RAW.encode())<=4*1024*1024,'Bounded scalar-only response exceeded; no truncation')\nprint(_RAW)\n"
 packed=base64.b64encode(lzma.compress(program.encode(),preset=9)).decode();command='env CUDA_VISIBLE_DEVICES= python3 -B -c '+shlex.quote('import base64,lzma;exec(lzma.decompress(base64.b64decode('+repr(packed)+')))')
 spec=importlib.util.spec_from_file_location('fast_terminal_retained18',W/HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);argv=helper.route(command,'h100-10')
 transport=dict(program_bytes=len(program.encode()),target_command_bytes=len(command.encode()),management_command_bytes=len(argv[-1].encode()),argv_bytes=sum(len(v.encode())+1 for v in argv));need(transport['target_command_bytes']<=32768 and transport['management_command_bytes']<=65536 and transport['argv_bytes']<=80000,'Retained route length caps before SSH')
 marker=B/(STEM+'_ATTEMPT.json');attempt=encode(dict(schema='tree_fast_final_terminal_delta_once_v1',started_utc=datetime.now(timezone.utc).isoformat(),input_manifest=pin(args.pins),transport=transport,automatic_retry=False,authorized_remote_calls=1))
 try:fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 except FileExistsError:
  existing=marker.read_bytes();raise RuntimeError('Existing once qualification; inspect saved result, never repeat helper: '+sha(existing))
 with os.fdopen(fd,'wb') as f:f.write(attempt);f.flush();os.fsync(f.fileno())
 publish(marker.name,attempt)
 try:
  started=datetime.now(timezone.utc).isoformat();wall_start=time.perf_counter();cpu_start=time.process_time();response=helper.run(command,target='h100-10',timeout=180)
  stdout=response.stdout.encode();stderr=response.stderr.encode();need(len(stdout)<=MAX_RESPONSE and len(stderr)<=32768,'Bounded metadata return/stderr budget')
  header=dict(schema='tree_fast_final_terminal_delta_response_v1',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),returncode=response.returncode,actual_remote_calls=1,local_wall_seconds=time.perf_counter()-wall_start,local_process_CPU_seconds=time.process_time()-cpu_start,input_manifest=pin(args.pins),source_command=request['source_command'],remote_source=request['remote_source'],transport=transport,stdout_bytes=len(stdout),stdout_sha256=sha(stdout),stderr_bytes=len(stderr),stderr_sha256=sha(stderr))
  raw=encode(header)+stdout+stderr;need(len(raw)<=MAX_RESPONSE+65536,'Complete framed response cap');saved=publish(STEM+'_RESPONSE.txt',raw)
  h,out,err=saved_response(saved['bytes'],saved['sha256']);finish(args,request,h,out,err)
 except Exception as error:retain_error('collect',error);raise
if __name__=='__main__':main()

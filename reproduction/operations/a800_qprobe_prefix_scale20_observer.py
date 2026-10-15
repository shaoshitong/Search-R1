"""One bounded, read-only metadata observation of an existing scale20 handle.

Uses the immutable controller's route, COMMON and observation logic. Adds native
pipeline/driver/retriever/stager visibility without launching or changing work.
No QA, tokens, tensors, model readback or log message text is returned.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
CONTROLLER = Path(__file__).with_name('a800_qprobe_prefix_scale20.py')
MAX_OUTPUT = 128 * 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


NATIVE = r'''
import math,re
MAX_DOCUMENT=64*1024**2
def bounded(q,limit=128*1024):
 q=pathlib.Path(q)
 assert q.resolve()==q and not q.is_symlink() and q.stat().st_uid==os.getuid()
 assert q.stat().st_size<=limit
 return read(q)
def safe(value,key=''):
 if value is None or isinstance(value,bool):return value
 if isinstance(value,(int,float)):return value if not isinstance(value,float) or math.isfinite(value) else None
 if isinstance(value,str):
  if re.fullmatch(r'[0-9a-f]{32,64}',value) or re.fullmatch(r'GPU-[0-9a-fA-F-]+',value):return value
  if key in ('stage','state','role','error_class','failure_type') and re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}',value):return value
  return None
 if isinstance(value,list):return [safe(x,key) for x in value[:64]]
 if isinstance(value,dict):
  return {k:safe(v,k) for k,v in value.items() if isinstance(k,str) and
   k not in ('questions','question','prompt','response','tokens','input_ids','position_ids','argv','arguments','command','error','stderr','stdout','shards','reuse','reused_cache','spec') and
   (not isinstance(v,str) or safe(v,k) is not None)}
 return None
def scalar_log(q):
 q=pathlib.Path(q)
 if not q.exists():return dict(exists=False)
 assert q.resolve()==q and not q.is_symlink() and q.stat().st_uid==os.getuid()
 stat=q.stat()
 with q.open('rb') as stream:stream.seek(max(0,stat.st_size-32768));tail=stream.read(32768)
 errors=[];frames=[];stages=[]
 for line in tail.decode(errors='replace').splitlines():
  error=re.match(r'^\s*((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception))(?::|$)',line)
  if error:errors.append(error.group(1))
  frame=re.search(r'File "([^"\n]+)", line (\d+)(?:, in ([A-Za-z_]\w*|<module>))?',line)
  if frame:
   name=pathlib.Path(frame.group(1)).name
   if re.fullmatch(r'[A-Za-z0-9_.-]{1,120}',name):frames.append(dict(file=name,line=int(frame.group(2)),function=frame.group(3)))
  try:
   item=json.loads(line)
   if isinstance(item,dict) and re.fullmatch(r'[A-Z][A-Z0-9_:-]{0,119}',str(item.get('stage',''))):
    keys=('stage','completed_chunks','completed_questions','completed_trajectories','new_trajectories','elapsed_seconds','checked_unix','observed_unix','real_tokens_written','total_worker_completed_trajectories','compact_shard_ordinal')
    stages.append(safe({k:v for k,v in item.items() if k in keys}))
  except ValueError:pass
 return dict(exists=True,bytes=stat.st_size,mtime_unix=stat.st_mtime,tail_bytes_read=len(tail),
  latest_stages=stages[-3:],exception_classes=errors[-4:],traceback_frames=frames[-8:])
staged=bounded(b/'logs/staged.json');stage=bounded(b/'stage-complete.json')
assert digest(b/'stage-complete.json')==staged['producer_stage_sha256']
assert staged['runtime_source']==h['runtime_source'] and staged['operator_commit']==h['operator_commit']
run=bounded(b/'logs/run-manifest.json',4*1024**2)
assert run['run_id']==stage['run_id'] and run['execution_id']==stage['execution_id']
assert run['source_commit']==h['runtime_source'] and run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None
assert run['actor_training_budget']==run['head_training_budget']==stage['actor_training_budget']==stage['head_training_budget']==0
assert run['diagnostic_selection_sha256']==stage['selection_sha256']
assert stage['config_sha256']==digest(b/'teacher-config.json')
assert stage['dataset_dir']==h['dataset_dir'] and stage['dataset_plan_sha256']==digest(pathlib.Path(h['dataset_dir'])/'plan.json')
assert stage['requested_training_indices']==h['preferred_training_indices']
assert stage['question_count']==102400 and stage['trajectories']==512000
if run.get('gpu_placement_bound'):
 assert run['training_physical_indices']==h['preferred_training_indices']
 assert len(run['training_gpu_uuids'])==len(set(run['training_gpu_uuids']))==4
 assert digest(b/run['gpu_placement_binding_receipt'])==run['gpu_placement_binding_sha256']
 if 'logs/launch.json' in docs:assert docs['logs/launch.json']['training_indices']==run['training_physical_indices']
 launch_claim=bounded(b/'logs/launch-once.json')
 assert launch_claim['training_indices']==run['training_physical_indices'] and launch_claim['training_uuids']==run['training_gpu_uuids']
run_keys=('run_id','execution_id','source_commit','execution_backend','slurm_job_id','training_physical_indices',
 'training_gpu_uuids','auxiliary_physical_indices','assigned_gpu_uuids','gpu_placement_bound',
 'gpu_placement_binding_sha256','diagnostic_selection_sha256','actor_training_budget','head_training_budget')
native=dict(binding_accepted=True,stage_complete_sha256=digest(b/'stage-complete.json'),
 run_manifest_sha256=digest(b/'logs/run-manifest.json'),run_manifest=safe({k:run[k] for k in run_keys if k in run}),
 runtime_source_map_sha256=hashlib.sha256(json.dumps(run['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 stage_complete=safe(stage))
q=b/'logs/pipeline-status.json'
if q.exists():
 value=bounded(q);assert value['run_id']==run['run_id'];native['pipeline_status']=safe(value)
else:native['pipeline_status']=dict(exists=False,stage=None)
ledger_path=b/'logs/process-ledger.json';jobs=[]
if ledger_path.exists():
 ledger=bounded(ledger_path,MAX_DOCUMENT);assert ledger['run_id']==run['run_id'] and len(ledger['jobs'])<=64
 for name,job in ledger['jobs'].items():
  assert re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',name)
  leader={k:job['leader'][k] for k in ('pid','start_ticks','sid','uid')}
  assert leader['uid']==os.getuid() and leader['pid']==leader['sid']
  records=job['ledger'];assert len(records)<=50000
  count=0;unknown=0
  for record in records.values():
   own={k:record[k] for k in ('pid','start_ticks','sid','uid')}
   assert own['sid']==leader['sid'] and own['uid']==leader['uid']
   try:count+=int(alive(own))
   except (PermissionError,OSError):unknown+=1
  jobs.append(dict(name=name,state=job['state'],leader=leader,leader_alive=alive(leader),
   recorded_births=len(records),recorded_births_alive=count,recorded_births_unknown=unknown,exit_code=job.get('exit_code')))
 native['owned_ledger']=dict(exists=True,bytes=ledger_path.stat().st_size,mtime_unix=ledger_path.stat().st_mtime,jobs=jobs)
else:native['owned_ledger']=dict(exists=False,jobs=[])
native['logs']={name:scalar_log(b/'logs'/name) for name in (
 'diagnostic-teacher-driver.log','diagnostic-retriever.log','diagnostic-retrieval-stage.log',
 'teacher.log','worker.log','stage.log','diagnostic-teacher.log','encoder-00.log','encoder-01.log','encoder-02.log','encoder-03.log')}
for name in ('diagnostic-driver-once.json','diagnostic-driver-route.json','diagnostic-driver-exit.json','diagnostic-driver-failure.json','diagnostic-retriever-ready.json'):
 q=b/name
 if q.exists():
  value=bounded(q)
  for key in ('run_id','execution_id'):
   if key in value:assert value[key]==run[key]
  native[name]=safe(value)
value=dict(stage='PREFIX_SCALE20_NATIVE_METADATA_OBSERVATION',checked_unix=time.time(),
 metadata=safe(docs),native=native,latest_markers=safe(markers[-3:]),error_classes=safe(errors[-4:]),gpu_inventory=gpu,
 actor_training_budget=0,head_training_budget=0,read_only=True,resource_cleanup_acceptance=False)
raw=json.dumps(value,allow_nan=False)
assert len(raw.encode())<=128*1024
print(raw,flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    handle=json.loads(args.handle.read_bytes())
    assert sha(CONTROLLER)==handle['operator_sha256'][CONTROLLER.name], 'Use the original handle controller bytes'
    spec=importlib.util.spec_from_file_location('immutable_scale20_controller',CONTROLLER)
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    assert handle['runtime_source']==controller.SOURCE
    marker="\nprint(json.dumps(dict(stage='PREFIX_SCALE20_OBSERVATION'"
    index=controller.OBSERVE.rfind(marker)
    assert index>0
    code=controller.OBSERVE[:index]
    old="gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)"
    assert code.count(old)==1
    code=code.replace(old,old[:-1]+',timeout=15)')+NATIVE
    command=controller.route().ssh_command(code)
    assert command[-1].startswith('python3 -c ')
    command[-1]=controller.PYTHON+' -c '+command[-1][len('python3 -c '):]
    started=time.time()
    receipt=dict(role='read_only_scale20_native_metadata_observer',handle_sha256=sha(args.handle),
        observer_sha256=sha(__file__),original_controller_sha256=sha(CONTROLLER),runtime_source=handle['runtime_source'],
        operator_commit=handle['operator_commit'],started_unix=started,read_only=True)
    try:
        result=subprocess.run(command,input=json.dumps(dict(handle=handle)).encode(),capture_output=True,timeout=65)
        receipt.update(return_code=result.returncode,stdout_bytes=len(result.stdout),stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
        if result.returncode==0:
            assert len(result.stdout)<=MAX_OUTPUT
            receipt['observation']=json.loads(result.stdout)
        else:
            receipt['status']='UNKNOWN_REMOTE_METADATA_READ_FAILED'
            lines=result.stderr.decode(errors='replace').splitlines()
            receipt['exception_classes']=[line.split(':',1)[0] for line in lines if line.startswith(('AssertionError','ValueError','RuntimeError','FileNotFoundError','KeyError','PermissionError'))][-4:]
    except subprocess.TimeoutExpired:
        receipt.update(status='UNKNOWN_SSH_TIMEOUT',retry_launch_forbidden=True)
    receipt['finished_unix']=time.time()
    output=args.output or ROOT/'preparation'/('qprobe-prefix-scale20-native-observation-'+str(time.time_ns())+'.json')
    with output.open('x') as stream:
        json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(receipt=str(output),**receipt),allow_nan=False),flush=True)
    if receipt.get('return_code',1):raise SystemExit(1)


if __name__=='__main__':main()

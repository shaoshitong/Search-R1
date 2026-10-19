"""Code-only isolated full-joint fault recovery using the unchanged SSH route."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from .control import REMOTE as ORIGINAL_REMOTE
from .control import archive_source,route,ROOT

REMOTE=ORIGINAL_REMOTE.split("action=p['action']")[0]+r'''
action=p['action']
def admitted(c):
 result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 assert result.returncode==0
 approved=set(c['physical_gpu_uuids']+c['auxiliary_gpu_uuids'])
 assert len(approved)==6 and c['physical_gpu_indices']==[0,1,2,3] and c['auxiliary_gpu_indices']==[5,6]
 assert not any(line.split(',')[0].strip() in approved for line in result.stdout.splitlines() if line.strip()),'Approved GPU acquired by another task; never stop it'
if action=='stage':
 parent=pathlib.Path(h['parent_handle']['base'])
 assert read(parent/'preparation/deployment.json')==h['parent_handle']
 assert sha(h['terminal_path'])==h['terminal_sha256']
 terminal=read(h['terminal_path']);assert terminal['owned_writers_gone'] and terminal['termination_cause']=='FAULT_EXIT'
 assert same(read(parent/'preparation/submit-receipt.json')['leader']) is None
 assert read(parent/'logs/cleanup-status.json')['owned_births_all_gone']
 assert sha(parent/'latest-joint.json')==terminal['source_joint_sha256']
 assert not b.exists();b.mkdir();repo.mkdir();(b/'preparation').mkdir();(b/'logs').mkdir();(b/'probe').mkdir()
 raw=base64.b64decode(p['archive_b64']);assert hashlib.sha256(raw).hexdigest()==h['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
  members=tf.getmembers();assert all(not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts and (x.isfile() or x.isdir()) for x in members)
  tf.extractall(repo,members=members)
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest,name
 allowed={'reproduction/actor50_grpo/bootstrap.py','reproduction/prefix_selection200/launch.py','reproduction/actor50_grpo/full_joint_resume.py','reproduction/actor50_grpo/full_joint_control.py',
  'reproduction/actor50_grpo/control.py','reproduction/actor50_grpo/local_watch_once.py'}
 for name,digest in h['parent_handle']['source_files'].items():
  if pathlib.Path(name).suffix in ('.py','.sh') and name not in allowed:
   assert h['source_files'].get(name)==digest,'Unrelated scientific source changed: '+name
 # CPU staging/qualification must remain possible while approved GPUs are
 # occupied. The submit gate and native allocation both require them free.
 c=read(parent/'preparation/launch-config.json')
 c.pop('restart_from_actor50',None);c.pop('recovery_sampling',None)
 c.update(base=str(b),execution_id=h['execution_id'],source_commit=h['source_commit'],
  pool_state_path=str(b/'question-pool/state.json'),prefix_probe_receipt=str(b/'probe/latest.json'),
  full_joint_fault_recovery=True,
  resume_joint=dict(adapter='actor50_full_joint_v1',source_base=str(parent),completed_outer=terminal['completed_outer'],
   source_execution_id=h['parent_handle']['execution_id'],source_commit=h['parent_handle']['source_commit'],
   terminal_path=h['terminal_path'],terminal_sha256=h['terminal_sha256'],source_joint_sha256=terminal['source_joint_sha256']),
  rollout_max_num_seqs=64,rollout_max_batched_tokens=2048,rollout_utilization_cap=.7,
  fault_infrastructure_repair=dict(kind='bounded_native_flash_kvcache_workload',old_sequences=128,new_sequences=64,
   old_batched_tokens=4096,new_batched_tokens=2048,old_utilization_cap=.85,new_utilization_cap=.7,
   attention_backend_unchanged=True,enforce_eager=True,prefix_cache_enabled=False))
 import socket
 def ports(start,width):
  for number in range(start,60000-width):
   sockets=[]
   try:
    for i in range(width):
     item=socket.socket();sockets.append(item);item.bind(('127.0.0.1',number+i))
    return number
   except OSError:pass
   finally:
    for item in sockets:item.close()
  raise RuntimeError('No unused loopback ports')
 c['retrieval_port']=ports(19000,1);c['search_url']='http://127.0.0.1:%d/retrieve'%c['retrieval_port']
 c['grounding_probe_base_port']=ports(24000,4)
 once(b/'preparation/launch-config.json',c);once(b/'preparation/deployment.json',h)
 for name in ('human-intent.json','warmup-parent-handle.json'):
  if (parent/'preparation'/name).exists():once(b/'preparation'/name,read(parent/'preparation'/name))
 sys.path.insert(0,str(repo))
 from reproduction.actor50_grpo.bootstrap import validate_config
 validate_config(c,b)
 value=dict(status='FULL_JOINT_RECOVERY_CODE_STAGED_NO_CUDA',completed_outer=terminal['completed_outer'],source_commit=h['source_commit'],checked_unix=time.time())
 once(b/'preparation/source-receipt.json',value)
elif action=='prepare':
 source();c=read(b/'preparation/launch-config.json')
 once(b/'preparation/prepare-full-intent.json',dict(checked_unix=time.time(),source_commit=h['source_commit']))
 value=launch('prepare-full',['env','SEARCH_G1_GRPO_PREFIX=40',str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.full_joint_resume','--base',str(b)],c)
elif action=='submit':
 source();c=read(b/'preparation/launch-config.json');assert read(b/'preparation/prepare-full-exit.json')['success']
 assert same(read(b/'preparation/prepare-full-launch.json')['leader']) is None
 sys.path.insert(0,str(repo));os.environ['SEARCH_G1_GRPO_FROM50']='1';os.environ['SEARCH_G1_GRPO_PREFIX']='20'
 from reproduction.actor50_grpo.full_joint_resume import validate_boundary
 assert validate_boundary(b,c)==c['resume_joint']['completed_outer']
 admitted(c)
 once(b/'preparation/submit-intent.json',dict(config_sha256=sha(b/'preparation/launch-config.json'),checked_unix=time.time()))
 value=launch('pipeline',[str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.run','--base',str(b)],c)
 once(b/'preparation/submit-receipt.json',value)
else:
 source();value=dict(checked_unix=time.time())
 for name in ('prepare-full-exit.json','prepare-progress.json'):
  path=b/'preparation'/name;value[name]=read(path) if path.is_file() else None
 identity_path=b/'preparation/prepare-full-launch.json'
 value['prepare_full_alive']=same(read(identity_path)['leader']) if identity_path.exists() else None
 path=b/'logs/prepare-full.log'
 if path.is_file():
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-8000));value['prepare_full_tail']=f.read(8000).decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('stage','prepare','submit','inspect'),required=True)
    parser.add_argument('--handle',type=Path,required=True);parser.add_argument('--parent-handle',type=Path)
    parser.add_argument('--terminal-path');parser.add_argument('--terminal-sha256');args=parser.parse_args()
    work=Path(__file__).resolve().parents[2];payload=dict(action=args.action)
    if args.action=='stage':
        assert args.parent_handle and args.terminal_path and args.terminal_sha256 and not args.handle.exists()
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=work).strip()
        archive,files=archive_source(work);parent=json.loads(args.parent_handle.read_text());execution=uuid.uuid4().hex
        h=dict(run_id=parent['run_id'],execution_id=execution,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip(),
            base='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+parent['run_id']+'-resume-'+execution,
            source_files=files,archive_sha256=hashlib.sha256(archive).hexdigest(),parent_handle=parent,
            terminal_path=args.terminal_path,terminal_sha256=args.terminal_sha256,created_unix=time.time())
        args.handle.write_text(json.dumps(h,indent=2)+'\n');payload['archive_b64']=base64.b64encode(archive).decode()
    else:h=json.loads(args.handle.read_text())
    payload['handle']=h;result=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),capture_output=True,timeout=65)
    path=ROOT/'preparation'/('actor50-full-joint-'+args.action+'-'+str(time.time_ns())+'.json')
    path.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n')
    if result.returncode:raise RuntimeError('Inspect actual once-operation receipt: '+str(path))
    print(json.dumps(dict(receipt=str(path),result=json.loads(result.stdout))))


if __name__=='__main__':main()

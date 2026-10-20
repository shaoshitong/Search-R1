"""CPU-only final artifact/coverage audit and exact owned-process observations.

No training/evaluation call, CUDA allocation, signal, lock acquisition or cleanup.
All tensors, canonical questions and evaluation rows stay on retained H10018.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .control import OUT, REPO, remote
from .post200_observe import PHYSICAL, CANONICAL, PID, START

SOURCE = '2384d73c8b3626d1a1c05986afac7d8808b5a274'
FILES = ['reproduction/prefix_selection200/'+f+'.py' for f in
         ('actor','probe','eval','launch','ab_lifecycle','ab_policy','prefix_diversity',
          'b_progressive','b_progressive_control','interaction_protocol')]

ARTIFACTS = r'''
import os,pathlib,json,hashlib,time,gc,struct,math
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,torch,pyarrow.parquet as pq
torch.set_num_threads(2)
started=time.time();cpu=time.process_time();p=pathlib.Path(BASE)
def read(f):return json.loads(pathlib.Path(f).read_text())
def physical(f):
 s=str(f);prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
 return pathlib.Path('/mnt/data/shaoshitong')/s[len(prefix):] if s.startswith(prefix) else pathlib.Path(s)
def ident(f):
 s=f.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
files=[]
def prove(f,expected=None,finite_hf=False):
 f=physical(f);before=ident(f);h=hashlib.sha256();finite=None
 with f.open('rb') as stream:
  if finite_hf:
   first=stream.read(8);size=struct.unpack('<Q',first)[0];header=stream.read(size)
   tensors={k:v for k,v in json.loads(header).items() if k!='__metadata__'}
   assert tensors and all(v['dtype']=='F32' for v in tensors.values())
   h.update(first);h.update(header);finite=dict(dtype='F32',tensors=len(tensors),all_finite=True)
  for block in iter(lambda:stream.read(8*1024**2),b''):
   h.update(block)
   if finite_hf:assert bool(np.isfinite(np.frombuffer(block,dtype='<f4')).all())
 assert ident(f)==before
 row=dict(path=str(f),bytes=before[2],sha256=h.hexdigest(),identity=list(before))
 if expected:
  assert row['sha256']==expected['sha256'],str(f)
  if 'bytes' in expected:assert row['bytes']==expected['bytes'],str(f)
 if finite:row['fp32_master_values']=finite
 files.append(row);return row
cpath=p/'preparation/launch-config.json';c=read(cpath);intent=read(p/'preparation/submit-intent.json')
prove(cpath,dict(sha256=intent['config_sha256']))
assert c['source_commit']==SOURCE and c['world_size']==4
assert c['common_manifest_sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert c['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
for name,expected in EXPECTED_SOURCE.items():prove(p/'repo'/name,dict(sha256=expected))
for pathkey,shakey in [('common_manifest_path','common_manifest_sha256'),('composite_manifest_path','composite_manifest_sha256'),('canonical_train','canonical_train_sha256')]:
 prove(c[pathkey],dict(sha256=c[shakey]))
assert pq.ParquetFile(physical(c['canonical_train'])).metadata.num_rows==169028
jointproof=prove(p/'outers/outer-000200/complete.json',dict(sha256=JOINT_SHA))
prove(p/'latest-joint.json',dict(sha256=JOINT_SHA));j=read(p/'outers/outer-000200/complete.json');a=j['actor']
assert j['outer']==200 and a['actual_adam_steps_per_rank']==[4000]*4 and a['scheduler_completed_outer']==200
assert a['new_adam_steps_per_rank']==[20]*4 and a['skipped_update_opportunities']==[0]*4
assert a['actual_global_micro']==4 and a['actual_local_micro']==1 and a['gradient_accumulation']==32 and a['global_mini']==128
rank_states=[]
def finite(t):
 assert t.device.type=='cpu'
 if t.is_floating_point():
  assert t.dtype==torch.float32
  for block in t.reshape(-1).split(2*1024**2):assert bool(torch.isfinite(block).all())
def adam(optimizer,steps,rate):
 assert optimizer['state'] and all(g['lr']==rate for g in optimizer['param_groups'])
 for state in optimizer['state'].values():
  assert int(state['step'])==steps
  for key in ('exp_avg','exp_avg_sq'):finite(state[key])
 return len(optimizer['state'])
for rank,item in enumerate(a['rank_states']):
 prove(item['path'],item);saved=torch.load(physical(item['path']),map_location='cpu',weights_only=False)
 assert saved['rank']==rank and saved['completed_outer']==200 and saved['actual_adam_steps']==4000
 assert saved['scheduler_completed_outer']==saved['scheduler']['last_epoch']==200
 assert saved['model_master_dtype']=='torch.float32' and all(x['dtype']=='torch.float32' for group in saved['optimizer_parameter_layout'] for x in group)
 assert saved['optimizer_layout']=='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4'
 for group,layout in zip(saved['optimizer']['param_groups'],saved['optimizer_parameter_layout']):
  assert len(group['params'])==len(layout)
  for key,shape in zip(group['params'],layout):
   assert list(saved['optimizer']['state'][key]['exp_avg'].shape)==shape['shape']
   assert list(saved['optimizer']['state'][key]['exp_avg_sq'].shape)==shape['shape']
 assert {'python','numpy','torch_cpu','torch_cuda'}<=set(saved['rng'])
 count=adam(saved['optimizer'],4000,1e-6)
 rank_states.append(dict(rank=rank,actual_adam_steps=4000,scheduler=200,fp32_finite_adam_parameters=count,rng_present=True))
 del saved;gc.collect()
hf=physical(a['hf_model_dir']);binding=read(p/'outers/outer-000200/actor-binding.json')
changed=[]
for name,item in a['hf_files'].items():
 prove(hf/name,item,finite_hf=name.endswith('.safetensors'))
 if name.endswith('.safetensors') and binding['actor_model_files_sha256'].get(name)!=item['sha256']:changed.append(name)
assert changed and binding['actor_version'].endswith('actor0199') and a['actor_version'].endswith('actor0200')
prove(a['driver_state']['path'],a['driver_state']);driver=torch.load(physical(a['driver_state']['path']),map_location='cpu',weights_only=False)
poolproof=prove(j['pool_snapshot'],dict(sha256=j['pool_sha256']));pool=read(physical(j['pool_snapshot']))
assert pool['next_outer']==201 and driver['candidate_pool_state']['value']==pool
assert pool['cycle']*len(pool['order'])+pool['cursor']==pool['consumed_candidates']
del driver
probeproof=prove(j['probe_snapshot'],j['probe']);probe=torch.load(physical(j['probe_snapshot']),map_location='cpu',weights_only=False)
assert probe['completed_outer']==200 and probe['actual_adam_steps_per_rank']==[2760]*4 and probe['ema_updates']==2760
assert probe['binding']['prefix_tokens']==80 and len(probe['rank_rng'])==4
for key in ('model','ema_model'):
 assert tuple(probe[key]['output.weight'].shape)==(1,353*2048)
 for tensor in probe[key].values():finite(tensor)
adam(probe['optimizer'],2760,5e-6);del probe;gc.collect()
evaluations={}
for label,root,pathkey,count in [('validation200',p/'outers/outer-000200/validation','validation_questions',512),('final51713',p/'final-evaluation','final_questions',51713)]:
 result=read(root/'complete.json');prove(root/'complete.json');question_path=physical(c[pathkey]);qproof=prove(question_path,dict(sha256=c[pathkey+'_sha256']))
 assert result['qa_count']==count and result['question_source_sha256']==qproof['sha256']
 assert result['actor_version']==a['actor_version'] and result['actual_compute_dtype']=='torch.bfloat16'
 assert result['actual_worker_exit_codes']=={str(i):0 for i in range(4)}
 assert result['evaluation_contract']['temperature']==0 and result['evaluation_contract']['n']==1
 questions=pq.read_table(question_path).to_pylist();assert len(questions)==count
 rows=[]
 for rank,receipt in enumerate(result['rank_receipts']):
  prove(receipt['path'],dict(sha256=receipt['sha256']));sealed=read(physical(receipt['path']))
  assert sealed['rank']==rank and sealed['actor_version']==a['actor_version']
  assert sealed['evaluation_contract']==result['evaluation_contract'] and sealed['interaction_protocol']==result['interaction_protocol']
  assert sealed['question_source_sha256']==qproof['sha256'] and sealed['actual_compute_dtype']=='torch.bfloat16'
  prove(sealed['full_output']['path'],sealed['full_output']);rows+=sealed['rows']
 rows.sort(key=lambda x:x['evaluation_ordinal'])
 assert [x['evaluation_ordinal'] for x in rows]==list(range(count)) and len({x['qid'] for x in rows})==count
 for ordinal,(row,question) in enumerate(zip(rows,questions)):
  extra=question.get('extra_info')or{};qid=question.get('qid',question.get('__g1_question_id'))
  if qid is None:qid=json.dumps([str(question['data_source']),str(extra.get('index',ordinal))])
  assert row['qid']==str(qid) and row['data_source']==question['data_source']
 assert math.isclose(sum(x['em'] for x in rows)/count,result['em'],abs_tol=1e-14)
 assert math.isclose(sum(x['f1'] for x in rows)/count,result['f1'],abs_tol=1e-14)
 for source,metrics in result['per_source'].items():
  subset=[x for x in rows if x['data_source']==source];assert len(subset)==metrics['qa_count']
  for key in ('em','f1'):assert math.isclose(sum(x[key] for x in subset)/len(subset),metrics[key],abs_tol=1e-14)
 evaluations[label]=dict(qa_count=count,unique_qids=count,exact_canonical_qid_order=True,all_rank_and_output_sha_pass=True,
  em=result['em'],f1=result['f1'],per_source=result['per_source'],actual_worker_exit_codes=result['actual_worker_exit_codes'],question_source_sha256=qproof['sha256'])
pipeline=read(p/'logs/pipeline-complete.json');assert pipeline['complete_outers']==200 and pipeline['latest_joint']['outer']==200
assert pipeline['final_evaluation']==read(p/'final-evaluation/complete.json');prove(p/'logs/pipeline-complete.json')
cleanup=read(p/'logs/cleanup-status.json');assert cleanup['cleaned'] and cleanup['owned_births_all_gone'] and not cleanup['remaining_owned_births']
ledger=read(p/'logs/process-ledger.json');births={}
for job in ledger['jobs'].values():
 for rec in job['ledger'].values():births[(rec['pid'],rec['start_ticks'])]={k:rec[k] for k in ('pid','start_ticks','sid','uid')}
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',observed_unix=time.time(),elapsed_seconds=time.time()-started,reader_cpu_seconds=time.process_time()-cpu,
 base=BASE,source_commit=SOURCE,complete_outer=200,joint_sha256=JOINT_SHA,actor_ranks=rank_states,all_hf_fp32_finite=True,
 actual_parameter_bytes_changed_from199=changed,probe=dict(raw_ema_finite=True,adam=2760,ema=2760,prefix=80,world=4),
 cursor=dict(next_outer=201,consumed_candidates=pool['consumed_candidates'],cycle=pool['cycle'],cursor=pool['cursor']),
 files=files,evaluations=evaluations,cleanup=cleanup,owned_births=list(births.values()),
 gpu_uuid_lock_root=c['gpu_uuid_lock_root'],gpu_uuids=c['physical_gpu_uuids'],cuda_initialized=False)))
'''

NODE = r'''
import pathlib,json,time,subprocess,fcntl,struct,os,socket
assert socket.gethostname()=='h100-11'
def state(rec):
 try:
  p=pathlib.Path('/proc')/str(rec['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
  return int(f[19])==rec['start_ticks'] and int(f[3])==rec['sid'] and p.stat().st_uid==rec['uid']
 except (FileNotFoundError,ProcessLookupError):return False
remaining=[r for r in BIRTHS if state(r)]
try:f=(pathlib.Path('/proc')/str(PID)/'stat').read_text().rsplit(')',1)[1].split();native=int(f[19])==START
except (FileNotFoundError,ProcessLookupError):native=False
units={};journal={}
for name in ('sst-b-diverse-cp188-four-h10011-r2.service','sst-b-diverse-cp188-four-h10011-r2-resources.service'):
 q=subprocess.run(['systemctl','show',name,'--property=ActiveState,SubState,MainPID,ExecMainStatus,Result,ControlGroup,CPUUsageNSec'],capture_output=True,text=True,timeout=15)
 units[name]=dict(line.split('=',1) for line in q.stdout.splitlines() if '='in line)
 assert q.returncode==0 and units[name]['MainPID']=='0' and units[name]['ActiveState']=='inactive' and units[name]['ExecMainStatus']=='0' and units[name]['Result']=='success'
 q=subprocess.run(['journalctl','-u',name,'-n','60','--no-pager','-o','json'],capture_output=True,text=True,timeout=15)
 journal[name]=[]
 for line in q.stdout.splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if any(k.startswith('CPU_') for k in r)or any(x in str(r.get('MESSAGE','')) for x in ('Consumed ','Deactivated successfully','Failed with result','Main process exited','Finished ')):
   journal[name].append({k:v for k,v in r.items()if k.startswith(('CPU_','MEMORY_'))or k in ('MESSAGE','__REALTIME_TIMESTAMP','UNIT')})
 q=units[name].get('ControlGroup')
 if q:assert not(pathlib.Path('/sys/fs/cgroup')/q.lstrip('/')/'cgroup.procs').read_text().strip()
locks=[]
for uuid in UUIDS:
 path=pathlib.Path(LOCK_ROOT)/(uuid+'.lock')
 try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 except OSError as error:
  locks.append(dict(uuid=uuid,query_only_no_lock_acquired=True,error=type(error).__name__,own_descriptors_closed_with_verified_process_exit=True));continue
 try:
  result=fcntl.fcntl(fd,fcntl.F_OFD_GETLK,struct.pack('hhqqi4x',fcntl.F_WRLCK,os.SEEK_SET,0,0,0))
  kind,_,_,_,holder=struct.unpack('hhqqi4x',result);locks.append(dict(uuid=uuid,lock_type=kind,unlocked=kind==fcntl.F_UNLCK,holder=holder,query_only_no_lock_acquired=True))
 finally:os.close(fd)
q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
assert not native and not remaining
print(json.dumps(dict(status='PASS',observed_unix=time.time(),native_same_birth=native,remaining_owned_births=remaining,owned_births_checked=len(BIRTHS),
 units=units,journal_resource_records=journal,uuid_lock_observations=locks,gpu_query_returncode=q.returncode,gpu_query_stdout=q.stdout,
 gpu_query_stderr=q.stderr,foreign_processes_modified=False,signal_sent=False,lock_acquired=False)))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('artifacts','node'))
    args=parser.parse_args();evidence=OUT/('final200-'+args.phase+'-'+str(time.time_ns())+'-transport.json')
    if args.phase=='artifacts':
        expected={name:hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name],cwd=REPO)).hexdigest() for name in FILES}
        proof=json.loads((OUT/'CP200_COMPLETE_AUDIT_OBSERVATION1.json').read_text())
        prelude='BASE='+repr(PHYSICAL)+'\nSOURCE='+repr(SOURCE)+'\nEXPECTED_SOURCE='+repr(expected)+'\nJOINT_SHA='+repr(proof['joint']['sha256'])+'\n'
        script="import subprocess,os\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')\nr=subprocess.run(['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-'],input="+repr(prelude+ARTIFACTS)+",text=True,capture_output=True,env=env,timeout=220)\nassert r.returncode==0,r.stderr\nprint(r.stdout)"
        value=json.loads(remote(script,evidence,target='h100-10',established_controller=True))
        path=OUT/'FINAL200_ARTIFACT_AUDIT.json';value['evidence']=str(evidence);path.write_text(json.dumps(value,indent=2))
        print(json.dumps(dict(status=value['status'],elapsed_seconds=value['elapsed_seconds'],files=len(value['files']),qa=value['evaluations']['final51713']['qa_count'],evidence=str(path))))
    else:
        proof=json.loads((OUT/'FINAL200_ARTIFACT_AUDIT.json').read_text())
        from . import read_ip14_live
        original=read_ip14_live.SCRIPT
        read_ip14_live.SCRIPT='BIRTHS='+repr(proof['owned_births'])+'\nUUIDS='+repr(proof['gpu_uuids'])+'\nLOCK_ROOT='+repr(proof['gpu_uuid_lock_root'])+'\n'+NODE
        try:value=read_ip14_live.read(CANONICAL,PID,START)
        finally:read_ip14_live.SCRIPT=original
        path=OUT/('FINAL200_NODE_AUDIT_'+str(time.time_ns())+'.json');path.write_text(json.dumps(value,indent=2))
        print(json.dumps(value))


if __name__=='__main__':main()

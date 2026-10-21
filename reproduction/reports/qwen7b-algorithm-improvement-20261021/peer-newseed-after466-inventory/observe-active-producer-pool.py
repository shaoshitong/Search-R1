"""One read-only aggregate of real generation workers and declared leases."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKER = '''import pathlib,json,hashlib,time,os,resource
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');now=time.time();node=os.uname().nodename
allow={'status','stage','state','phase','epoch','updated_epoch','heartbeat_epoch','created_epoch','captured_epoch','node','host','hostname','owner_host','pid','main_pid','controller_pid','controller_birth','start_ticks','birth','worker_pid','worker_pids','child_pid','child_pids','worker','owner','source_owner','model_process','model_processes','lease','lease_id','slot','slot_id','slot_name','job_id','run_id','deadline_epoch','expires_epoch','expiry_epoch','expiration_epoch','hard_end_epoch','consumer_deadline_epoch','lease_deadline_epoch','generation','model_indices','model_engine_count','native_unit','unit','allocation','controller','ready','questions','trajectories','completed_own_batches','completed_batches','global_goal_complete','latest_input','latest_batch_seconds','longest_batch_seconds','active','busy','available','expired','reason','claimed','returncode','sampler_returncode','heartbeat','current_job','current_batch','batch','attempt','leases','slots','workers','transport','model_slot','parent','registration','bound_node','bound_pid','current_owner','producer','last_completed_epoch','last_progress_epoch'}
def safe(v,depth=0):
 if depth>5:return None
 if isinstance(v,dict):return{k:safe(x,depth+1)for k,x in v.items()if k in allow}
 if isinstance(v,list):return[safe(x,depth+1)for x in v[:12]]
 return v if isinstance(v,(str,int,float,bool,type(None)))else None
def readmeta(p):
 st=p.stat()
 if st.st_size>128*1024:return None
 raw=p.read_bytes()
 try:v=json.loads(raw)
 except ValueError:return None
 if not isinstance(v,dict):return None
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime=st.st_mtime,age_seconds=now-st.st_mtime,fields=safe(v),root_keys=sorted(v))
leases=[];registries=[]
for name in ['gpu-slot-leases','relay-live-slots','gpu-subleases','retrieval-relay-leases']:
 root=B/'pool'/name;files=[p for p in root.rglob('*')if p.is_file()and not any(s in p.name.lower()for s in ['secret','token','credential'])]
 registries.append(dict(name=name,files=len(files)))
 for path in sorted(files,key=lambda p:p.stat().st_mtime,reverse=True)[:32]:
  row=readmeta(path)
  if row:row['registry']=name;leases.append(row)
active=[];terminal_count=0;stale_count=0
for job in (B/'pool/jobs').iterdir():
 if not job.is_dir():continue
 attempts=[p for p in job.glob('attempt-*')if p.is_dir()]
 if not attempts:continue
 attempt=max(attempts,key=lambda p:p.stat().st_mtime)
 if(attempt/'complete.json').exists()or(attempt/'failure.json').exists():terminal_count+=1;continue
 names=['progress.json','heartbeat.json','worker-heartbeat.json','controller.json','allocation.json','registration.json','owner-ready.json','relay-owner-ready.json']
 rows=[]
 for name in names:
  path=attempt/name
  if path.exists():
   row=readmeta(path)
   if row:rows.append(row)
 if not rows:continue
 newest=max(r['mtime']for r in rows)
 if now-newest>3600:stale_count+=1
 entry=dict(job=job.name,attempt=attempt.name,last_metadata_epoch=newest,last_metadata_age_seconds=now-newest,terminal_receipt_present=False,metadata=rows)
 # Same-host PIDs only; remote workers are reported from their bound receipts.
 for row in rows:
  f=row['fields'];pid=f.get('controller_pid')or f.get('pid');host=f.get('node')or f.get('host')or f.get('hostname')
  if type(pid)is int and host==node:
   proc=pathlib.Path('/proc')/str(pid);entry['same_host_pid_exists']=proc.exists()
   if(proc/'stat').exists():entry['same_host_actual_birth']=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
 active.append(entry)
active.sort(key=lambda v:v['last_metadata_epoch'],reverse=True)
seals=list((B/'sampling').glob('shard-*/batches/batch*/complete.json'));questions=0;latest=[]
for path in seals:
 raw=path.read_bytes();v=json.loads(raw);assert v['run_id']=='70188b8e50fd4b45bc420ce58014f46e';questions+=v['questions']
 latest.append(dict(path=str(path.relative_to(B)),mtime=path.stat().st_mtime,sha256=hashlib.sha256(raw).hexdigest(),questions=v['questions']))
latest.sort(key=lambda x:x['mtime'],reverse=True)
print(json.dumps(dict(status='READONLY_ACTIVE_PRODUCER_POOL_METADATA_SINGLE_AGGREGATE',epoch=now,node=node,lease_registry_counts=registries,lease_metadata=leases,nonterminal_job_candidates=active[:20],nonterminal_job_count=len(active),nonterminal_older_than_1h=stale_count,terminal_jobs_skipped=terminal_count,sealed_batches=len(seals),sealed_questions=questions,remaining_questions_to169028=169028-questions,latest_seals=latest[:4],pool_worker_PIDs_on_other_hosts_not_directly_probed=True,lease_free_slot_claim_not_inferred_from_missing_json=True,frozen_training_cohort=124194,frozen_cohort_expanded=False,source_modified=False,GPU_operation=False,launch_invoked=False,raw_payload_read=False),sort_keys=True))
'''

def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('h100-10', 'python3 -c '+shlex.quote(WORKER)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-700:])
    value = json.loads(result.stdout.decode().splitlines()[-1])
    (HERE/'actual-active-producer-pool.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, sort_keys=True))

if __name__ == '__main__':
    main()

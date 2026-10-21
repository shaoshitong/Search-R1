"""Read-only existing gpu4 allocation/guard and current process ownership."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKER = '''import pathlib,json,hashlib,time,subprocess,os
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');now=time.time()
allow={'status','role','mode','epoch','captured_epoch','node','host','pid','main_pid','controller_pid','controller_birth','birth','start_ticks','unit','native_unit','unit_hard_end_epoch','source_model_deadline_epoch','deadline_epoch','consumer_deadline_epoch','hard_end_epoch','owner_hard_seconds','slot_hard_seconds','owner_seconds','slot','anchor_slot','anchor_unit','anchor_terminal_required','source_parent_owned','model_engine_count','model_indices','tensor_parallel_size','model_cuda_visible_devices','model_cuda_uuids','model_processes_owned','foreign_processes_signalled','primary_gpu_locks_modified','local_relay_process','slot_lease_paths','source_owner','owner','supplementary_specification','supplementary_model_lease','source_model_deadline_epoch','source_retrieval_owner','source_owner_ready','source_owner_allocation_sha256','resource_limits','RuntimeMaxUSec','MemoryMax','CPUQuotaPerSecUSec','AllowedCPUs','original_cuda_visible_devices','unit_show','Id','ActiveState','SubState','MainPID','ExecMainStatus','Result','indices','cuda_uuids','device','source_allocation','finite_slot','run_id','source_processes_owned','shared_retriever_owned_by_consumer','physical_model_indices','finite_node10_slots','supplementary_slots','generation','instance','slot_name','lifecycle_anchor_is_model_gpu','no_slurm_changes','old_stages_unchanged','owner_registry_unchanged','original_protocol_unchanged'}
def safe(v,depth=0):
 if depth>6:return None
 if isinstance(v,dict):return{k:safe(x,depth+1)for k,x in v.items()if k in allow or (depth>0 and k.startswith('gpu'))}
 if isinstance(v,list):return[safe(x,depth+1)for x in v[:20]]
 return v if isinstance(v,(str,int,float,bool,type(None)))else None
paths=[B/'retrieval-relay-generation/launch/native-3731224/allocation-evidence.json',B/'pool/jobs/job-relay-slot-gpu4-tp1-3731224/attempt-1792555662309369517-3731224/allocation.json',B/'preparation/relay-generation-queued-supp-v1-slot-gpu4-tp1-r1/4f2b3b3581d755ff6d851ba8ea36eb443ba48174fe6d25f7b89fc1e7bd7fd032/manifest.json']
receipts=[]
for path in paths:
 raw=path.read_bytes();v=json.loads(raw);receipts.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),fields=safe(v)))
root=B/'preparation/relay-generation-queued-supp-v1-slot-gpu4-tp1-r1/4f2b3b3581d755ff6d851ba8ea36eb443ba48174fe6d25f7b89fc1e7bd7fd032';code=[]
for name in ['node10_supplementary_control.py','relay_slot_consumer_worker.py','queued_relay_slots.py']:
 path=root/name
 if not path.exists():continue
 raw=path.read_bytes();lines=raw.decode().splitlines();chosen=[]
 for i,line in enumerate(lines,1):
  if any(x in line for x in ['slot_hard_seconds','RuntimeMax','supplement','deadline','require(','add_argument(','primary_lock','lease','source_parent_owned','owner','controller_status'])and not any(x in line.lower()for x in ['secret','token','authorization']):chosen.append(dict(line=i,code=line[:240]))
 code.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),selected_guard_lines=chosen))
def query(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 return dict(exit_code=p.returncode,rows=p.stdout.strip().splitlines(),stderr_present=bool(p.stderr.strip()))
gpu=query(['nvidia-smi','-i','4','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'])
apps=query(['nvidia-smi','-i','4','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']);owners=[]
for line in apps['rows']:
 fields=[s.strip()for s in line.split(',')]
 if len(fields)<3 or not fields[1].isdigit():continue
 pid=int(fields[1]);p=pathlib.Path('/proc')/str(pid)
 if not p.exists():continue
 d=dict(pid=pid,process_name=fields[2],uid=p.stat().st_uid)
 if(p/'stat').exists():s=(p/'stat').read_text().rsplit(')',1)[1].split();d.update(state=s[0],birth_ticks=int(s[19]),parent_pid=int(s[1]))
 if(p/'cgroup').exists():d['cgroup']=(p/'cgroup').read_text().strip().splitlines()
 owners.append(d)
print(json.dumps(dict(status='READONLY_EXISTING_SUPPLEMENTARY_SLOT_BOUNDARY_AND_GPU4_OWNERSHIP',epoch=now,existing_allocation_receipts=receipts,guard_source=code,gpu4_query=gpu,gpu4_compute_processes=apps,gpu4_process_ownership=owners,empty_GPU_is_not_resource_authorization=True,launch_invoked=False,producer_or_lease_modified=False,other_thread_contacted=False),sort_keys=True))
'''

def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('h100-10', 'python3 -c '+shlex.quote(WORKER)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-700:])
    value = json.loads(result.stdout.decode().splitlines()[-1])
    (HERE/'actual-supp-lease-boundary.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, sort_keys=True))

if __name__ == '__main__':
    main()

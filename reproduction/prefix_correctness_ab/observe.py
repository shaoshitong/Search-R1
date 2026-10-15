"""Read-only bounded scalar progress for the one offline prefix10/20 job."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time


def small(path,limit=262144):
    p=Path(path)
    if not p.is_file():return None
    if p.stat().st_size>limit:return {'metadata_omitted':'exceeds bounded scalar size','bytes':p.stat().st_size}
    try:return json.loads(p.read_text())
    except (ValueError,OSError) as e:return {'metadata_read_error':str(e)}


def tail(path,limit=65536):
    p=Path(path)
    if not p.is_file():return None
    with p.open('rb') as f:
        size=p.stat().st_size;f.seek(max(0,size-limit));data=f.read()
    for line in reversed(data.splitlines()):
        try:return json.loads(line)
        except ValueError:continue
    return None


def scalars(value,depth=0):
    if depth>5:return 'nested metadata omitted'
    if isinstance(value,float) and not math.isfinite(value):return 'nonfinite diagnostic: '+repr(value)
    if isinstance(value,(bool,int,float)) or value is None:return value
    if isinstance(value,str):return value[:2000]
    if isinstance(value,dict):return {str(k):scalars(v,depth+1) for k,v in list(value.items())[:100]}
    if isinstance(value,list) and len(value)<=32:return [scalars(v,depth+1) for v in value]
    return 'non-scalar array omitted'


def observe(base):
    from reproduction.a800.owned_processes import same,identity
    from reproduction.common8.native_allocation import fields
    base=Path(base).resolve()
    if socket.gethostname().split('.')[0]!='h100-8':raise ValueError('OBSERVE_ACTUAL_BIRTHS_ON_H1008_ONLY')
    c=small(base/'preparation/launch-config.json');receipt=small(base/'preparation/submit-receipt.json')
    result=dict(base=str(base),run_id=(c or {}).get('run_id'),checked_unix=time.time(),submit_receipt=receipt,groups={})
    ledger=small(base/'logs/process-ledger.json',limit=4*1024*1024)
    births=[r for j in (ledger or {}).get('jobs',{}).values() for r in j.get('ledger',{}).values()]
    from reproduction.prefix_correctness_ab.launch import birth_states
    present=birth_states(births)
    result['owned_births_recorded']=len(births);result['owned_births_present']=len(present)
    result['owned_active_births']=sum(r['state']!='Z' for r in present)
    result['owned_zombie_births']=sum(r['state']=='Z' for r in present)
    result['owned_births_still_live']=result['owned_active_births']
    result['owned_births_all_gone']=not present
    for prefix in (10,20):
        root=base/'groups'/('prefix-'+str(prefix));feature=root/'features';train=root/'train'
        result['groups'][str(prefix)]=dict(physical_gpu_indices=list(range(4)) if prefix==10 else list(range(4,8)),
            feature_progress={str(rank):small(feature/('progress-rank-'+str(rank)+'.json')) for rank in range(4)},
            feature_complete=small(feature/'features-complete.json'),
            feature_errors={str(rank):small(feature/('feature-failure-rank-'+str(rank)+'.json')) for rank in range(4)},
            train_progress=tail(train/'progress.jsonl'),latest_epoch_metrics=tail(train/'metrics.jsonl'),
            train_result=small(train/'RESULT.json'),train_error=small(train/'ERROR.json'),
            train_rank_errors={str(rank):small(train/('rank-'+str(rank)+'-error.json')) for rank in range(4)})
    result['scan_complete']=small(base/'prepared-inputs/scan-complete.json')
    result['cleanup']=small(base/'logs/cleanup-status.json');result['pipeline_complete']=small(base/'logs/pipeline-complete.json')
    job=(receipt or {}).get('job_id');result['native_terminal_verified']=False;result['native_resource_closure_verified']=False
    if job:
        try:
            raw=subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True,stderr=subprocess.PIPE,timeout=30)
            j=fields(raw)
            if j.get('JobId')!=str(job):raise ValueError('NATIVE_JOB_ID_CHANGED')
            result['native']=dict(job_id=str(job),state=j.get('JobState'),exit_code=j.get('ExitCode'),node=j.get('NodeList'))
            result['native_terminal_verified']=j.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE')
            if result['native_terminal_verified']:
                result['native_terminal_raw']=raw
                cgroup=list(Path('/sys/fs/cgroup').glob('**/job_'+str(job)))+list(Path('/sys/fs/cgroup').glob('**/job'+str(job)))
                gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=30).split()
                result['native_cgroup_exists']=bool(cgroup)
                owned_contexts=[]
                for pid in set(gpu)&{str(r['pid']) for r in births}:
                    try:actual=identity(int(pid))
                    except (FileNotFoundError,ProcessLookupError):owned_contexts.append(pid);continue
                    if any(str(r['pid'])==pid and all(actual[k]==r[k] for k in ('start_ticks','sid','uid')) for r in births):owned_contexts.append(pid)
                result['recorded_gpu_pids_present']=sorted(owned_contexts)
                result['native_resource_closure_verified']=bool(result['cleanup'] and result['cleanup'].get('cleaned') is True and
                    result['owned_births_all_gone'] and not cgroup and not result['recorded_gpu_pids_present'])
                result['native_actual_exit0_verified']=j.get('JobState')=='COMPLETED' and j.get('ExitCode')=='0:0'
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            result['native_observation_unavailable']=str(e)
            result['native_terminal_verified']=False
    return scalars(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();print(json.dumps(observe(a.base),allow_nan=False))

if __name__=='__main__':main()

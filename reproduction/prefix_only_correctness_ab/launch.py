"""Once native job: prefix10/20/40/80 on physical0-1/2-3/4-5/6-7; offline probes only."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

ROLE='prefix_only_four_correctness_balanced_warmup20'
PREFIXES=(10,20,40,80)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def need(condition,message):
    if not condition:raise ValueError(message)

def once(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def config(base):
    base=Path(base).resolve();p=base/'preparation/launch-config.json';c=read(p)
    need(c['base']==str(base) and re.fullmatch('[0-9a-f]{32}',c['run_id']) and re.fullmatch('[0-9a-f]{40}',c['source_commit']),'CONFIG_IDENTITY_CHANGED')
    need(str(base).startswith('/mnt/data/shaoshitong/search-g1-prefix-only-four-warmup20-'),'OUTPUT_MUST_BE_SEPARATE_REMOTE_NAMESPACE')
    need(len(c['physical_gpu_uuids'])==len(set(c['physical_gpu_uuids']))==8,'EIGHT_ACTUAL_UUIDS_REQUIRED')
    need(len(c['master_ports'])==4 and len(set(c['master_ports']))==4 and all(type(v) is int and 1024<=v<=65535 for v in c['master_ports']),'FOUR_DISTINCT_MASTER_PORTS_REQUIRED')
    need(Path(c['environment_root']).is_absolute() and (Path(c['environment_root'])/'bin/python').is_file(),'EXISTING_REMOTE_ENV_REQUIRED')
    need(Path(c['model_path']).is_absolute() and Path(c['sources_json']).is_absolute(),'REMOTE_INPUT_PATHS_REQUIRED')
    need(sha(c['sources_json'])==c['sources_json_sha256'],'SOURCE_DATA_BINDING_CHANGED')
    return c


def sbatch_command(base,c):
    return ['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1',
        '--cpus-per-task=96','--mem=512G','--gres=gpu:8','--time=2-00:00:00','--signal=B:TERM@60','--export=NONE',
        '--job-name=g1-prefix-only-four-'+c['run_id'][:12],'--comment=g1-prefix-only-four:'+c['run_id'],
        '--chdir='+str(base),'--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),
        str(base/'preparation/native-launch.sh')]


def native_matches(raw,base,intent):
    from reproduction.common8.native_allocation import fields
    j=fields(raw)
    return j.get('JobName')==intent['job_name'] and j.get('Comment')==intent['comment'] and j.get('WorkDir')==str(base) and j.get('Command')==str(base/'preparation/native-launch.sh') and j.get('NodeList') in ('h100-8','(null)')


def query(base):
    base=Path(base).resolve();intent_path=base/'preparation/submit-intent.json'
    if not intent_path.exists():return dict(status='NOT_SUBMITTED',submit_attempted=False)
    intent=read(intent_path);need(intent['role']==ROLE and intent['base']==str(base),'SUBMIT_INTENT_IDENTITY_CHANGED')
    receipt_path=base/'preparation/submit-receipt.json';receipt=read(receipt_path) if receipt_path.exists() else None
    if receipt:need(receipt['intent_sha256']==sha(intent_path),'SUBMIT_RECEIPT_BINDING_CHANGED')
    queue=subprocess.check_output(['squeue','-h','-o','%i|%j|%T'],text=True,timeout=30)
    candidates=[r.split('|')[0] for r in queue.splitlines() if len(r.split('|'))==3 and r.split('|')[1]==intent['job_name']]
    if receipt and receipt.get('job_id') and str(receipt['job_id']) not in candidates:candidates.append(str(receipt['job_id']))
    matches=[]
    for job in candidates:
        try:raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=30)
        except subprocess.CalledProcessError:continue
        from reproduction.common8.native_allocation import fields
        if fields(raw).get('JobId')==job and native_matches(raw,base,intent):matches.append(dict(job_id=job,native=raw))
    need(len(matches)<=1,'MULTIPLE_MATCHING_NATIVE_JOBS_INSPECT')
    return dict(status='RECONCILED_EXISTING_JOB' if matches else 'UNKNOWN_NO_REPLAY',submit_attempted=True,
        receipt=receipt,matching_jobs=matches,replay_allowed=False)


def submit(base):
    base=Path(base).resolve();root=base/'preparation';lock=(root/'submit-control.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'submit-intent.json').exists():return query(base)
        need(not (root/'submit-receipt.json').exists() and not (root/'native-launch.sh').exists(),'PARTIAL_SUBMISSION_INSPECT_NO_REPLAY')
        c=config(base)
        # Only published code, no model weights or input rollouts are copied here.
        source=read(root/'source-files.json')
        need(sha(base/'repo/reproduction/prefix_only_correctness_ab/launch.py')==source['reproduction/prefix_only_correctness_ab/launch.py'],'LAUNCHER_SOURCE_CHANGED')
        launch=root/'native-launch.sh';python=str(Path(c['environment_root'])/'bin/python')
        lines=['#!/bin/bash','set -euo pipefail','export PATH='+shlex.quote(str(Path(c['environment_root'])/'bin')+':/usr/local/bin:/usr/bin:/bin'),
            'export PYTHONPATH='+shlex.quote(str(base/'repo')),'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
            'unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy',
            'exec '+shlex.join([python,'-m','reproduction.prefix_only_correctness_ab.launch','run','--base',str(base)])]
        with launch.open('x') as f:f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
        launch.chmod(0o700);command=sbatch_command(base,c)
        intent=dict(role=ROLE,base=str(base),run_id=c['run_id'],source_commit=c['source_commit'],config_sha256=sha(root/'launch-config.json'),
            command=command,job_name='g1-prefix-only-four-'+c['run_id'][:12],comment='g1-prefix-only-four:'+c['run_id'],launch_sha256=sha(launch),created_unix=time.time())
        once(root/'submit-intent.json',intent)
        try:
            r=subprocess.run(command,capture_output=True,text=True,timeout=30)
            receipt=dict(status='REJECTED_NO_REPLAY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent_sha256=sha(root/'submit-intent.json'))
            if r.returncode==0:
                job=r.stdout.strip().split(';')[0]
                if job.isdecimal():receipt.update(status='SUBMITTED',job_id=job)
                else:receipt['status']='UNKNOWN_NO_REPLAY'
        except (subprocess.TimeoutExpired,OSError) as e:receipt=dict(status='UNKNOWN_NO_REPLAY',error=repr(e),intent_sha256=sha(root/'submit-intent.json'))
        once(root/'submit-receipt.json',receipt);return receipt


def native_admission(base,c):
    from reproduction.common8.native_allocation import fields
    from reproduction.common8.stage_retrieval import memory_bytes
    from reproduction.a800.owned_processes import same
    need(socket.gethostname().split('.')[0]=='h100-8' and os.environ.get('SLURM_JOB_ID'),'NATIVE_H1008_REQUIRED')
    job=os.environ['SLURM_JOB_ID'];raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=30);j=fields(raw)
    need(j['JobId']==job and j['NodeList']=='h100-8' and j['JobState']=='RUNNING' and j['NumNodes']=='1','ACTUAL_NATIVE_JOB_CHANGED')
    need(int(j['NumCPUs'])>=96 and memory_bytes(j['TRES'])>=512*2**30,'NATIVE_CPU_MEMORY_BUDGET_REQUIRED')
    need(j.get('OverSubscribe')!='NO' and j.get('TresPerNode') in ('gres:gpu:8','gres:gpu:h100:8'),'SHARED_EIGHT_GPU_GRES_REQUIRED')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3,4,5,6,7','PHYSICAL_SPLIT_CHANGED')
    queue=subprocess.check_output(['squeue','-h','-w','h100-8','-t','RUNNING,COMPLETING','-o','%i'],text=True,timeout=30).split()
    allowed={str(r['job_id']):r for r in c.get('preserved_foreign_allocations',[])}
    need(set(queue)-{job}<=set(allowed),'UNBOUND_FOREIGN_NATIVE_JOB_PRESERVE_AND_INSPECT')
    for foreign in set(queue)-{job}:
        actual=fields(subprocess.check_output(['scontrol','show','job',foreign,'-o'],text=True,timeout=30));expected=allowed[foreign]
        need(actual['NodeList']=='h100-8' and actual['UserId']==expected['user_id'] and actual['StartTime']==expected['start_time'] and actual['Command']==expected['command'],'FOREIGN_NATIVE_BIRTH_CHANGED')
    rows=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=30))))
    physical={int(r[0]):r[1].strip() for r in rows};need([physical[i] for i in range(8)]==c['physical_gpu_uuids'],'ACTUAL_UUIDS_CHANGED')
    allowed_apps={(r['gpu_uuid'],str(r['identity']['pid'])) for r in c.get('preserved_foreign_gpu_processes',[]) if same(r['identity'])}
    apps=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=30))))
    need(all((r[0].strip(),r[1].strip()) in allowed_apps for r in apps if r),'UNBOUND_FOREIGN_GPU_PROCESS_PRESERVE_AND_INSPECT')
    result=dict(role=ROLE,run_id=c['run_id'],slurm_job_id=job,node='h100-8',actual_native=raw,
        actual_physical_gpu_uuids=c['physical_gpu_uuids'],groups={str(prefix):c['physical_gpu_uuids'][i*2:(i+1)*2] for i,prefix in enumerate(PREFIXES)},
        allocation_mode='shared',gpu_gres_reserved=True,accepted=True,checked_unix=time.time())
    once(base/'logs/native-allocation.json',result);return result


def birth_states(records):
    from reproduction.a800.owned_processes import identity
    result=[]
    for record in records:
        try:current=identity(record['pid'])
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if all(current[k]==record[k] for k in ('start_ticks','sid','uid')):result.append(current)
    return result


def torchrun_owned_class():
    """Elastic ranks use new sessions; bind actual PPID ancestry, not just SID."""
    from reproduction.a800.owned_processes import OwnedProcesses,identity,same,signal_owned
    class TorchrunOwnedProcesses(OwnedProcesses):
        def __init__(self,*args):
            self.handles={}
            super().__init__(*args)
        def start(self,name,*args):
            process=super().start(name,*args);self.handles[name]=process;return process
        def reap(self):
            for process in self.handles.values():
                if process.poll() is not None:process.wait()
        def observe(self,name):
            self.reap();job=self.jobs[name];snapshot={}
            for path in Path('/proc').iterdir():
                if not path.name.isdecimal():continue
                try:
                    record=identity(int(path.name));raw=(path/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
                    if int(parts[19])!=record['start_ticks']:continue
                    record['ppid']=int(parts[1]);snapshot[record['pid']]=record
                except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            bound={int(pid) for pid,record in job['ledger'].items() if record['pid'] in snapshot and
                all(snapshot[record['pid']][k]==record[k] for k in ('start_ticks','sid','uid'))}
            changed=True
            while changed:
                changed=False
                for pid,record in snapshot.items():
                    if pid in bound or record['ppid'] not in bound or record['uid']!=job['leader']['uid']:continue
                    parent=snapshot[record['ppid']]
                    if record['start_ticks']<parent['start_ticks']:continue
                    record['ownership_parent_birth']={k:parent[k] for k in ('pid','start_ticks','sid','uid')}
                    bound.add(pid);changed=True
            for pid in bound:
                current=snapshot[pid];old=job['ledger'].get(str(pid))
                if old and any(current[k]!=old[k] for k in ('start_ticks','sid','uid')):raise ValueError('OWNED_BIRTH_CHANGED')
                if old and 'ownership_parent_birth' in old:current['ownership_parent_birth']=old['ownership_parent_birth']
                job['ledger'][str(pid)]=current
            self.save()
            return [snapshot[pid] for pid in bound if snapshot[pid]['state']!='Z']
        def stop(self,name):
            job=self.jobs[name]
            if job['state']=='STOPPED':self.reap();return
            try:
                job['state']='CLEANUP_PENDING';self.save()
                for sig,seconds in ((signal.SIGINT,15),(signal.SIGTERM,3),(signal.SIGKILL,3)):
                    for record in self.observe(name):
                        try:signal_owned(record,sig)
                        except ProcessLookupError:pass
                    deadline=time.monotonic()+seconds
                    while self.observe(name) and time.monotonic()<deadline:time.sleep(.2)
                    if not self.observe(name):break
                self.reap();need(not self.observe(name),'OWNED_ACTIVE_PROCESS_CLEANUP_INCOMPLETE')
                job['state']='STOPPED';self.save()
            except Exception as e:
                job.update(state='CLEANUP_FAILED',cleanup_error=repr(e));self.save();raise
    return TorchrunOwnedProcesses


class DeferredInterrupt:
    def __init__(self):self.requested=None
    def handler(self,sig,frame):self.requested=sig
    def check(self):
        if self.requested is not None:raise KeyboardInterrupt('native signal '+str(self.requested))


def start_owned(owned,interrupt,name,command,env,log,cwd):
    process=owned.start(name,command,env,str(log),str(cwd));owned.observe(name);interrupt.check();return process


def wait_one(owned,interrupt,name,process):
    while process.poll() is None:owned.observe(name);interrupt.check();time.sleep(.25)
    owned.observe(name);interrupt.check();need(process.returncode==0,'STAGE_FAILED_'+name);owned.stop(name)


def import_prepared_inputs(c,base):
    """Bind staged short inputs without rescanning raw teacher or model weights."""
    from reproduction.prefix_only_correctness_ab.features import validate_prepared_inputs
    sources=read(c['sources_json']);record=sources['prepared_scan']
    source=Path(c.get('prepared_inputs_source') or record['path'])
    need(source.is_absolute() and source.resolve()==Path(record['path']).resolve(),'PREPARED_INPUTS_SOURCE_BINDING_CHANGED')
    proof,relocation=validate_prepared_inputs(source,c['sources_json'],c['model_path'],verify_model=True)
    target=base/'prepared-inputs';need(not target.exists() and not target.is_symlink(),'PREPARED_INPUTS_TARGET_OCCUPIED_INSPECT')
    target.symlink_to(source.resolve(),target_is_directory=True)
    once(base/'logs/prepared-input-import.json',dict(role='remote_short_input_import',source=str(source.resolve()),target=str(target),
        scan_sha256=record['scan_complete_sha256'],relocation=relocation,
        raw_scan_repeated=False,encoder_model_full_sha_verified_once=True,all_short_input_files_full_sha_match=True,
        encoder_global_max_tokens=proof['encoder_global_max_tokens'],head_max_tokens=proof['head_max_tokens']))
    return proof['head_max_tokens']


def torchrun(c,prefix,stage,max_tokens,base):
    group=base/'groups'/('prefix-'+str(prefix));python=str(Path(c['environment_root'])/'bin/python')
    command=[python,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2','--node_rank=0','--master_addr=127.0.0.1',
        '--master_port='+str(c['master_ports'][PREFIXES.index(prefix)]),'-m','reproduction.prefix_only_correctness_ab.'+stage]
    if stage=='features':command+=['--sources',c['sources_json'],'--inputs',str(base/'prepared-inputs'),'--output',str(group/'features'),
        '--prefix-length',str(prefix),'--model',c['model_path'],'--max-tokens',str(max_tokens),'--micro-batch-size','8']
    else:command+=['--cache',str(group/'features'),'--output',str(group/'train'),'--model-dir',c['model_path'],'--prefix-tokens',str(prefix),'--sampling','balanced']
    return command


def run(base):
    from reproduction.a800.owned_processes import same,identity
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base=Path(base).resolve();c=config(base);root=base/'preparation'
    intent=read(root/'submit-intent.json');need(intent['config_sha256']==sha(root/'launch-config.json'),'SUBMITTED_CONFIG_CHANGED')
    need(not (base/'logs/process-ledger.json').exists(),'EXISTING_PROCESS_LEDGER_INSPECT_NO_REPLAY')
    native_admission(base,c);lease=UUIDLocks(c['physical_gpu_uuids']);owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt();previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in previous:signal.signal(s,interrupt.handler)
    states={};success=False
    try:
        env=os.environ.copy();env.update(PYTHONPATH=str(base/'repo'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',NCCL_NVLS_ENABLE='0')
        for key in ('RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
        if c.get('prepared_inputs_source') or read(c['sources_json']).get('prepared_scan'):
            max_tokens=import_prepared_inputs(c,base)
        else:
            scanenv=dict(env,CUDA_VISIBLE_DEVICES='');python=str(Path(c['environment_root'])/'bin/python')
            scan=[python,'-m','reproduction.prefix_only_correctness_ab.features','--scan-only','--sources',c['sources_json'],'--output',str(base/'prepared-inputs'),'--model',c['model_path']]
            p=start_owned(owned,interrupt,'scan',scan,scanenv,base/'logs/scan.log',base/'repo');wait_one(owned,interrupt,'scan',p)
            max_tokens=read(base/'prepared-inputs/scan-complete.json')['head_max_tokens']
        need(type(max_tokens) is int and max_tokens==80,'PREFIX_HEAD_MAX_TOKENS_INVALID')
        for prefix in PREFIXES:
            group=base/'groups'/('prefix-'+str(prefix));group.mkdir(parents=True,exist_ok=False)
            groupenv=dict(env,CUDA_VISIBLE_DEVICES=','.join(c['physical_gpu_uuids'][PREFIXES.index(prefix)*2:(PREFIXES.index(prefix)+1)*2]))
            name='features-prefix'+str(prefix)
            p=start_owned(owned,interrupt,name,torchrun(c,prefix,'features',max_tokens,base),groupenv,base/'logs'/(name+'.log'),base/'repo')
            states[prefix]=dict(stage='features',name=name,process=p,env=groupenv)
        while any(s['stage']!='complete' for s in states.values()):
            interrupt.check()
            for prefix,s in states.items():
                if s['stage']=='complete':continue
                owned.observe(s['name']);code=s['process'].poll()
                if code is None:continue
                need(code==0,'STAGE_FAILED_'+s['name']);owned.stop(s['name'])
                group=base/'groups'/('prefix-'+str(prefix))
                if s['stage']=='features':
                    need((group/'features/features-complete.json').is_file(),'FEATURE_COMPLETION_RECEIPT_MISSING')
                    name='train-prefix'+str(prefix);p=start_owned(owned,interrupt,name,torchrun(c,prefix,'train_ddp',max_tokens,base),s['env'],base/'logs'/(name+'.log'),base/'repo')
                    s.update(stage='train',name=name,process=p)
                else:
                    need((group/'train/RESULT.json').is_file(),'TRAIN_COMPLETION_RESULT_MISSING')
                    s['stage']='complete'
            time.sleep(.25)
        success=True
    finally:
        try:
            owned.stop_all();owned.reap();births=[r for j in owned.jobs.values() for r in j['ledger'].values()];deadline=time.monotonic()+60
            while True:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=30).split()
                present=birth_states(births)
                live=[r for r in present if r['state']!='Z'];zombies=[r for r in present if r['state']=='Z'];contexts=[]
                for pid in set(apps)&{str(r['pid']) for r in births}:
                    try:current=identity(int(pid))
                    except (FileNotFoundError,ProcessLookupError):contexts.append(pid);continue
                    if any(str(r['pid'])==pid and all(current[k]==r[k] for k in ('start_ticks','sid','uid')) for r in births):contexts.append(pid)
                if not live and not contexts:break
                need(time.monotonic()<deadline,'OWNED_GPU_OR_PROCESS_RELEASE_PENDING');time.sleep(.5)
            once(base/'logs/cleanup-status.json',dict(role=ROLE,run_id=c['run_id'],cleaned=True,owned_active_processes_released=True,owned_births_released=not zombies,
                zombie_owned_births_remaining=[r['pid'] for r in zombies],owned_gpu_contexts_released=True,checked_unix=time.time()))
        finally:
            lease.close()
            for s,h in previous.items():signal.signal(s,h)
    if success:once(base/'logs/pipeline-complete.json',dict(role=ROLE,run_id=c['run_id'],prefix_lengths=list(PREFIXES),epochs=20,
        intended_steps_per_rank=2000,completed_group_processes=True,native_actual_exit0_verified=False,native_cgroup_release_verified=False,post_exit_observer_required=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('submit','run','query'));p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    result={'submit':submit,'query':query,'run':run}[a.phase](a.base)
    if result is not None:print(json.dumps(result))

if __name__=='__main__':main()

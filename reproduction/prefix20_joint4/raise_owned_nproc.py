"""Once-only soft RLIMIT_NPROC promotion for exact job727251 step_batch.

No signals, forks of workers, Ray restart, GPU calls, training/code changes,
hard-limit changes or global limit changes. Run as jxu957 inside existing job.
"""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time

BASE='/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727148-20261016'
JOB='727251'
RUN='a79333e9574742a38f922974e30203e9'
COMMIT='78074b669a0d47648ad7c5b2a687159af00cd090'
UID=14090
NODE='ACD1-25'
TARGET=8192
PROTECTED=('logs/run-manifest.json','logs/native-allocation-admission.json',
           'preparation/training-submission.json','preparation/run-source.json')
KEYS=('pid','start_ticks','sid','uid','uids','boot_id','hostname','cgroup','proc_device','proc_inode')


def need(value,reason):
    if not value:raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe(path):
    path=Path(path)
    need(path.is_absolute() and path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents)),
         'NPROC_UNSAFE_PATH')
    return path


def document(path):return json.loads(safe(path).read_text())


def publish(path,value):
    path=Path(path);safe(path.parent);temp=path.with_name(path.name+'.partial-'+str(time.time_ns()))
    try:
        with temp.open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temp.exists():temp.unlink()


def validate_plan(plan):
    fields={'schema_version','base','job','run_id','training_commit','target_soft_nproc',
            'node','boot_id','operator_sha256','operator_publication_commit','protected_sha256','include_current_cpu_helper'}
    need(isinstance(plan,dict) and set(plan)==fields and plan['schema_version']==1,'NPROC_PLAN_SCHEMA')
    need((plan['base'],plan['job'],plan['run_id'],plan['training_commit'],plan['node'],plan['target_soft_nproc'])==
         (BASE,JOB,RUN,COMMIT,NODE,TARGET),'NPROC_EXACT_ACTIVE_RUN_REQUIRED')
    need(plan['include_current_cpu_helper'] is True,'NPROC_EXPLICIT_CURRENT_CPU_HELPER_PROMOTION_REQUIRED')
    need(isinstance(plan['boot_id'],str) and re.fullmatch('[0-9a-f-]{36}',plan['boot_id']),'NPROC_BOOT_ID_REQUIRED')
    need(re.fullmatch('[0-9a-f]{64}',plan['operator_sha256']) is not None and
         re.fullmatch('[0-9a-f]{40}',plan['operator_publication_commit']) is not None,'NPROC_PUBLISHED_OPERATOR_REQUIRED')
    need(set(plan['protected_sha256'])==set(PROTECTED) and all(re.fullmatch('[0-9a-f]{64}',h) for h in plan['protected_sha256'].values()),
         'NPROC_PROTECTED_PUBLIC_RECEIPT_SHA_REQUIRED')
    return plan


def in_scope(cgroup,helper=False):
    entries={}
    for line in cgroup.splitlines():
        row=line.split(':',2)
        if len(row)!=3:return False
        for controller in row[1].split(','):entries[controller]=row[2]
    scope=r'/slurm/uid_14090/job_727251/step_[0-9]+' if helper else re.escape('/slurm/uid_14090/job_727251/step_batch')
    for name in ('cpuset','freezer','memory','devices'):
        value=entries.get(name,'')
        if not re.fullmatch(scope+r'(?:/task_[0-9]+)?',value):return False
    roots={re.sub(r'/task_[0-9]+$','',value) for value in entries.values() if value.startswith('/slurm/')}
    if len(roots)!=1:return False
    # Any additional Slurm controller must identify this exact uid/job/step too.
    return all(re.fullmatch(scope+r'(?:/task_[0-9]+)?',value)
               for value in entries.values() if value.startswith('/slurm/'))


def in_batch(cgroup):return in_scope(cgroup)


def read_identity(pid,proc=Path('/proc')):
    root=proc/str(pid)
    raw=(root/'stat').read_text();fields=raw.rsplit(')',1)[1].split()
    status=dict(line.split(':',1) for line in (root/'status').read_text().splitlines() if ':' in line)
    return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),state=fields[0],
        uid=root.stat().st_uid,proc_device=root.stat().st_dev,proc_inode=root.stat().st_ino,
        uids=[int(x) for x in status['Uid'].split()],threads=int(status['Threads']),
        name=status.get('Name','').strip(),cgroup=(root/'cgroup').read_text(),hostname=socket.gethostname().split('.')[0],
        boot_id=(proc/'sys/kernel/random/boot_id').read_text().strip())


def owned(value,plan,helper=False):
    return (value['state']!='Z' and value['uid']==UID and value['uids']==[UID]*4 and
            value['hostname']==NODE and value['boot_id']==plan['boot_id'] and in_scope(value['cgroup'],helper) and
            (not helper or value['pid']==os.getpid()))


def same(expected,actual,plan,helper=False):
    need(owned(actual,plan,helper) and all(expected[k]==actual[k] for k in KEYS),'NPROC_TARGET_BIRTH_OR_SCOPE_CHANGED')


def snapshot(plan,reader=read_identity,proc=Path('/proc')):
    rows=[]
    for path in proc.iterdir():
        if not path.name.isdecimal():continue
        try:value=reader(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if owned(value,plan):rows.append(value)
    need(rows,'NPROC_NO_OWN_ACTIVE_BATCH_PROCESSES')
    return sorted(rows,key=lambda row:row['pid'])


def limits_ok(pair):
    soft,hard=pair
    need(type(soft) is int and type(hard) is int and (soft>=0 or soft==resource.RLIM_INFINITY) and
         (hard>=0 or hard==resource.RLIM_INFINITY),'NPROC_INVALID_KERNEL_LIMITS')
    need(hard==resource.RLIM_INFINITY or hard>=TARGET,'NPROC_HARD_LIMIT_BELOW_REQUEST_NO_CHANGES')
    need(hard==resource.RLIM_INFINITY or soft!=resource.RLIM_INFINITY and soft<=hard,'NPROC_INVALID_SOFT_HARD_ORDER')


def kernel_prlimit(pid,kind,new=None):
    need(hasattr(resource,'prlimit'),'NPROC_LINUX_RESOURCE_PRLIMIT_REQUIRED')
    return resource.prlimit(pid,kind) if new is None else resource.prlimit(pid,kind,new)


def promote(record,plan,*,reader=read_identity,prlimit=kernel_prlimit,journal=lambda value:None,helper=False):
    """Revalidate birth/scope around each actual kernel read/set/readback."""
    try:
        same(record,reader(record['pid']),plan,helper)
        before=tuple(prlimit(record['pid'],resource.RLIMIT_NPROC));limits_ok(before)
        same(record,reader(record['pid']),plan,helper)
    except (FileNotFoundError,ProcessLookupError):
        value=dict(process=record,status='EXITED_BEFORE_ANY_WRITE',changed=False);journal(value);return value
    soft,hard=before
    if soft==resource.RLIM_INFINITY or soft>=TARGET:
        after=tuple(prlimit(record['pid'],resource.RLIMIT_NPROC));same(record,reader(record['pid']),plan,helper)
        need(after==before,'NPROC_LIMIT_CHANGED_CONCURRENTLY_INSPECT')
        value=dict(process=record,status='ALREADY_SUFFICIENT_UNCHANGED',changed=False,before=list(before),after=list(after))
        journal(value);return value
    intent=dict(process=record,status='PROMOTION_INTENT',before=list(before),requested_after=[TARGET,hard],changed=None)
    journal(intent)  # Must durably exist before the actual prlimit syscall.
    try:
        # Shared-FS journal fsync can be slow. Validate again only after it is
        # durable, immediately around the actual syscall, never on old birth.
        current=tuple(prlimit(record['pid'],resource.RLIMIT_NPROC));limits_ok(current)
        same(record,reader(record['pid']),plan,helper)
        if current[0]==resource.RLIM_INFINITY or current[0]>=TARGET:
            value=dict(process=record,status='BECAME_SUFFICIENT_UNCHANGED',changed=False,before=list(current),after=list(current))
            journal(value);return value
        need(current==before,'NPROC_LIMIT_CHANGED_DURING_JOURNAL_NO_WRITE')
        previous=tuple(prlimit(record['pid'],resource.RLIMIT_NPROC,(TARGET,hard)))
    except ProcessLookupError:
        value=dict(process=record,status='EXITED_AT_KERNEL_SET',changed=False,before=list(before));journal(value);return value
    except BaseException as error:
        journal(dict(process=record,status='KERNEL_SET_FAILED',before=list(before),changed=None,
                     error_type=type(error).__name__,reason=str(error)));raise
    # A mismatch is not safe to retry. Capture the actual atomic old values and
    # post-state even when verification fails, to preserve partial outcome.
    value=dict(process=record,status='KERNEL_SET_RETURNED_VERIFYING',changed=True,
               before=list(before),kernel_returned_old=list(previous),requested_after=[TARGET,hard])
    journal(value)
    try:
        same(record,reader(record['pid']),plan,helper)
        after=tuple(prlimit(record['pid'],resource.RLIMIT_NPROC));value['after']=list(after)
        same(record,reader(record['pid']),plan,helper)
        need(previous==before,'NPROC_CONCURRENT_LIMIT_WRITE_DETECTED_INSPECT_NO_RETRY')
        need(after==(TARGET,hard),'NPROC_ACTUAL_KERNEL_READBACK_DIFFERS')
    except BaseException as error:
        value.update(status='POST_WRITE_UNVERIFIED_INSPECT_NO_RETRY',error_type=type(error).__name__,reason=str(error))
        journal(value);raise
    value['status']='PROMOTED_SOFT_ONLY_VERIFIED';journal(value);return value


def allocated_indices(detail):
    line=next((line for line in detail.splitlines() if 'Nodes='+NODE+' ' in line),'')
    match=re.search(r'GRES=gpu:(?:[A-Za-z0-9_-]+:)?4\(IDX:([^)]*)\)',line)
    need(match is not None,'NPROC_CURRENT_ACTUAL_FOUR_GPU_ALLOCATION_REQUIRED')
    values=[]
    for part in match.group(1).split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));values.extend(range(lo,hi+1))
        else:values.append(int(part))
    need(len(values)==4 and len(set(values))==4,'NPROC_ACTUAL_GPU_INDICES_DIFFER')
    return values


def binding(plan,runner=subprocess.run):
    validate_plan(plan)
    need(os.getuid()==UID and socket.gethostname().split('.')[0]==NODE and
         Path('/proc/sys/kernel/random/boot_id').read_text().strip()==plan['boot_id'] and
         os.environ.get('SLURM_JOB_ID')==JOB,'NPROC_ACTUAL_JOB_NODE_UID_REQUIRED')
    need('torch' not in sys.modules and 'ray' not in sys.modules,'NPROC_OPERATOR_MUST_REMAIN_CPU_ONLY')
    need(sha(__file__)==plan['operator_sha256'],'NPROC_OPERATOR_SOURCE_SHA_CHANGED')
    source=safe(__file__)
    need(source.is_relative_to(Path(BASE)/'preparation/nproc-rescue-code'/plan['operator_publication_commit']),
         'NPROC_OPERATOR_MUST_USE_PUBLISHED_SEPARATE_CODE')
    for name,digest in plan['protected_sha256'].items():need(sha(safe(Path(BASE)/name))==digest,'NPROC_PROTECTED_RECEIPT_CHANGED: '+name)
    need((Path(BASE)/'job.id').read_text().strip()==JOB,'NPROC_JOB_ID_CHANGED')
    run=document(Path(BASE)/'logs/run-manifest.json');allocation=document(Path(BASE)/'logs/native-allocation-admission.json')
    need(run['run_id']==RUN and run['source_commit']==COMMIT and run['training_world_size']==4 and
         run['execution_site']=='hpc3' and run['data_selection_mode']=='prefix20_probe','NPROC_RUN_MANIFEST_DIFFERS')
    need(allocation['accepted'] is True and allocation['run_id']==RUN and allocation['slurm_job_id']==JOB and
         allocation['node']==NODE and allocation['training_world_size']==4 and
         len(set(allocation['actual_cuda_runtime_uuids']))==4,'NPROC_NATIVE4_ACCEPTANCE_DIFFERS')
    r=runner(['scontrol','show','job','-d',JOB],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=25)
    need(r.returncode==0 and not r.stderr.strip(),'NPROC_SLURM_ACTUAL_OBSERVATION_FAILED')
    fields=dict(re.findall(r'(\w+)=(\S+)',r.stdout))
    expected=dict(JobId=JOB,JobState='RUNNING',UserId='jxu957(14090)',Account='jxu957',QOS='formal-user',
        Partition='emergency_acd',NodeList=NODE,NumNodes='1',NumCPUs='48',MinMemoryNode='512G',
        Command=BASE+'/source/'+COMMIT+'/reproduction/prefix20_joint4/train.sh',WorkDir=BASE+'/repo')
    need(all(fields.get(k)==v for k,v in expected.items()) and fields.get('TresPerNode') in ('gres/gpu:4','gres:gpu:4'),
         'NPROC_ACTUAL_JOB_RESOURCES_OR_COMMAND_DIFFERS')
    need(allocated_indices(r.stdout)==allocation['actual_allocated_indices'],'NPROC_CURRENT_NATIVE4_INDICES_CHANGED')
    # No mutable pipeline status/ledger is overwritten or relabeled.
    return dict(raw_scontrol=r.stdout,checked_unix=time.time(),protected_sha256=plan['protected_sha256'],
                operator_sha256=plan['operator_sha256'],run_id=RUN,job=JOB,actual_four_gpu_binding_verified=True)


def execute(plan,*,root=None,check=binding,collect=snapshot,reader=read_identity,prlimit=kernel_prlimit,helper_reader=read_identity):
    validate_plan(plan);root=Path(BASE)/'preparation/nproc-rescue727251' if root is None else Path(root)
    root.mkdir(mode=0o770,exist_ok=True);safe(root)
    with (root/'operator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'claim.json').exists():
            return dict(status='ALREADY_ATTEMPTED_NO_RETRY',result=document(root/'result.json') if (root/'result.json').exists() else None)
        before=check(plan);initial=collect(plan);helper=helper_reader(os.getpid())
        same(helper,helper_reader(os.getpid()),plan,helper=True)
        limits_ok(tuple(prlimit(os.getpid(),resource.RLIMIT_NPROC)))
        for row in initial:
            same(row,reader(row['pid']),plan);limits_ok(tuple(prlimit(row['pid'],resource.RLIMIT_NPROC)))
        publish(root/'claim.json',dict(plan=plan,binding=before,operator_birth=helper,initial_processes=initial,created_unix=time.time(),
            effects='raise own current CPU helper and own verified step_batch softNPROC to8192; preserve hard values; no other steps/signals/restarts/code edits'))
        changes={};passes=[];journal_index=0;failure=None;final=None
        def journal(value):
            nonlocal journal_index
            if value.get('changed') is True:
                record=value['process'];changes[(record['pid'],record['start_ticks'])]=dict(value)
            journal_index+=1;publish(root/('kernel-event-%04d.json'%journal_index),dict(value,checked_unix=time.time()))
        try:
            # First allow this exact CPU helper to fork later read-only scontrol
            # checks after Ray resumes and UID thread count passes the old2048.
            promote(helper,plan,reader=helper_reader,prlimit=prlimit,journal=journal,helper=True)
            for pass_index in (1,2):
                guard=check(plan);rows=collect(plan);results=[]
                publish(root/('pass-%d-before.json'%pass_index),dict(binding=guard,processes=rows,checked_unix=time.time()))
                for row in rows:
                    check(plan)  # Live job/source/allocation binding before every possible set.
                    value=promote(row,plan,reader=reader,prlimit=prlimit,journal=journal);results.append(value)
                after=collect(plan)
                value=dict(pass_index=pass_index,results=results,processes_after=after,checked_unix=time.time())
                publish(root/('pass-%d-after.json'%pass_index),value);passes.append(value)
            check(plan);final=[]
            for row in collect(plan):
                same(row,reader(row['pid']),plan);pair=tuple(prlimit(row['pid'],resource.RLIMIT_NPROC));limits_ok(pair)
                need(pair[0]==resource.RLIM_INFINITY or pair[0]>=TARGET,'NPROC_NEW_OR_SURVIVING_CHILD_STILL_BELOW8192')
                same(row,reader(row['pid']),plan);final.append(dict(process=row,actual_kernel_limits=list(pair)))
        except BaseException as error:
            failure=dict(type=type(error).__name__,reason=str(error))
        result=dict(status='PARTIAL_OR_FAILED_INSPECT_NO_RETRY' if failure else 'SOFT_NPROC_PROMOTION_ACTUALLY_VERIFIED',
            job=JOB,run_id=RUN,training_commit=COMMIT,target_soft_nproc=TARGET,passes_completed=len(passes),
            changed_processes=list(changes.values()),final_actual_kernel_readbacks=final,failure=failure,
            hard_limit_changed=any(v.get('kernel_returned_old',[None,None])[1]!=v.get('requested_after',[None,None])[1]
                                   for v in changes.values()),global_limits_changed=False,signals_sent=0,training_job_changed=False,
            training_source_changed=False,actual_gpu_allocations=0,checked_unix=time.time(),
            semantics='resource-only kernel limit repair; training progress requires separate actual observation')
        publish(root/'result.json',result);return result


def main():
    os.umask(0o007);os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    plan=validate_plan(document(a.plan))
    if not a.apply:
        value=dict(binding=binding(plan),processes=snapshot(plan),self_actual_nproc=list(kernel_prlimit(os.getpid(),resource.RLIMIT_NPROC)),
                   mutation_performed=False,signals_sent=0)
    else:value=execute(plan)
    print(json.dumps(value,allow_nan=False),flush=True)
    if a.apply and value['status']!='SOFT_NPROC_PROMOTION_ACTUALLY_VERIFIED':raise SystemExit(1)


if __name__=='__main__':main()

"""Once-only owned source handoff at a complete joint, before another update."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import uuid


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path)  # Atomic no-clobber publication, including terminal proof.
        directory_fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:
        if temporary.exists():temporary.unlink()


def native(job):
    p=subprocess.run(['scontrol','show','job',str(job),'-o'],capture_output=True,text=True,timeout=15)
    if p.returncode: raise RuntimeError(p.stderr)
    return dict(x.split('=',1) for x in p.stdout.split() if '=' in x)


def births(base, run_id):
    ledger=read(base/'logs/process-ledger.json')
    if ledger['run_id']!=run_id:raise ValueError('Another task ownership ledger')
    present=[]
    for item in ledger['jobs'].values():
        for row in item['ledger'].values():
            path=Path('/proc')/str(row['pid'])
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split()
                if (int(fields[19])==row['start_ticks'] and int(fields[3])==row['sid']
                        and path.stat().st_uid==row['uid']):present.append(dict(row,state=fields[0]))
            except (FileNotFoundError,ProcessLookupError):pass
    return present


def check_native(value, base, run_id, job):
    actual_node=value.get('ReqNodeList') if value.get('JobState')=='PENDING' else value.get('NodeList')
    if (value['JobId']!=str(job) or value['WorkDir']!=str(base)
            or actual_node!='h100-10' or value['Comment']!='g1-prefix200:'+run_id
            or value['TresPerNode']!='gres:gpu:8'):
        raise ValueError('Actual native ownership/allocation differs')


def exact_dependency(value, job):
    # scontrol may append current state to each dependency job, e.g. (unfulfilled).
    import re
    parts = re.split(r'[,?]', value)
    expected = re.escape(str(job))
    return any(re.fullmatch(r'afterany:' + expected + r'(?:\([^)]*\))?', part) for part in parts)


def controller_identity(pid, base, config, job):
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    args=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    expected=[str(Path(config['environment_root'])/'bin/python'),'-m',
              'reproduction.prefix_diversity200.launch','run','--base',str(base)]
    if args!=expected or proc.stat().st_uid!=0:
        raise ValueError('Controller command/owner is not exactly the frozen source')
    env=(proc/'environ').read_bytes().split(b'\0')
    if ('SLURM_JOB_ID='+str(job)).encode() not in env:
        raise ValueError('Controller native job differs')
    cgroup=(proc/'cgroup').read_text()
    if '1:freezer:/slurm_h100-10/uid_0/job_'+str(job)+'/step_batch' not in cgroup.splitlines():
        raise ValueError('Controller is not in the exact source Slurm batch cgroup')
    status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
    if int(status.get('TracerPid','0').strip()):
        raise ValueError('Controller has an unrelated tracer')
    return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=proc.stat().st_uid,
                state=fields[0],args=args,cgroup=cgroup)


def freeze_controller(base, config, job):
    candidates=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if b'reproduction.prefix_diversity200.launch' not in args:continue
            if ('SLURM_JOB_ID='+str(job)).encode() not in (proc/'environ').read_bytes().split(b'\0'):continue
            candidates.append(controller_identity(int(proc.name),base,config,job))
        except (FileNotFoundError,ProcessLookupError):pass
    if len(candidates)!=1:raise ValueError('Require exactly one actual source controller')
    before=candidates[0];fd=os.pidfd_open(before['pid'])
    try:
        opened=controller_identity(before['pid'],base,config,job)
        if any(opened[k]!=before[k] for k in ('pid','start_ticks','sid','uid','args','cgroup')):
            raise ValueError('Controller birth changed while opening pidfd')
        if opened['state'] in ('T','t'):raise ValueError('Controller already stopped by another action')
        signal.pidfd_send_signal(fd,signal.SIGSTOP)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            current=controller_identity(before['pid'],base,config,job)
            if current['start_ticks']!=before['start_ticks']:raise ValueError('Controller birth changed')
            threads=[]
            for task in (Path('/proc')/str(before['pid'])/'task').iterdir():
                try:
                    parts=(task/'stat').read_text().rsplit(')',1)[1].split()
                    status=dict(line.split(':',1) for line in (task/'status').read_text().splitlines() if ':' in line)
                    if int(status.get('TracerPid','0').strip()):raise ValueError('A controller thread has another tracer')
                    threads.append(dict(tid=int(task.name),state=parts[0]))
                except (FileNotFoundError,ProcessLookupError):pass
            if current['state']=='T' and threads and all(row['state']=='T' for row in threads):
                return fd,dict(before=before,frozen=current,threads=threads,observed_unix=time.time())
            time.sleep(.05)
        raise TimeoutError('Whole controller group-stop was not confirmed')
    except BaseException:
        # A failed reversible freeze must leave the original controller runnable.
        try:signal.pidfd_send_signal(fd,signal.SIGCONT)
        except ProcessLookupError:pass
        os.close(fd);raise


def cgroup_members(job):
    root=Path('/sys/fs/cgroup/freezer/slurm_h100-10/uid_0')/('job_'+str(job))
    if not Path('/sys/fs/cgroup/freezer').is_mount():
        raise ValueError('Actual source freezer cgroup mount is unavailable')
    result=[]
    if root.exists():
        for path in root.rglob('tasks'):
            result.extend(int(pid) for pid in path.read_text().split())
    return sorted(set(result))


def stopped_boundary(base, config, n):
    joint=read(base/'latest-joint.json');stage=read(base/'logs/pipeline-status.json')
    safe=((stage['outer']==n and stage['stage']=='OUTER_JOINT_COMPLETE') or
          (stage['outer']==n+1 and stage['stage'] in ('CANDIDATE_POOL_RESERVATION','outer-%06d-prefix'%(n+1))))
    actor=read(base/'actor/latest.json');probe=read(base/'probe/latest.json')
    next_root=base/'outers'/('outer-%06d'%(n+1))
    if (not safe or joint['outer']!=n or joint['run_id']!=config['run_id']
            or actor!=joint['actor'] or probe!=joint['probe']
            or actor['actual_adam_steps_per_rank']!=[20*n]*8
            or probe['actual_adam_steps_per_rank']!=[20*n]*8
            or any((base/'actor/outers').glob('outer-%04d-*'%(n+1)))
            or any(next_root.glob('actor-attempt-*-config.json'))
            or any((next_root/'probe-update').glob('*'))):
        raise ValueError('Frozen source is not the exact complete unoptimized boundary')
    return joint,stage,actor,probe

def run(directory):
    directory=Path(directory);plan=read(directory/'plan.json');arm=read(directory/'arm.json')
    base=Path(plan['source_base']);n=int(plan['target_complete_outer']);job=str(plan['source_job'])
    if (base.resolve()!=base or base.parent!=Path('/mnt/data/shaoshitong')
            or directory!=base/'preparation/calibration-handoff-v1' or not 1<=n<50):
        raise ValueError('Only the exact pre50 own protocol handoff is supported')
    if sha(base/'preparation/launch-config.json')!=plan['source_config_sha256']:
        raise ValueError('Original frozen source config changed')
    c=read(base/'preparation/launch-config.json')
    if c['run_id']!=plan['run_id'] or c['source_commit']!=plan['source_commit']:
        raise ValueError('Original frozen source identity changed')
    if arm['plan_sha256']!=sha(directory/'plan.json'):
        raise ValueError('Arm request does not bind this exact plan')
    successor=str(arm['successor_job']);target=Path(plan['successor_base'])
    result=directory/'terminal.json';deadline=time.monotonic()+3600
    if result.exists():raise FileExistsError('Terminal proof already exists; never replay a handoff')
    while time.monotonic()<deadline:
        state=native(job);check_native(state,base,c['run_id'],job)
        if state['JobState']!='RUNNING':raise RuntimeError('Source is no longer RUNNING before planned action')
        joint=read(base/'latest-joint.json') if (base/'latest-joint.json').exists() else None
        if not joint or joint['outer']<n:
            time.sleep(2);continue
        if joint['outer']!=n:raise ValueError('Target already passed; source left running, reconcile successor')
        waiting=native(successor);check_native(waiting,target,c['run_id'],successor)
        if (waiting['JobState']!='PENDING' or not exact_dependency(waiting['Dependency'],job)
                or sha(target/'preparation/launch-config.json')!=arm['successor_config_sha256']):
            raise ValueError('Exact successor must already be queued behind this source')
        fd=None;killed=False
        try:
            fd,frozen=freeze_controller(base,c,job)
            # The sole phase scheduler cannot progress between this check and kill.
            joint,stage,actor,probe=stopped_boundary(base,c,n)
            state=native(job);check_native(state,base,c['run_id'],job)
            if state['JobState']!='RUNNING':raise ValueError('Source changed during reversible freeze')
            before=dict(plan=plan,plan_sha256=sha(directory/'plan.json'),source_native=state,
                successor_native=waiting,stage=stage,completed_outer=n,joint_sha256=sha(base/'latest-joint.json'),
                controller_freeze=frozen,actor_adam_per_rank=actor['actual_adam_steps_per_rank'],
                probe_adam_per_rank=probe['actual_adam_steps_per_rank'],observed_unix=time.time())
            once(directory/'before-stop.json',before)
            once(directory/'cancel-intent.json',dict(job_id=job,source_config_sha256=plan['source_config_sha256'],
                controller=frozen['frozen'],reason='Isolated calibration source handoff; no completed updates reset',
                authority=plan['authority'],created_unix=time.time()))
            signal.pidfd_send_signal(fd,signal.SIGKILL);killed=True
            deadline_kill=time.monotonic()+15
            while time.monotonic()<deadline_kill:
                proc=Path('/proc')/str(frozen['frozen']['pid'])
                if not proc.exists():break
                fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[19])!=frozen['frozen']['start_ticks'] or fields[0]=='Z':break
                time.sleep(.05)
            else:raise TimeoutError('Exact stopped controller has not died; no successor import')
            state=native(job);check_native(state,base,c['run_id'],job)
            if state['JobState'] not in ('CANCELLED','FAILED','COMPLETED'):
                cancel=subprocess.run(['scancel',job],text=True,capture_output=True,timeout=30)
                once(directory/'cancel-result.json',dict(returncode=cancel.returncode,stdout=cancel.stdout,stderr=cancel.stderr))
                if cancel.returncode:
                    state=native(job);check_native(state,base,c['run_id'],job)
                    if state['JobState'] not in ('CANCELLED','FAILED','COMPLETED'):
                        raise RuntimeError('Cancellation outcome needs reconciliation; never blind retry')
            else:once(directory/'cancel-result.json',dict(action='SOURCE_ALREADY_ACTUALLY_TERMINAL',native=state))
        finally:
            if fd is not None:
                if not killed:
                    try:signal.pidfd_send_signal(fd,signal.SIGCONT)
                    except ProcessLookupError:pass
                os.close(fd)

        break
    else:raise TimeoutError('Target boundary was not reached; source unchanged')
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        after=native(job);present=births(base,c['run_id'])
        check_native(after,base,c['run_id'],job)
        members=cgroup_members(job)
        if after['JobState'] in ('CANCELLED','FAILED','COMPLETED') and not present and not members:
            joint=read(base/'latest-joint.json')
            if joint['outer']!=n or read(base/'actor/latest.json')!=joint['actor'] or read(base/'probe/latest.json')!=joint['probe']:
                raise ValueError('Boundary advanced during handoff; do not import an older state')
            if any((base/'actor/outers').glob('outer-%04d-*'%(n+1))):
                raise ValueError('An optimization attempt started; preserve and reconcile its real partial writes')
            queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%j'],text=True,timeout=15)
            if queue.strip():raise ValueError('Source still present in queue')
            value=dict(status='PLANNED_CALIBRATION_HANDOFF_COMPLETE',run_id=c['run_id'],job_id=job,
                successor_job=successor,source_base=str(base),completed_outer=n,
                plan_sha256=sha(directory/'plan.json'),source_config_sha256=plan['source_config_sha256'],
                native=after,owned_births_present=[],native_job_cgroup_members=[],controller_dead=True,queue=queue,joint_sha256=sha(base/'latest-joint.json'),
                discarded_actor_or_probe_updates=0,prior_completed_outers_reexecuted=0,
                uncommitted_prefix_directory_preserved=str(base/'outers'/('outer-%06d'%(n+1))/'prefix'),
                measured_unix=time.time(),native_executor_exact_resume_claimed=False)
            once(result,value);print(json.dumps(value),flush=True);return
        time.sleep(2)
    raise TimeoutError('Source terminal/owned cleanup still unproven; successor must not import')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();run(a.directory)

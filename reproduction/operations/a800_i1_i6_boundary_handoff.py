"""Finish one already-running ordinary actor at its exact saved boundary.

Explicit execution only: freeze the authenticated polling parent, let its
original actor child finish, request native parent cleanup, then use that
immutable source's CPU commit function. No model/tensor load or GPU lease.
An independent watchdog resumes the parent after worker failure/deadline.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback

PREFIX='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'
MAX_JSON=64*1024**2


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve()!=path or path.stat().st_size>MAX_JSON:
        raise ValueError('Require bounded canonical regular metadata: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def once(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def identity(value):
    return {k:value[k] for k in ('pid','sid','uid')}|dict(start_ticks=value.get('start_ticks',value.get('birth')))


def process(pid):
    try:
        path=Path('/proc')/str(pid);raw=(path/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,sid=int(fields[3]),uid=path.stat().st_uid,start_ticks=int(fields[19]),
                    state=fields[0],wait_status=int(fields[49]) if len(fields)>49 else None)
    except (FileNotFoundError,ProcessLookupError):return None


def same(who):
    actual=process(who['pid'])
    return actual if actual and identity(actual)==identity(who) else None


def send(who,sig):
    actual=same(who)
    if actual is None:raise ProcessLookupError('Authenticated birth is gone')
    descriptor=os.pidfd_open(who['pid'])
    try:
        if same(who) is None:raise ProcessLookupError('Birth changed before pidfd signal')
        signal.pidfd_send_signal(descriptor,sig)
    finally:os.close(descriptor)


def owned_present(base):
    ledger=read(base/'logs/process-ledger.json');found=[]
    for job in ledger.get('jobs',{}).values():
        for row in [job.get('leader'),*job.get('ledger',{}).values()]:
            if isinstance(row,dict) and all(k in row for k in ('pid','sid','uid')) and ('birth'in row or 'start_ticks'in row):
                actual=same(identity(row))
                if actual:found.append(actual)
    return list({(x['pid'],x['start_ticks']):x for x in found}.values())


def watchdog(spec,output,owner,deadline):
    """Independent CPU fail-safe; it can only CONT the one authenticated parent."""
    who=spec['pipeline_identity']
    while not (output/'watchdog-disarm.json').exists():
        owner_state=same(owner);owner_gone=owner_state is None or owner_state['state'] in ('Z','X')
        if owner_gone or time.time()>deadline:
            actual=same(who);resumed=False
            if actual and actual['state'] in ('T','t'):
                send(who,signal.SIGCONT);resumed=True
            once(output/'watchdog-result.json',dict(parent_resumed=resumed,owner_gone=owner_gone,
                deadline_expired=time.time()>deadline,checked_unix=time.time()))
            return
        time.sleep(2)


def guard(spec):
    base=Path(spec['base']);outer=spec['outer'];who=identity(spec['pipeline_identity']);child=identity(spec['actor_identity'])
    if (str(base)!=PREFIX+spec['run_id'] or base.resolve()!=base
            or not re.fullmatch('[0-9a-f]{32}',spec['run_id'])
            or not re.fullmatch('[0-9a-f]{32}',spec['execution_id'])
            or not re.fullmatch('[0-9a-f]{40}',spec['source_commit'])
            or type(outer)is not int or not 1<=outer<200 or outer%10==0 or outer in (5,50,100,150)
            or spec.get('authorization')!='complete_current_ordinary_actor_then_native_cleanup_and_cpu_joint_commit'):
        raise ValueError('Only the explicitly authorized ordinary-actor boundary is supported')
    h=read(base/'preparation/deployment.json');c=read(base/'preparation/launch-config.json')
    human=read(spec['human_intent_path'])
    if (sha(spec['human_intent_path'])!=spec['human_intent_sha256'] or
            human.get('role')!='latest_direct_human_variable_candidate_schedule_authorization' or
            human.get('current_run_id')!=spec['run_id'] or human.get('current_execution_id')!=spec['execution_id']):
        raise ValueError('Bind the exact latest human candidate-schedule authorization')
    for key in ('base','run_id','execution_id','source_commit'):
        if h[key]!=spec[key] or c[key]!=spec[key]:raise ValueError('Actual deployment/config identity differs')
    if sha(base/'repo/reproduction/prefix_selection200/launch.py')!=h['source_files']['reproduction/prefix_selection200/launch.py']:
        raise ValueError('Original native commit source changed')
    submit=read(base/'preparation/submit-receipt.json')
    if identity(submit['leader'])!=who or who['uid']!=os.getuid() or who['sid']!=who['pid']:
        raise ValueError('Owned pipeline birth/SID differs')
    if child['uid']!=os.getuid() or child['sid']!=child['pid'] or same(who)is None or same(child)is None:
        raise ValueError('Original ordinary actor child is not present at its actual birth')
    stage=read(base/'logs/pipeline-status.json');ledger=read(base/'logs/process-ledger.json')
    name=spec['actor_job_name']
    if (not re.fullmatch('outer-%06d-actor-[0-9a-f]{32}'%outer,name)
            or stage.get('run_id')!=spec['run_id'] or stage.get('outer')!=outer or stage.get('stage')!=name
            or ledger.get('run_id')!=spec['run_id'] or identity(ledger['jobs'][name]['leader'])!=child):
        raise ValueError('Parent has advanced beyond the exact ordinary actor phase')
    root=base/'outers'/('outer-%06d'%outer);pool=read(base/'question-pool/state.json')
    if any((root/name).exists() for name in ('complete.json','pool-state-at-completion.json','probe-state-at-completion.pt')):
        raise ValueError('Parent has begun/finished its joint commit; reconcile without external duplicate writes')
    if pool['next_outer']!=outer+1 or pool['consumed_candidates']!=c['candidate_questions']*outer:
        raise ValueError('An unexpected later reservation already exists')
    previous=read(base/'latest-joint.json') if outer>1 else None
    if previous and previous['outer']!=outer-1:raise ValueError('Prior full joint boundary differs')
    return base,c,root


COMMIT=r'''
import json,os,sys,pathlib
sys.dont_write_bytecode=True
p=json.load(sys.stdin);base=pathlib.Path(p['base'])
from reproduction.prefix_selection200.launch import commit_outer
from reproduction.prefix20_online.launch import read,sha,once
from reproduction.prefix_selection200.periodic import is_probe_update,probe_boundary
from reproduction.prefix_selection200.storage import actual_actor_boundary
c=read(base/'preparation/launch-config.json');n=p['outer'];root=base/'outers'/('outer-%06d'%n)
assert not is_probe_update(n)
a=read(base/'actor/latest.json');assert a['completed_outer']==n and actual_actor_boundary(a,n)
q=read(base/'probe/latest.json');b=probe_boundary(n)
assert q['completed_outer']==b['last_update_outer'] and q['actual_adam_steps_per_rank']==[b['actual_adam_per_rank']]*4
f=dict(status='PROBE_FROZEN_NO_OPTIMIZER_UPDATE',outer=n,new_adam_steps_per_rank=0,
 actual_adam_steps_per_rank=q['actual_adam_steps_per_rank'],last_update_outer=b['last_update_outer'])
if (root/'probe-frozen.json').exists():assert read(root/'probe-frozen.json')==f
else:once(root/'probe-frozen.json',f)
t=read(base/'logs/phase-timings.json') if (base/'logs/phase-timings.json').exists() else {}
# Parent did not record phase return; preserve the separately observed actual
# zombie exit0 rather than fabricating a parent-recorded timing or job result.
complete=commit_outer(base,c,n,root,root/'probe-score/selection.json',t,grounding_state=a['grounding_state'])
print(json.dumps(dict(outer=n,actor_adam_per_rank=a['actual_adam_steps_per_rank'],
 probe_adam_per_rank=q['actual_adam_steps_per_rank'],joint_sha256=sha(root/'complete.json'),
 latest_joint_sha256=sha(base/'latest-joint.json'),scientific_boundary_committed_by_external_cpu_helper=True)))
'''


def execute(spec_path,output,timeout):
    spec=read(spec_path);output=Path(output);base=Path(spec['base'])
    if (output.parent!=base/'preparation' or not re.fullmatch('boundary-handoff-[0-9a-f]{32}',output.name)
            or output.exists() or output.resolve()!=output or timeout<60 or timeout>3600):
        raise ValueError('Require a new unique owned preparation namespace and bounded deadline')
    output.mkdir();lock=os.open(base/'preparation/boundary-handoff-control.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=identity(process(os.getpid()));deadline=time.time()+timeout;paused=False;terminating=False;watch=None
    receipt=dict(run_id=spec['run_id'],execution_id=spec['execution_id'],source_commit=spec['source_commit'],
        base=str(base),outer=spec['outer'],spec_sha256=sha(spec_path),worker_identity=owner,started_unix=time.time(),
        gpu_lease_acquired=False,model_or_tensor_read=False)
    once(output/'intent.json',receipt)
    try:
        base,c,root=guard(spec)
        watch=subprocess.Popen([sys.executable,'-m','reproduction.operations.a800_i1_i6_boundary_handoff',
            '--watchdog','--spec',str(spec_path),
            '--output',str(output),'--owner',json.dumps(owner),'--deadline',str(deadline)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        who=spec['pipeline_identity'];send(who,signal.SIGSTOP);paused=True
        for _ in range(50):
            actual=same(who)
            if actual and actual['state'] in ('T','t'):break
            time.sleep(.02)
        else:raise RuntimeError('Parent did not enter authenticated stopped state')
        # Recheck phase and reservation after STOP closes the admission race.
        guard(spec);once(output/'parent-paused.json',dict(identity=identity(who),actual_state=same(who),paused_unix=time.time()))
        while True:
            actual=same(spec['actor_identity'])
            if actual is None:raise RuntimeError('Actor child disappeared without authenticated zombie exit status')
            if actual['state']=='Z':
                if actual['wait_status']!=0:raise RuntimeError('Original actor naturally exited nonzero: '+str(actual['wait_status']))
                break
            if time.time()>deadline:raise TimeoutError('Actor natural-completion deadline exceeded')
            time.sleep(5)
        # CPU scalar validation from frozen original source, no checkpoint load.
        validation=r"import json,sys,pathlib;sys.dont_write_bytecode=True;from reproduction.prefix_selection200.storage import actual_actor_boundary;b=pathlib.Path(sys.argv[1]);n=int(sys.argv[2]);a=json.loads((b/'actor/latest.json').read_text());assert a['completed_outer']==n and actual_actor_boundary(a,n);print(json.dumps({k:a[k] for k in ('completed_outer','actual_adam_steps_per_rank','new_adam_steps_per_rank','opportunities_processed','skipped_update_opportunities','scheduler_completed_outer')}))"
        env=dict(os.environ,PYTHONPATH=str(base/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        python=str(Path(c['environment_root'])/'bin/python')
        done=subprocess.run([python,'-c',validation,str(base),str(spec['outer'])],cwd=base/'repo',env=env,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
        actor=json.loads(done.stdout);once(output/'actor-natural-complete.json',dict(actor_process=actual,
            native_actor_metadata=actor,actual_zombie_exit_status=0,parent_job_exit_metadata_not_fabricated=True,
            checked_unix=time.time()))
        # TERM is pending while stopped; CONT lets the original deferred-check
        # unwind into its own all-owned cleanup, before any joint CPU writes.
        send(who,signal.SIGTERM);terminating=True;send(who,signal.SIGCONT);paused=False
        while same(who)is not None or owned_present(base):
            if time.time()>deadline:raise TimeoutError('Native cleanup did not finish before deadline')
            time.sleep(2)
        cleanup=read(base/'logs/cleanup-status.json')
        if cleanup.get('cleaned')is not True or cleanup.get('owned_births_all_gone')is not True:
            raise ValueError('Native cleanup receipt does not prove every original owned writer gone')
        pool=read(base/'question-pool/state.json')
        if pool['next_outer']!=spec['outer']+1:raise ValueError('Parent reserved a new outer; do not commit')
        once(output/'native-terminal.json',dict(pipeline_identity=identity(who),pipeline_gone=True,
            remaining_owned_births=[],cleanup=cleanup,checked_unix=time.time(),manual_switch_not_failure=True))
        done=subprocess.run([python,'-c',COMMIT],input=json.dumps(dict(base=str(base),outer=spec['outer'])).encode(),
            cwd=base/'repo',env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,check=True)
        result=json.loads(done.stdout)
        ledger=read(base/'logs/process-ledger.json');births={}
        for job in ledger['jobs'].values():
            for row in [job.get('leader'),*job.get('ledger',{}).values()]:
                if isinstance(row,dict) and all(k in row for k in ('pid','sid','uid')) and ('birth' in row or 'start_ticks' in row):
                    value=identity(row);births[(value['pid'],value['start_ticks'])]=value
        if any(job.get('state')!='STOPPED' for job in ledger['jobs'].values()):
            raise ValueError('Original native ledger has an unreconciled owned job')
        terminal=dict(run_id=spec['run_id'],execution_id=spec['execution_id'],source_commit=spec['source_commit'],base=str(base),
            host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            termination_cause='AUTHORIZED_BOUNDARY_HANDOFF',human_intent_sha256=spec['human_intent_sha256'],
            stopped_after_complete_outer=spec['outer'],pipeline_alive=False,owned_writers_gone=True,actual_births_present=[],
            owned_births=list(births.values()),process_ledger_sha256=sha(base/'logs/process-ledger.json'),
            source_joint_sha256=result['latest_joint_sha256'],actor_natural_exit_status=0,
            boundary_cpu_finalization_disclosed=True,uncommitted_after_joint=dict(actor_actual_writes_per_rank=[0]*4,
                probe_actual_writes_per_rank=[0]*4,optimizer_work_in_flight=False,all_updates_accounted=True),checked_unix=time.time())
        once(output/'terminal-proof.json',terminal)
        receipt.update(status='NATIVE_ORDINARY_ACTOR_COMPLETED_PARENT_CLEANED_CPU_JOINT_COMMITTED',
            actor_boundary=actor,joint=result,original_actor_rerun=False,new_probe_adam=0,
            parent_native_cleanup_verified=True,current_outer_actor_updates_lost=0,
            terminal_proof=str(output/'terminal-proof.json'),terminal_proof_sha256=sha(output/'terminal-proof.json'),finished_unix=time.time())
        once(output/'complete.json',receipt);return receipt
    except BaseException as error:
        receipt.update(status='BOUNDARY_HANDOFF_FAILED_RECONCILE_REQUIRED',error_class=type(error).__name__,
            error=str(error),parent_termination_requested=terminating,checked_unix=time.time())
        once(output/'failure.json',receipt);raise
    finally:
        if paused:
            try:send(spec['pipeline_identity'],signal.SIGCONT)
            except ProcessLookupError:pass
        once(output/'watchdog-disarm.json',dict(parent_continued_or_gone=True,checked_unix=time.time()))
        os.close(lock)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout-seconds',type=int,default=1800)
    p.add_argument('--execute',action='store_true');p.add_argument('--watchdog',action='store_true')
    p.add_argument('--owner');p.add_argument('--deadline',type=float);args=p.parse_args()
    if args.watchdog:watchdog(read(args.spec),args.output,json.loads(args.owner),args.deadline)
    elif args.execute:print(json.dumps(execute(args.spec,args.output,args.timeout_seconds)))
    else:
        base,c,root=guard(read(args.spec));print(json.dumps(dict(status='EXACT_ORDINARY_ACTOR_METADATA_ADMISSION_ONLY',
            base=str(base),outer=read(args.spec)['outer'],remote_signal_sent=False)))


if __name__=='__main__':main()

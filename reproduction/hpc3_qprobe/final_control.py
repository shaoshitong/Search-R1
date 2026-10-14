"""Owner-only once control for CPU pair archives and a final native4 audit.

CPU supervisors detach from the SSH window, retain their archive child and
record its actual wait status. No command here starts training or sends signals.
An unknown submit/launch result is inspectable, never an invitation to replay.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
OWNER=14090
STEPS=(5,50,100,150,200)
MODULE='reproduction.hpc3_qprobe.final_control'
ENTRY='reproduction/hpc3_qprobe/final_control.sh'
CAPS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
      'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def need(ok,message):
    if not ok:raise ValueError('QPROBE_FINAL_CONTROL: '+message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*2**20):h.update(block)
    return h.hexdigest()


def publish(path,value):
    """Atomic exclusive publication, including parent-directory durability."""
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()
    fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def read(path):
    path=Path(path)
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and
         path.stat().st_uid==OWNER and path.is_file(),'noncanonical/unowned control file')
    return json.loads(path.read_text())


def process(pid):
    from reproduction.a800.owned_processes import identity
    row=identity(pid)
    return {key:row[key] for key in ('pid','start_ticks','sid','uid','state')}


def birth(row):return {key:row[key] for key in ('pid','start_ticks','sid','uid')}


def live(row):
    try:now=process(row['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    return birth(now)==birth(row) and now['state']!='Z'


def owner():need(os.getuid()==OWNER,'actual jxu957 owner required')


def source(commit):
    owner();need(re.fullmatch('[0-9a-f]{40}',commit) is not None,'exact published commit required')
    repository=ROOT/'source'/commit
    need(Path(__file__).resolve().parents[2]==repository and repository.resolve()==repository,
         'loaded controller differs from immutable published source')
    proof_path=ROOT/'preparation'/('source-'+commit+'-verification.json');proof=read(proof_path)
    need(proof['commit']==commit and proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS','source publication missing')
    from reproduction.common8.source_files import runtime_files
    actual={str(p.relative_to(repository)):p for p in runtime_files(repository)}
    need(set(actual)<=set(proof['files']),'runtime file omitted from publication')
    for name,sha in proof['files'].items():
        p=repository/name
        need(p.resolve()==p and p.is_relative_to(repository) and not p.is_symlink() and digest(p)==sha,
             'published source changed: '+name)
    return repository,dict(commit=commit,publication_path=str(proof_path),publication_sha256=digest(proof_path))


def producer(base,*,complete=False):
    owner();base=Path(base)
    need(base.is_absolute() and base.resolve()==base and base.parent==ROOT.parent and
         base.name.startswith('qprobe-teacher-warmup200-') and base.stat().st_uid==OWNER,
         'canonical owned producer base required')
    manifest=base/'logs/run-manifest.json';run=read(manifest)
    need(re.fullmatch('[0-9a-f]{32}',run['run_id']) is not None and run['training_world_size']==4 and
         run['actor_micro_batch_per_rank']==4 and run.get('execution_site')=='hpc3' and
         run.get('execution_backend','slurm')=='slurm','actual native4 producer required')
    prior=read(base/'preparation/training-submission.json');claim=read(base/'preparation/training-submit-claim.json')
    job=prior['job'];need(re.fullmatch('[1-9][0-9]*',job) is not None,'actual producer job missing')
    entry=ROOT/'source'/run['source_commit']/'reproduction/hpc3_qprobe/train.sh'
    need(prior['returncode']==0 and prior['commit']==claim['commit']==run['source_commit'] and
         run.get('slurm_job_id',job)==job and claim['command'][-3:]==[str(entry),str(base),run['source_commit']],
         'producer submission/source binding differs')
    # The native producer manifest deliberately remains backend-neutral. Its
    # actual Slurm job is recorded by the immutable allocation admission, not
    # an invented/retrofitted field in the live scientific manifest.
    allocation_path=base/'logs/native-allocation-admission.json';allocation=read(allocation_path)
    uuids=allocation.get('actual_cuda_runtime_uuids',[])
    indices=allocation.get('actual_allocated_indices',[])
    need(allocation.get('accepted') is True and allocation.get('run_id')==run['run_id'] and
         allocation.get('slurm_job_id')==job and type(allocation.get('training_world_size')) is int and
         allocation['training_world_size']==4 and allocation.get('requested_gres')=='gres/gpu:4' and
         allocation.get('visible_devices')==['0','1','2','3'] and len(indices)==4 and
         all(type(index) is int and index>=0 for index in indices) and len(set(indices))==4 and
         len(uuids)==4 and len(set(uuids))==4 and all(isinstance(value,str) and
             re.fullmatch('GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',value) is not None
             for value in uuids),'actual native4 producer allocation binding differs')
    from reproduction.common8.native_allocation import fields,allocated_indices
    detail=allocation.get('actual_allocation_detail','');observed=fields(detail)
    need(observed.get('JobId')==job and observed.get('UserId')=='jxu957(14090)' and
         observed.get('Account')=='jxu957' and observed.get('QOS')=='formal-user' and
         observed.get('Partition')=='emergency_acd' and observed.get('WorkDir')==str(base) and
         observed.get('Command')==str(entry) and observed.get('NumNodes')=='1' and
         observed.get('NumCPUs')=='48' and observed.get('MinMemoryNode')=='512G' and
         observed.get('NodeList')==allocation.get('node') and
         'gres/gpu=4' in observed.get('TRES','').split(','),
         'actual native producer detail binding differs')
    need(allocated_indices(detail,allocation['node'],count=4)==indices,
         'actual native producer GPU index detail differs')
    raw=subprocess.check_output(['sacct','-nP','-X','-j',job,
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition'],text=True,timeout=30)
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    need(len(rows)==1 and rows[0][0]==job and rows[0][3:]==[
        'jxu957','14090','jxu957','formal-user','emergency_acd'],'actual producer accounting owner differs')
    if complete:need(rows[0][1:3]==['COMPLETED','0:0'],'producer must actually finish COMPLETED0 before final submission')
    else:need(rows[0][1] in ('RUNNING','COMPLETING','COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY') and
              (rows[0][1]!='COMPLETED' or rows[0][2]=='0:0'),'producer has no observed execution')
    binding=dict(base=str(base),run_id=run['run_id'],producer_commit=run['source_commit'],producer_job=job,
        source_manifest_sha256=digest(manifest),original_submission_sha256=digest(base/'preparation/training-submission.json'),
        original_claim_sha256=digest(base/'preparation/training-submit-claim.json'),
        native_allocation_sha256=digest(allocation_path))
    return run,binding,raw


def pair(base,run,step):
    need(type(step) is int and step in STEPS,'only five fixed real pairs are supported')
    path=Path(base)/'artifacts/question-probe/completed-pairs'/('global_step_'+str(step))/'source-receipt.json'
    value=read(path)
    expected_checkpoint=Path(run['checkpoint_dir'])/'actor'/('global_step_'+str(step))
    if step==5 and (run.get('full_state_restore') or {}).get('kind')=='question_probe_saved_checkpoint':
        expected_checkpoint=Path(run['full_state_restore']['source_checkpoint'])
        need(value.get('checkpoint_origin') is not None,'actual restored source5 boundary missing')
        from search_g1.qprobe_checkpoint import read_source
        read_source(path)
    need(type(value['step']) is int and type(value['world_size']) is int and
         value['schema_version']==1 and value['kind']=='question_probe_pair' and
         value['status']=='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS' and value['step']==step and
         value['world_size']==4 and value['run_id']==run['run_id'] and
         value['actor_snapshot_id']==run['run_id']+':'+str(step) and
         value['source_base']==str(base) and value['checkpoint']==str(expected_checkpoint),
         'actual pair identity/path differs')
    manifest=Path(base)/'logs/run-manifest.json'
    need(value['source_run_manifest']['path']==str(manifest) and
         value['source_run_manifest']['full_read_sha256']==digest(manifest) and
         value['attempt_progress']['completed_outer']==value['driver_state']['completed_outer']==step and
         sorted(row['rank'] for row in value['ranks'])==list(range(4)) and
         all(row['completed_outer']==step for row in value['ranks']),'pair producer/counter/rank binding differs')
    return path,dict(path=str(path),sha256=digest(path),step=step,run_id=run['run_id'])


def control_root(binding,*,create=True):
    root=ROOT/'final-control'/binding['run_id']
    if create:root.mkdir(parents=True,exist_ok=True,mode=0o770)
    else:need(root.is_dir(),'control has not been attempted')
    need(root.resolve()==root and not root.is_symlink() and root.stat().st_uid==OWNER,'control root owner differs')
    p=root/'producer-binding.json'
    if p.exists():need(read(p)==binding,'control already belongs to another producer')
    else:
        need(create,'existing control has no durable producer binding')
        publish(p,binding)
    return root


def caps():
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in CAPS),
         'CPU archive requires an empty CUDA mask and explicit threadcaps1')


def command(repository,mode,claim,token):
    return ['bash',str(repository/ENTRY),str(repository),mode,'--claim',str(claim),'--token',token]


def launch_archive(base,commit,step):
    repository,publication=source(commit);run,binding,accounting=producer(base);p,checkpoint=pair(Path(base),run,step)
    root=control_root(binding);directory=root/('archive-'+str(step))
    need(not directory.exists(),'archive control already attempted; inspect, never replay')
    need(not (ROOT/'archives'/run['run_id']/('global_step_'+str(step))).exists(),
         'archive payload already exists; inspect original archive rather than submitting another worker')
    directory.mkdir(mode=0o770);token=uuid.uuid4().hex;claim_path=directory/'submit-claim.json'
    cmd=command(repository,'archive-worker',claim_path,token)
    claim=dict(schema_version=1,role='detached_cpu_pair_archive',token=token,step=step,producer=binding,
        pair=checkpoint,source=publication,command=cmd,controller=birth(process(os.getpid())),
        actual_accounting=accounting,created_unix=time.time(),training_permitted=False,
        direct_gpu_device_access_permitted=False,native_gpu_audit_submission_permitted=False)
    publish(claim_path,claim)  # Durable exclusive intent before Popen.
    with (directory/'worker.log').open('x') as output:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    actual=process(child.pid)
    need(actual['uid']==OWNER and actual['sid']==actual['pid'],'detached actual CPU worker birth differs')
    publish(directory/'runner-binding.json',dict(token=token,claim_sha256=digest(claim_path),
        actual_process=birth(actual),controller=claim['controller'],created_unix=time.time()))
    value=dict(status='ACTUAL_DETACHED_CPU_ARCHIVE_STARTED',token=token,actual_process=birth(actual),
               claim_path=str(claim_path),claim_sha256=digest(claim_path),source_commit=commit,step=step)
    publish(directory/'launch-receipt.json',value);return value


def worker_binding(claim_path,token,kind,role='detached_cpu_pair_archive'):
    owner();caps();claim_path=Path(claim_path);claim=read(claim_path)
    need(kind in ('runner','task'),'invalid actual CPU process binding kind')
    need(role in ('detached_cpu_pair_archive','detached_cpu_final_audit_submission'),'unknown CPU control role')
    need(claim['schema_version']==1 and claim['role']==role and claim['token']==token and
         claim['training_permitted'] is False and claim['direct_gpu_device_access_permitted'] is False and
         claim['native_gpu_audit_submission_permitted'] is (role=='detached_cpu_final_audit_submission'),
         'CPU control role permission differs')
    _,publication=source(claim['source']['commit']);need(publication==claim['source'],'worker source publication changed')
    from reproduction.hpc3_qprobe.probe_cpu_runtime import verify as verify_cpu_runtime
    verify_cpu_runtime()  # Full dependency reads are detached too, never a55s foreground assumption.
    run,binding,_=producer(claim['producer']['base'],complete=role=='detached_cpu_final_audit_submission')
    component='audit-controller' if role=='detached_cpu_final_audit_submission' else 'archive-'+str(claim['step'])
    if role=='detached_cpu_pair_archive':
        _,checkpoint=pair(Path(binding['base']),run,claim['step']);need(checkpoint==claim['pair'],'CPU pair changed')
    need(binding==claim['producer'] and
         claim_path==ROOT/'final-control'/run['run_id']/component/'submit-claim.json',
         'CPU archive claim source/producer/path changed')
    receipt=claim_path.parent/(kind+'-binding.json');deadline=time.monotonic()+15
    while not receipt.exists():
        need(time.monotonic()<deadline,'CPU launch outcome unknown: runner binding missing; do not replay')
        time.sleep(.05)
    actual=read(receipt)
    need(actual['token']==token and actual['claim_sha256']==digest(claim_path) and
         actual['actual_process']==birth(process(os.getpid())),'only the actually bound CPU process may enter')
    return claim,claim_path.parent


def archive_exec(claim_path,token):
    claim,directory=worker_binding(claim_path,token,'task')
    # Serialize our archive reservations/copies, including du+budget admission.
    lease=ROOT/'final-control/archive-storage.lock'
    need(not lease.is_symlink(),'archive storage lease is a symlink')
    with lease.open('a') as lock:
        need(lease.stat().st_uid==OWNER,'archive storage lease has another owner')
        fcntl.flock(lock,fcntl.LOCK_EX)
        publish(directory/'task-entered.json',dict(token=token,actual_process=birth(process(os.getpid())),
                                                 claim_sha256=digest(claim_path),entered_unix=time.time()))
        from reproduction.hpc3_qprobe.archive_completed_pair import archive
        from reproduction.hpc3_qprobe.final_restore_audit import verify_transport
        result=archive(claim['pair']['path'])
        transport=read(result['path']);verified=verify_transport(result['path'],transport['target_receipt']['path'],
                                                               expected_step=claim['step'])
        publish(directory/'task-result.json',dict(status='CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',
            token=token,claim_sha256=digest(claim_path),actual_process=birth(process(os.getpid())),
            archive=result,transport=verified,training_performed=False,direct_gpu_device_access_performed=False,
            native_gpu_audit_submission_attempted=False,actor_model_loaded=False,completed_unix=time.time()))


def supervise(claim_path,token,role,task_mode,task_status,success_status):
    claim,directory=worker_binding(claim_path,token,'runner',role);repository=ROOT/'source'/claim['source']['commit']
    cmd=command(repository,task_mode,Path(claim_path),token)
    # The detached supervisor retains this child, so its final record contains
    # a real wait status rather than the child's anticipated exit code.
    with (directory/'task.log').open('x') as output:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
    actual=process(child.pid)
    need(actual['uid']==OWNER and actual['sid']==process(os.getpid())['sid'],'CPU archive task birth differs')
    publish(directory/'task-binding.json',dict(token=token,claim_sha256=digest(claim_path),
        actual_process=birth(actual),supervisor=birth(process(os.getpid())),created_unix=time.time()))
    code=child.wait();value=dict(token=token,actual_task_exit_code=code,actual_task_birth=birth(actual),
        actual_supervisor=birth(process(os.getpid())),claim_sha256=digest(claim_path),finished_unix=time.time(),
        status='CPU_CHILD_FAILED_OR_MISSING_RECEIPT',role=role)
    if code==0:
        try:
            result=read(directory/'task-result.json')
            need(result['token']==token and result['claim_sha256']==digest(claim_path) and
                 result['actual_process']==birth(actual) and result['status']==task_status and
                 result['training_performed'] is False and result['direct_gpu_device_access_performed'] is False and
                 result['native_gpu_audit_submission_attempted'] is (role=='detached_cpu_final_audit_submission') and
                 result['actor_model_loaded'] is False,'successful CPU child result has another birth/scope')
            value.update(status=success_status,task_result_sha256=digest(directory/'task-result.json'))
        except Exception as error:value['result_validation_error']=repr(error)
    publish(directory/'worker-result.json',value)
    need(code==0 and value['status']==success_status,
         'actual archive child failed or did not publish a valid result; inspect immutable claim and log')
    return value


def archive_worker(claim_path,token):
    return supervise(claim_path,token,'detached_cpu_pair_archive','archive-exec',
                     'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS','ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0')


def start_audit(base,commit):
    """Short SSH entry; every heavy SHA read and actual submit runs detached."""
    repository,publication=source(commit);_,binding,accounting=producer(base,complete=True)
    root=control_root(binding);directory=root/'audit-controller'
    need(not directory.exists() and not (root/'final-audit').exists(),'audit controller/submission already attempted; inspect, never replay')
    directory.mkdir(mode=0o770);token=uuid.uuid4().hex;claim_path=directory/'submit-claim.json'
    cmd=command(repository,'audit-worker',claim_path,token)
    claim=dict(schema_version=1,role='detached_cpu_final_audit_submission',token=token,producer=binding,
        source=publication,command=cmd,controller=birth(process(os.getpid())),actual_accounting=accounting,
        created_unix=time.time(),training_permitted=False,direct_gpu_device_access_permitted=False,
        native_gpu_audit_submission_permitted=True)
    publish(claim_path,claim)
    with (directory/'worker.log').open('x') as output:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    actual=process(child.pid);need(actual['uid']==OWNER and actual['sid']==actual['pid'],'actual detached submit worker birth differs')
    publish(directory/'runner-binding.json',dict(token=token,claim_sha256=digest(claim_path),actual_process=birth(actual),
                                               controller=claim['controller'],created_unix=time.time()))
    value=dict(status='ACTUAL_DETACHED_FINAL_SUBMIT_CPU_STARTED',token=token,actual_process=birth(actual),
               claim_path=str(claim_path),claim_sha256=digest(claim_path),source_commit=commit)
    publish(directory/'launch-receipt.json',value);return value


def audit_exec(claim_path,token):
    claim,directory=worker_binding(claim_path,token,'task','detached_cpu_final_audit_submission')
    publish(directory/'task-entered.json',dict(token=token,actual_process=birth(process(os.getpid())),
                                             phase='FRESH_FULL_SHA_THEN_ONCE_NATIVE_SUBMIT',entered_unix=time.time()))
    result=submit_audit(claim['producer']['base'],claim['source']['commit'])
    publish(directory/'task-result.json',dict(status='CPU_FINAL_SUBMIT_CONTROLLER_RETURNED',token=token,
        claim_sha256=digest(claim_path),actual_process=birth(process(os.getpid())),submission=result,
        training_performed=False,direct_gpu_device_access_performed=False,
        native_gpu_audit_submission_attempted=True,actor_model_loaded=False,completed_unix=time.time()))
    return result


def audit_worker(claim_path,token):
    return supervise(claim_path,token,'detached_cpu_final_audit_submission','audit-exec',
                     'CPU_FINAL_SUBMIT_CONTROLLER_RETURNED','ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0')


def controller_state(directory):
    claim=read(directory/'submit-claim.json');sha=digest(directory/'submit-claim.json')
    value=dict(status='CPU_LAUNCH_OUTCOME_UNKNOWN_DO_NOT_REPLAY',actual_task_exit_code=None,claim_sha256=sha)
    for kind in ('runner','task'):
        p=directory/(kind+'-binding.json')
        if not p.is_file():continue
        record=read(p);need(record['token']==claim['token'] and record['claim_sha256']==sha,'actual CPU control binding changed')
        row=record['actual_process'];active=live(row);value[kind+'_alive']=active;value['actual_'+kind]=row
        if active:
            value['status']='DETACHED_CPU_CONTROLLER_RUNNING'
            try:
                io=(Path('/proc')/str(row['pid'])/'io').read_text()
                counters={k:int(v.strip()) for k,v in (line.split(':',1) for line in io.splitlines())}
                need(live(row),'CPU process birth changed while reading IO')
                value[kind+'_actual_io']={k:counters[k] for k in ('rchar','wchar','read_bytes','write_bytes')}
            except (FileNotFoundError,ProcessLookupError):pass
    p=directory/'worker-result.json'
    if p.is_file():
        result=read(p);need(result['token']==claim['token'] and result['claim_sha256']==sha and
                           not live(result['actual_task_birth']),'CPU wait receipt has another/live child')
        value.update(status=result['status'],actual_task_exit_code=result['actual_task_exit_code'])
        if result.get('task_result_sha256'):
            p=directory/'task-result.json';need(digest(p)==result['task_result_sha256'],'CPU task result changed')
            task=read(p)
            if 'submission' in task:value['actual_slurm_submission']=task['submission']
    return value


def inspect_archive(base,step,*,full=False):
    need(type(step) is int and step in STEPS,'only fixed archive steps may be inspected')
    run,binding,_=producer(base);root=control_root(binding,create=False);directory=root/('archive-'+str(step))
    claim=read(directory/'submit-claim.json');need(claim['producer']==binding and claim['step']==step,'archive claim differs')
    value=dict(step=step,run_id=run['run_id'],claim_sha256=digest(directory/'submit-claim.json'),
               actual_task_exit_code=None,runner_alive=None,status='LAUNCH_OUTCOME_UNKNOWN_DO_NOT_REPLAY')
    if (directory/'runner-binding.json').is_file():
        runner=read(directory/'runner-binding.json');need(runner['claim_sha256']==value['claim_sha256'],'runner claim differs')
        value.update(actual_runner=runner['actual_process'],runner_alive=live(runner['actual_process']),status='CPU_ARCHIVE_RUNNING' if live(runner['actual_process']) else 'CPU_ARCHIVE_STOPPED_WITHOUT_WAIT_RECEIPT')
    payload=ROOT/'archives'/run['run_id']/('global_step_'+str(step))
    sizes={};partial=0
    for p in payload.rglob('*') if payload.exists() else []:
        try:
            if not p.is_file() or p.is_symlink():continue
            st=p.stat()
        except FileNotFoundError:continue  # Atomic .partial publication can race this read-only snapshot.
        need(st.st_uid==OWNER,'archive payload has another owner')
        sizes[(st.st_dev,st.st_ino)]=st.st_size
        if p.name.endswith('.partial'):partial+=st.st_size
    value.update(actual_payload_bytes=sum(sizes.values()),actual_payload_files=len(sizes),partial_copy_bytes=partial)
    if (directory/'worker-result.json').is_file():
        result=read(directory/'worker-result.json');need(result['claim_sha256']==value['claim_sha256'] and result['token']==claim['token'],'CPU wait receipt differs')
        value.update(status=result['status'],actual_task_exit_code=result['actual_task_exit_code'])
        need(not live(result['actual_task_birth']),'archive child still live despite wait receipt')
        if result['status']=='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0':
            task=read(directory/'task-result.json');need(digest(directory/'task-result.json')==result['task_result_sha256'],'CPU task result changed')
            value.update(transport_path=task['archive']['path'],transport_sha256=task['archive']['sha256'])
            if full:
                from reproduction.hpc3_qprobe.final_restore_audit import verify_transport
                transport=read(task['archive']['path'])
                value['fresh_full_sha']=verify_transport(task['archive']['path'],transport['target_receipt']['path'],expected_step=step)
    return value


def final_ready(base,commit):
    repository,publication=source(commit);run,binding,accounting=producer(base,complete=True);base=Path(base)
    updates=read(base/'logs/full-run-acceptance.json');pipeline=read(base/'logs/pipeline-status.json');cleanup=read(base/'logs/cleanup-status.json')
    final=read(base/'logs/final-identity.json')
    need(updates['structural_accepted'] is True and updates['actual_outer_updates']==200 and updates['run_id']==run['run_id'] and
         pipeline['run_id']==run['run_id'] and pipeline['stage']=='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP' and
         cleanup['cleaned'] is True and cleanup['run_id']==run['run_id'] and final['run_id']==run['run_id'] and
         final['step']==200 and final['final'] is True and len(final['question_ids'])==len(final['em'])==len(final['f1'])==51713 and
         len(final['counts'])==7 and sum(final['counts'].values())==51713,'actual producer200/final51713/cleanup missing')
    records=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    offset=run.get('full_state_restore',{}).get('completed_outer',0)
    need([row['step'] for row in records]==list(range(offset+1,201)) and all(row['run_id']==run['run_id'] for row in records),
         'actual current successful suffix required; replayed prefix/update201 forbidden')
    from search_g1.qprobe_saved_checkpoint import history_parts
    from reproduction.hpc3_qprobe.verify_goal import part_updates
    records,history_proofs=part_updates(history_parts(base,run,last=200))
    need([row['step'] for row in records]==list(range(1,201)) and all(row['run_id']==run['run_id'] for row in records),
         'actual original successful1..200 required; update201 forbidden')
    proofs={}
    for step in STEPS:
        p,checkpoint=pair(base,run,step);value=inspect_archive(base,step,full=True)
        need(value['actual_task_exit_code']==0 and value['status']=='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0' and
             value.get('fresh_full_sha') is not None,'all five actual completed CPU archives required')
        transport=read(value['transport_path']);need(transport['source_receipt']==dict(path=str(p),sha256=checkpoint['sha256']),
                                                   'archive does not originate in this real producer pair')
        proofs[str(step)]=value['fresh_full_sha']
    receipt=proofs['200']['target_receipt']
    # Catch loading/science incompatibility before reserving a final allocation.
    from reproduction.hpc3_qprobe.final_restore_audit import bind_audit
    bind_audit(run,read(receipt['path']),Path(receipt['path']),ROOT.parent/'qprobe-teacher-warmup200-final200-audit-preflight',
               commit,read(publication['publication_path'])['files'],uuid.uuid4().hex)
    return repository,run,binding,dict(source=publication,actual_producer_accounting=accounting,
        final_identity_sha256=digest(base/'logs/final-identity.json'),producer_cleanup_sha256=digest(base/'logs/cleanup-status.json'),
        full_run_acceptance_sha256=digest(base/'logs/full-run-acceptance.json'),original_training_parts=history_proofs,archives=proofs)


def submit_audit(base,commit):
    repository,run,binding,proof=final_ready(base,commit);root=control_root(binding);directory=root/'final-audit'
    need(not directory.exists(),'final audit submission already attempted; inspect/reconcile, never replay')
    directory.mkdir(mode=0o770);token=uuid.uuid4().hex
    audit_base=ROOT.parent/('qprobe-teacher-warmup200-final200-audit-'+run['run_id'][:12]+'-'+token[:12])
    audit_base.mkdir(mode=0o770);(audit_base/'logs').mkdir(mode=0o770)
    transport=proof['archives']['200'];receipt=transport['target_receipt']['path']
    cmd=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=emergency_acd',
         '--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=12:00:00',
         '--exclude=ACD1-31,ACD1-67','--job-name=qprobe-final200audit','--export=NONE','--chdir='+str(audit_base),
         '--output='+str(audit_base/'logs/native-%j.log'),'--error='+str(audit_base/'logs/native-%j.log'),
         str(repository/'reproduction/hpc3_qprobe/final_restore_audit.sh'),str(audit_base),commit,receipt,transport['path']]
    claim=dict(schema_version=1,role='native4_final200_restore_only_submission',token=token,producer=binding,
        audit_base=str(audit_base),command=cmd,proof=proof,controller=birth(process(os.getpid())),
        created_unix=time.time(),dependency_policy='actual producer COMPLETED0 and final cleanup already verified; no retention-dependent Slurm dependency')
    publish(directory/'submit-claim.json',claim)
    test=subprocess.run(cmd[:1]+['--test-only']+cmd[1:],capture_output=True,text=True,timeout=30)
    publish(directory/'test-only.json',dict(command=cmd,returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
    need(test.returncode==0,'actual scheduler test-only failed; preserve claim, never replay')
    try:
        actual=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        value=dict(returncode=actual.returncode,stdout=actual.stdout,stderr=actual.stderr,status='SUBMIT_FAILED_DO_NOT_REPLAY')
        if actual.returncode==0:
            job=actual.stdout.strip().split(';')[0]
            if re.fullmatch('[1-9][0-9]*',job) is not None:value.update(job=job,status='ACTUAL_SLURM_SUBMITTED')
            else:value.update(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY',error='actual sbatch0 returned a malformed job id; raw output retained')
    except subprocess.TimeoutExpired as error:
        value=dict(returncode=None,status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY',error=repr(error))
    value.update(token=token,claim_sha256=digest(directory/'submit-claim.json'),created_unix=time.time(),
                 native_gpu_audit_submission_attempted=True,actual_cuda_restore_accepted=False)
    publish(directory/'submission.json',value);return value


def reconcile_audit(base,job):
    """Explicit unique-job adoption after unknown sbatch; never calls sbatch."""
    _,binding,_=producer(base,complete=True);directory=control_root(binding,create=False)/'final-audit';claim=read(directory/'submit-claim.json')
    need(re.fullmatch('[1-9][0-9]*',job) is not None and claim['producer']==binding,'actual job/producer binding required')
    need(not (directory/'reconciled-submission.json').exists(),'already reconciled; inspect instead')
    if (directory/'submission.json').is_file():
        value=read(directory/'submission.json')
        need(value.get('job',job)==job and value['returncode'] in (None,0),'cannot adopt a definite rejected submission')
    from reproduction.common8.native_allocation import fields
    raw=subprocess.check_output(['scontrol','show','job','-d',job],text=True,timeout=30);actual=fields(raw)
    repository=ROOT/'source'/claim['proof']['source']['commit']
    need(actual['JobId']==job and actual['UserId']=='jxu957(14090)' and actual['Account']=='jxu957' and
         actual['QOS']=='formal-user' and actual['Partition']=='emergency_acd' and actual['JobName']=='qprobe-final200audit' and
         actual['WorkDir']==claim['audit_base'] and actual['Command']==str(repository/'reproduction/hpc3_qprobe/final_restore_audit.sh') and
         actual['NumCPUs']=='48' and actual['MinMemoryNode']=='512G' and actual['NumNodes']=='1' and
         actual['TresPerNode'] in ('gres:gpu:4','gres:gpu:h100:4','gres/gpu:4') and
         actual['JobState'] in ('PENDING','RUNNING','COMPLETING','COMPLETED') and
         actual['NodeList'] not in ('ACD1-31','ACD1-67'),'actual final audit job does not match the one original intent')
    queue=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%128j'],text=True,timeout=30)
    matching={job}
    for line in queue.splitlines():
        other,name=(value.strip() for value in line.split('|',1))
        if name!='qprobe-final200audit' or other==job:continue
        detail=fields(subprocess.check_output(['scontrol','show','job','-d',other],text=True,timeout=30))
        if detail.get('UserId')=='jxu957(14090)' and detail.get('WorkDir')==claim['audit_base'] and detail.get('Command')==actual['Command']:
            matching.add(other)
    need(matching=={job},'more than one actual job matches this nonce workdir/source intent; diagnose, never replay')
    value=dict(status='ACTUAL_SLURM_RECONCILED',job=job,returncode=None,claim_sha256=digest(directory/'submit-claim.json'),
               token=claim['token'],actual_native_detail=raw,matching_active_intent_jobs=sorted(matching),
               uniqueness_scope='provided actual job plus current own active queue; exact nonce workdir and source command',checked_unix=time.time())
    publish(directory/'reconciled-submission.json',value);return value


def inspect_audit(base):
    _,binding,_=producer(base,complete=True);root=control_root(binding,create=False);directory=root/'final-audit'
    cpu=controller_state(root/'audit-controller') if (root/'audit-controller/submit-claim.json').is_file() else None
    if not (directory/'submit-claim.json').is_file():
        return dict(status='FINAL_SUBMIT_CPU_PREFLIGHT_RUNNING_OR_FAILED_NO_SLURM_INTENT',cpu_controller=cpu)
    claim=read(directory/'submit-claim.json')
    need(claim['producer']==binding,'audit producer differs')
    receipt=directory/'reconciled-submission.json'
    if not receipt.is_file():receipt=directory/'submission.json'
    value=dict(audit_base=claim['audit_base'],claim_sha256=digest(directory/'submit-claim.json'),
               status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY',job=None)
    if receipt.is_file():
        observed=read(receipt);need(observed['claim_sha256']==value['claim_sha256'],'audit receipt differs');value.update(observed)
    if value.get('job'):
        value['actual_accounting']=subprocess.check_output(['sacct','-nP','-X','-j',value['job'],
            '--format=JobID,State,ExitCode,UID,Account,QOS,Partition'],text=True,timeout=30)
    value['cpu_controller']=cpu
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('archive-start','archive-inspect','archive-worker','archive-exec','audit-start','audit-worker','audit-exec','audit-inspect','audit-reconcile'))
    p.add_argument('--base');p.add_argument('--commit');p.add_argument('--step',type=int);p.add_argument('--claim');p.add_argument('--token');p.add_argument('--job')
    a=p.parse_args();owner()
    if a.mode in ('archive-worker','archive-exec','audit-worker','audit-exec'):
        need(a.claim and a.token and not any((a.base,a.commit,a.step,a.job)),'worker cannot rebind controller arguments')
        result={'archive-worker':archive_worker,'archive-exec':archive_exec,
                'audit-worker':audit_worker,'audit-exec':audit_exec}[a.mode](a.claim,a.token)
    else:
        need(a.base and not any((a.claim,a.token)),'canonical producer base required')
        if a.mode in ('archive-start','audit-start'):need(a.commit,'published helper commit required')
        if a.mode.startswith('archive-'):need(a.step in STEPS and not a.job,'fixed archive step required')
        else:need(a.step is None,'final audit is always strict200; no alternate step accepted')
        if a.mode=='audit-reconcile':need(a.job and not a.commit,'one actual job required for explicit reconciliation')
        elif a.mode.startswith('audit-'):need(not a.job,'actual job may only be supplied to reconcile')
        if a.mode=='archive-start':result=launch_archive(a.base,a.commit,a.step)
        elif a.mode=='archive-inspect':result=inspect_archive(a.base,a.step)
        elif a.mode=='audit-start':result=start_audit(a.base,a.commit)
        elif a.mode=='audit-inspect':result=inspect_audit(a.base)
        else:result=reconcile_audit(a.base,a.job)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()

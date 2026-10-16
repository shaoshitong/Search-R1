"""Retire only the exact first150 CPU child after its actual hidepid refusal.

This is an actively retired CPU helper, never a spontaneous training failure.
The immutable S source, real failed DU, owner-authored visibility report and
original runner's real wait are retained. No actor, RAW, process group, source,
private directory mode, checkpoint or payload is changed. Python3.6 compatible.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid

from reproduction.hpc3_qprobe import owner_partition_storage as storage

ROOT=storage.ROOT
OLD='dc98dff19f64b3fc655ecf22d6be6991e16a1f3a'
ARCHIVE=ROOT/'final-control'/storage.RUN/'archive-150'
REQUEST=ARCHIVE/'owner-storage-attempt-1/request.json'
DIRECTORY=ROOT/'observations/archive150-hidepid-first-S-20261016'
FOLLOW=ROOT/'followthrough/721485-archive150-recovery-first-20261016'
PAYLOAD=ROOT/'archives'/storage.RUN/'global_step_150'
TARGET=dict(pid=230627,start_ticks=82969642,sid=224100,uid=14090)
RUNNER=dict(pid=224100,start_ticks=82967136,sid=224100,uid=14090)
FOLLOWER=dict(pid=136541,start_ticks=82929213,sid=136541,uid=14090)
CLAIM_SHA='72ac35da422c8ab388919951f08415a36d7cc89ff243043a510044bbdafbd6c7'
REQUEST_SHA='f7d5f8c56dc7a3f547e33c4a36a40127e9a1e98f6cb730adc1d467cee4a58d85'
DU_SHA='f983a12eb0f69a5c044a0e0bdb2f39ff5add75199283d31a5b6f278a288caaa2'
NEW_DENIAL=str(ROOT.parent/'qprobe-teacher-warmup200-periodic25-thread5-20261016-r16/tmp/ray-thread-smoke-72d37ba2')
MODULE='reproduction.hpc3_qprobe.archive150_visibility_failure'
RELATIVE='reproduction/hpc3_qprobe/archive150_visibility_failure.py'
TTL=120
WALL=0x40000000
SEIZE=0x4206
INTERRUPT=0x4207
DETACH=17
STATUS='ACTUAL_FIRST150_CPU_HELPER_RETIRED_AFTER_HIDEPID_REFUSAL'


def need(ok,message):storage.need(ok,'VISIBILITY_FAILURE: '+message)


def protected_publish(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    fd=os.open(str(temporary),os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o640)
    with os.fdopen(fd,'w') as stream:
        os.fchmod(stream.fileno(),0o640)
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(str(temporary),str(path))
    finally:temporary.unlink()
    storage.fsync_dir(path.parent)


def protected_document(path,owner):
    path=Path(path);before=path.lstat()
    need(path.is_absolute() and path.resolve()==path and not path.is_symlink(),'unsafe protected path')
    fd=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        opened=os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode)==0o640 and
             opened.st_uid==owner and opened.st_gid==storage.GID and opened.st_size<4*2**20,
             'protected owner/group/mode differs')
        raw=stream.read(4*2**20);after=os.fstat(stream.fileno())
    latest=path.lstat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_gid,s.st_mode)
    need(identity(before)==identity(opened)==identity(after)==identity(latest) and len(raw)==opened.st_size,
         'protected bytes changed while reading')
    return json.loads(raw),dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def host():
    storage.host_guard()
    return dict(hostname='ACD-Login-1',boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid_namespace=os.readlink('/proc/self/ns/pid'))


def guard(commit,owner):
    repo,source=storage.code_guard(commit,owner)
    need(Path(__file__).resolve()==repo/RELATIVE and storage.digest(repo/RELATIVE)==
         storage.document(source['publication_path'],(storage.JXU,))[0]['files'][RELATIVE],
         'visibility helper is not its published source')
    return repo,source


def old_module():
    path=ROOT/'source'/OLD/'reproduction/hpc3_qprobe/owner_partition_storage.py'
    spec=importlib.util.spec_from_file_location('_actual_first150_storage_S',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def exists(path):return Path(path).exists() or Path(path).is_symlink()


def preflight():
    old=old_module();request,request_proof=old.request_guard(REQUEST,OLD,active=False)
    claim,claim_proof=storage.document(ARCHIVE/'submit-claim.json',(storage.JXU,))
    runner,_=storage.document(ARCHIVE/'runner-binding.json',(storage.JXU,))
    child,_=storage.document(ARCHIVE/'task-binding.json',(storage.JXU,))
    need(request_proof['sha256']==REQUEST_SHA and claim_proof['sha256']==CLAIM_SHA and
         request['archive_child']==child['actual_process']==TARGET and runner['actual_process']==RUNNER and
         child['supervisor']==RUNNER and claim['controller']==FOLLOWER and claim['step']==150 and
         claim['source']['commit']==OLD and claim['token']==request['archive_token'],
         'exact original CPU request/claim/birth differs')
    for path in (REQUEST.parent/'mou-control',ARCHIVE/'task-result.json',ARCHIVE/'worker-result.json',
                 ARCHIVE/'storage-owner-partition-accepted.json',PAYLOAD,ARCHIVE.parent/'archive-200',
                 ARCHIVE.parent/'audit-controller',ARCHIVE.parent/'final-audit'):
        need(not exists(path),'owner, payload, accepted storage or later component exists; refuse retirement')
    du,du_proof=old.consume_du(REQUEST.parent,'jxu-du',old.du_command([old.MEASURE],request['excluded_paths']),storage.JXU)
    need(du_proof['sha256']==DU_SHA and du['actual_parent']==TARGET and du['actual_du_wait_exit_code']==1 and
         du['stderr']=="du: cannot read directory '"+NEW_DENIAL+"': Permission denied\n" and
         not old.live(du['actual_du']),'exact original failed DU must be fully waited, never live or accepted usage')
    return dict(request=request_proof,claim=claim_proof,old_source=request['source'],actual_child=TARGET,
                actual_runner=RUNNER,failed_du=du_proof,failed_du_exit=1,failed_stdout_is_usage=False)


def report_visibility(commit):
    _,source=guard(commit,storage.MOU);facts=preflight()
    need(not exists(DIRECTORY),'existing visibility report is inspect-only; never replay')
    mounts=Path('/proc/mounts').read_text()
    need(any(v.split()[1:3]==['/proc','proc'] and 'hidepid=2' in v.split()[3].split(',')
             for v in mounts.splitlines()),'actual hidepid=2 mount is missing')
    errors={}
    for name in ('stat','status'):
        try:(Path('/proc')/str(TARGET['pid'])/name).read_bytes()
        except FileNotFoundError as error:errors[name]=dict(errno=error.errno,kind=type(error).__name__)
        else:raise ValueError('foreign target is visible; actual diagnosis differs')
    own=storage.process(os.getpid());need(own is not None and own['uid']==storage.MOU,'own MOU process not visible')
    DIRECTORY.mkdir(mode=0o770)
    report=dict(status='UNKNOWN_CROSS_OWNER_PROCESS_VISIBILITY_HIDEPID2',source=source,facts=facts,
        host=host(),actual_reporter=storage.birth(own),foreign_proc_errors=errors,
        foreign_process_declared_dead=False,created_unix=time.time(),training_actions=0,permission_changes=0,signals_sent=0)
    protected_publish(DIRECTORY/'mou-report.json',report)
    return protected_document(DIRECTORY/'mou-report.json',storage.MOU)[1]


def visibility_report(facts,source,*,fresh=True):
    report,proof=protected_document(DIRECTORY/'mou-report.json',storage.MOU)
    need(report['status']=='UNKNOWN_CROSS_OWNER_PROCESS_VISIBILITY_HIDEPID2' and
         report['facts']==facts and report['source']['commit']==source['commit'] and
         report['host']==host() and report['actual_reporter']['uid']==storage.MOU and
         report['foreign_process_declared_dead'] is False and report['signals_sent']==report['training_actions']==
         report['permission_changes']==0 and report['foreign_proc_errors']==
         {name:dict(errno=2,kind='FileNotFoundError') for name in ('stat','status')},'actual owner visibility report differs')
    if fresh:need(0<=time.time()-report['created_unix']<=TTL,'actual MOU visibility report is stale')
    return report,proof


def target_identity(target,command,cwd,*,tracer):
    now=storage.process(target['pid']);need(now is not None and storage.birth(now)==target and now['state']!='Z',
                                          'target birth changed; no signal')
    root=Path('/proc')/str(target['pid']);status=(root/'status').read_text()
    fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines() if ':' in line}
    need(int(fields['TracerPid'])==tracer and int(fields['Threads'])==1 and int(fields['PPid'])==target['sid'] and
         (root/'cmdline').read_bytes().decode().split('\0')[:-1]==command and os.readlink(str(root/'cwd'))==str(cwd),
         'target tracer/thread/parent/command/cwd differs; no signal')
    return now


def ptrace(request,pid,data=0):
    library=ctypes.CDLL(None,use_errno=True);library.ptrace.restype=ctypes.c_long
    value=library.ptrace(ctypes.c_uint(request),ctypes.c_uint(pid),ctypes.c_void_p(0),ctypes.c_void_p(data))
    if value==-1:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))


def bound_sigterm(target,command,cwd,before_signal,timeout=15):
    """Only call in a dedicated short tracer process, which must then exit.

    SEIZE uses no EXITKILL and sends no stop signal. Before interrupt, and again
    after the exact task's ptrace-stop, recheck all identity fields. While this
    attached stopped task has not been terminally reaped, its PID cannot be
    reused. Queue SIGTERM there and detach/resume with no injected event-stop
    signal. A wrong-birth/error tracer exits and Linux auto-detaches it.
    """
    need(sys.platform=='linux' and target['uid']==os.getuid(),'Linux own target required')
    target_identity(target,command,cwd,tracer=0)
    ptrace(SEIZE,target['pid'],0)  # Never PTRACE_O_EXITKILL.
    target_identity(target,command,cwd,tracer=os.getpid())
    ptrace(INTERRUPT,target['pid'])
    deadline=time.monotonic()+timeout
    while True:
        pid,status=os.waitpid(target['pid'],os.WNOHANG|WALL)
        if pid:
            need(pid==target['pid'] and os.WIFSTOPPED(status) and status>>16==128,
                 'exact seized task did not enter PTRACE_EVENT_STOP; no signal')
            break
        need(time.monotonic()<deadline,'ptrace-stop timeout; no signal');time.sleep(.01)
    target_identity(target,command,cwd,tracer=os.getpid())
    before_signal()  # Retained scope guards and durable intent still exact.
    os.kill(target['pid'],signal.SIGTERM)  # PID is reserved by the unreaped attached task.
    ptrace(DETACH,target['pid'],0)  # Deliver queued SIGTERM through normal pending-signal processing.
    return dict(method='SEIZE_EXACT_BIRTH_INTERRUPT_UNREAPED_STOP_QUEUE_SIGTERM_DETACH',
                exact_target=target,ptrace_stop_status=status,signals_queued=1,process_group_signals=0,exitkill=False)


def expected_command(claim):
    return ['/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/bin/python3.10',
            '-m','reproduction.hpc3_qprobe.final_control','archive-exec','--claim',str(ARCHIVE/'submit-claim.json'),
            '--token',claim['token']]


def trace_exec(commit,token):
    _,source=guard(commit,storage.JXU);intent,intent_proof=protected_document(DIRECTORY/'intent.json',storage.JXU)
    bound,_=protected_document(DIRECTORY/'tracer-binding.json',storage.JXU)
    own=storage.birth(storage.process(os.getpid()))
    need(intent['source']==source and intent['token']==token==bound['token'] and bound['intent']==intent_proof and
         bound['actual_process']==own and storage.live(intent['actual_controller']), 'actual tracer binding differs')
    def recheck():
        facts=preflight();need(facts==intent['facts'],'retirement scope changed')
        _,proof=visibility_report(facts,source);need(proof==intent['mou_report'],'visibility report changed')
        need(storage.live(RUNNER) and storage.live(FOLLOWER),'original runner/follower no longer live')
        need(protected_document(DIRECTORY/'intent.json',storage.JXU)[1]==intent_proof,'retirement intent changed')
    recheck();result=bound_sigterm(TARGET,intent['expected_command'],ROOT/'source'/OLD,recheck)
    protected_publish(DIRECTORY/'tracer-result.json',dict(source=source,token=token,intent=intent_proof,
        actual_process=own,result=result,completed_unix=time.time(),training_actions=0,permission_changes=0))


def retire(commit):
    repo,source=guard(commit,storage.JXU);facts=preflight();_,report=visibility_report(facts,source)
    need(not exists(DIRECTORY/'intent.json'),'existing retirement intent is inspect-only; never replay')
    claim,_=storage.document(ARCHIVE/'submit-claim.json',(storage.JXU,));command=expected_command(claim)
    target_identity(TARGET,command,ROOT/'source'/OLD,tracer=0)
    need(storage.live(RUNNER) and storage.live(FOLLOWER),'original CPU runner/follower is not live')
    token=uuid.uuid4().hex
    protected_publish(DIRECTORY/'intent.json',dict(source=source,token=token,facts=facts,mou_report=report,
        actual_controller=storage.birth(storage.process(os.getpid())),expected_command=command,
        actively_retired_cpu_helper=True,training_actions=0,permission_changes=0,created_unix=time.time()))
    _,intent_proof=protected_document(DIRECTORY/'intent.json',storage.JXU)
    command=[sys.executable,'-m',MODULE,'trace-exec','--commit',commit,'--token',token]
    # A pipe gates entry until the parent publishes the actual tracer birth.
    with (DIRECTORY/'tracer.log').open('x') as output:
        child=subprocess.Popen(command,cwd=str(repo),env=dict(os.environ),stdin=subprocess.PIPE,stdout=output,stderr=subprocess.STDOUT)
        try:
            actual=storage.process(child.pid);need(actual is not None and actual['uid']==storage.JXU,'tracer birth differs')
            protected_publish(DIRECTORY/'tracer-binding.json',dict(actual_process=storage.birth(actual),intent=intent_proof,
                source=source,token=token,command=command,created_unix=time.time()))
        finally:
            child.stdin.close();code=child.wait()
    protected_publish(DIRECTORY/'tracer-held.json',dict(actual_tracer=storage.birth(actual),source=source,token=token,
        actual_tracer_wait_exit_code=code,completed_unix=time.time()))
    need(code==0,'actual tracer failed; inspect retained outcome, never replay')
    result,result_proof=protected_document(DIRECTORY/'tracer-result.json',storage.JXU)
    need(result['actual_process']==storage.birth(actual) and result['source']==source and result['token']==token and
         result['intent']==intent_proof and result['result']['exact_target']==TARGET and result['result']['signals_queued']==1,
         'actual tracer result/birth differs')
    deadline=time.monotonic()+180
    while storage.live(RUNNER) or storage.live(FOLLOWER):
        need(time.monotonic()<deadline,'original CPU retirement outcome pending; inspect no replay');time.sleep(1)
    held,held_proof=storage.document(ARCHIVE/'worker-result.json',(storage.JXU,))
    stopped,stopped_proof=storage.document(FOLLOW/'result.json',(storage.JXU,))
    need(not storage.live(TARGET) and held['actual_task_birth']==TARGET and held['actual_supervisor']==RUNNER and
         held['actual_task_exit_code']==-signal.SIGTERM and held['claim_sha256']==CLAIM_SHA and
         held['status']=='CPU_CHILD_FAILED_OR_MISSING_RECEIPT' and stopped['actual_process']==FOLLOWER and
         stopped['exit_code']==1 and stopped['error']=="ValueError('QPROBE_FOLLOWTHROUGH_SIGMA3_ACTUAL_ARCHIVE_FAILED_150')" and
         not exists(PAYLOAD) and not exists(REQUEST.parent/'mou-control') and not exists(ARCHIVE/'task-result.json') and
         not exists(ARCHIVE/'storage-owner-partition-accepted.json'), 'original real wait/follower/zero payload conjunction differs')
    value=dict(status=STATUS,source=source,intent=intent_proof,mou_report=report,tracer_result=result_proof,
        original_runner_wait=held_proof,original_follower_result=stopped_proof,actual_child=TARGET,actual_runner=RUNNER,
        actual_follower=FOLLOWER,actual_archive_child_wait_exit_code=-signal.SIGTERM,
        actively_retired_cpu_helper=True,spontaneous_failure_claimed=False,training_actions=0,
        permission_changes=0,process_group_signals=0,zero_payload=True,completed_unix=time.time())
    protected_publish(DIRECTORY/'complete.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('report-visibility','retire','trace-exec'));parser.add_argument('--commit',required=True)
    parser.add_argument('--token');args=parser.parse_args()
    if args.mode=='trace-exec':
        sys.stdin.buffer.read()  # Parent binding publication completed before EOF.
        trace_exec(args.commit,args.token)
    else:print(json.dumps(report_visibility(args.commit) if args.mode=='report-visibility' else retire(args.commit)),flush=True)

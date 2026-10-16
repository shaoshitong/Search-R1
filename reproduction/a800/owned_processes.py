"""Project subprocess ownership with PID/start ticks/session/UID and durable ledger.

Only bound project sessions are eligible; unknown members after leader exit cause
safe refusal. Never enumerate or stop another project's Ray service by name.
"""
import json
import errno
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    p=Path('/proc')/str(pid);text=(p/'stat').read_text();f=text[text.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(f[19]),sid=int(f[3]),uid=p.stat().st_uid,state=f[0],
                command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))


def same(record):
    try:
        current=identity(record['pid'])
        return all(current[k]==record[k] for k in ('start_ticks','sid','uid'))
    except (FileNotFoundError,ProcessLookupError):return False


def signal_owned(record, sig):
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise OSError(errno.ENOSYS,'Kernel/Python pidfd interface unavailable')
    fd=os.pidfd_open(record['pid'])
    try:
        assert same(record), 'Owned identity changed before signal'
        signal.pidfd_send_signal(fd,sig)
    finally:os.close(fd)


def signal_owned_child_group(record, process, sig):
    """Only a retained, unreaped direct child can pin cleanup on old kernels.

    A recovered ledger or a reaped leader must refuse this compatibility route.
    The wait lock prevents Popen from reaping that PID during validation/signal.
    """
    assert process is not None and process.pid==record['pid']==record['sid']
    assert signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL
    with process._waitpid_lock:
        assert process.returncode is None, 'Reaped leader cannot pin legacy cleanup'
        # ECHILD rejects another parent's handle. WNOWAIT retains a zombie PID
        # too; it cannot be recycled into a foreign group until it is reaped.
        os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
        assert same(record), 'Owned direct-child identity changed before signal'
        assert os.getpgid(process.pid)==record['sid'], 'Owned process group changed'
        os.killpg(record['sid'],sig)


def publish(path, data):
    tmp=path.with_suffix('.json.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class OwnedProcesses:
    def __init__(self,path,run_id):
        self.path=Path(path);self.run_id=run_id;self.jobs={};self.processes={}
        if self.path.exists():
            saved=json.loads(self.path.read_text());assert saved['run_id']==run_id
            self.jobs=saved['jobs'];self.recovery_only=True
        else:
            self.recovery_only=False;self.save()
    def save(self):publish(self.path,dict(run_id=self.run_id,jobs=self.jobs))
    def start(self,name,args,env,log,cwd):
        assert not self.recovery_only, 'Recovered ledger permits cleanup only, never duplicate launch'
        assert name not in self.jobs
        with Path(log).open('a') as output:
            process=subprocess.Popen(args,env=env,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        # Popen returns after exec; bind before poll/wait can reap a fast-exit leader.
        leader=identity(process.pid);assert leader['sid']==leader['pid'] and leader['uid']==os.getuid()
        self.jobs[name]=dict(leader=leader,ledger={str(leader['pid']):leader},log=str(log),state='STARTED',arguments=args)
        self.processes[name]=process
        self.save();self.observe(name)
        return process
    def observe(self,name):
        job=self.jobs[name];leader=job['leader'];live=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:r=identity(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if r['state']!='Z' and r['sid']==leader['sid'] and r['uid']==leader['uid']:live.append(r)
        valid_leader=same(leader)
        for r in live:
            old=job['ledger'].get(str(r['pid']))
            if old:
                assert all(old[k]==r[k] for k in ('start_ticks','sid','uid')), 'Recorded subprocess changed identity'
                if old['command']!=r['command']:
                    job.setdefault('command_history',[]).append(dict(pid=r['pid'],before=old['command'],after=r['command'],unix=time.time()))
                    job['ledger'][str(r['pid'])]=r
            else:
                assert valid_leader, 'Unbound process after session leader exit'
                job['ledger'][str(r['pid'])]=r
        self.save();return live
    def stop(self,name):
        job=self.jobs[name];leader=job['leader']
        if job['state']=='STOPPED':return
        try:
            self.observe(name);job['state']='CLEANUP_PENDING';self.save()
            legacy=False
            if same(leader) and identity(leader['pid'])['state']!='Z':
                try:signal_owned(leader,signal.SIGINT)
                except OSError as error:
                    if error.errno!=errno.ENOSYS:raise
                    signal_owned_child_group(leader,self.processes.get(name),signal.SIGINT)
                    legacy=True
            end=time.monotonic()+15
            while time.monotonic()<end and self.observe(name):time.sleep(.2)
            for sig in (signal.SIGTERM,signal.SIGKILL):
                live=self.observe(name)
                if legacy and live:
                    signal_owned_child_group(leader,self.processes.get(name),sig)
                else:
                    for r in live:
                        try:signal_owned(r,sig)
                        except ProcessLookupError:pass
                end=time.monotonic()+3
                while time.monotonic()<end and self.observe(name):time.sleep(.2)
                if not self.observe(name):break
            assert not self.observe(name), 'Owned component cleanup incomplete'
            job['state']='STOPPED';self.save()
        except Exception as e:
            job.update(state='CLEANUP_FAILED',cleanup_error=repr(e));self.save();raise
    def stop_all(self):
        errors=[]
        for name in reversed(self.jobs):
            try:self.stop(name)
            except Exception as e:errors.append(dict(name=name,error=repr(e)))
        if errors:raise RuntimeError(json.dumps(errors))

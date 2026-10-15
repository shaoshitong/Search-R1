"""Project subprocess ownership with PID/start ticks/session/UID and durable ledger.

Only bound project sessions are eligible; unknown members after leader exit cause
safe refusal. Never enumerate or stop another project's Ray service by name.
"""
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time


def identity(pid):
    p=Path('/proc')/str(pid);text=(p/'stat').read_text();f=text[text.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(f[19]),sid=int(f[3]),uid=p.stat().st_uid,state=f[0],ppid=int(f[1]),
                command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))


def same(record):
    try:
        current=identity(record['pid'])
        return all(current[k]==record[k] for k in ('pid','start_ticks','sid','uid'))
    except (FileNotFoundError,ProcessLookupError):return False


def _linux_pidfd_syscall(number, *arguments):
    # Python builds can omit the wrappers even when the running kernel supports
    # pidfds. These syscall numbers are shared by the explicitly supported ABIs;
    # unknown platforms must fail closed rather than signal a recyclable PID.
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise OSError(errno.ENOSYS, 'No supported race-safe pidfd implementation')
    library=ctypes.CDLL(None,use_errno=True)
    syscall=library.syscall
    syscall.restype=ctypes.c_long
    result=syscall(ctypes.c_long(number),*arguments)
    if result == -1:
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error))
    return int(result)


def _pidfd_open(pid):
    native=getattr(os,'pidfd_open',None)
    if native is not None:
        return native(pid,0)
    return _linux_pidfd_syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))


def _pidfd_send_signal(fd,sig):
    native=getattr(signal,'pidfd_send_signal',None)
    if native is not None:
        return native(fd,sig,None,0)
    return _linux_pidfd_syscall(424,ctypes.c_int(fd),ctypes.c_int(sig),
                               ctypes.c_void_p(None),ctypes.c_uint(0))


def _signal_with_pidfd(record,sig):
    fd=_pidfd_open(record['pid'])
    try:
        assert same(record), 'Owned identity changed before signal'
        _pidfd_send_signal(fd,sig)
    finally:os.close(fd)


def _require_waitable_children():
    """Fail closed unless this supported glibc process retains child zombies."""
    if (platform.system()!='Linux' or platform.machine() not in ('x86_64','aarch64')
            or ctypes.sizeof(ctypes.c_ulong)!=8):
        raise OSError(errno.ENOSYS,'Unsupported direct-child signal ABI')
    if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
        raise RuntimeError('Direct-child signaling requires default SIGCHLD')
    library=ctypes.CDLL(None,use_errno=True)
    if not hasattr(library,'gnu_get_libc_version'):
        raise OSError(errno.ENOSYS,'Direct-child signal fallback requires verified glibc ABI')
    class Sigaction(ctypes.Structure):
        # Supported 64-bit Linux glibc: handler,1024-bit sigset,flags,restorer.
        _fields_=[('handler',ctypes.c_void_p),('mask',ctypes.c_ulong*16),
                  ('flags',ctypes.c_int),('restorer',ctypes.c_void_p)]
    if ctypes.sizeof(Sigaction)!=152 or Sigaction.flags.offset!=136:
        raise OSError(errno.ENOSYS,'Unknown glibc sigaction layout')
    action=Sigaction();query=library.sigaction
    query.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(Sigaction)]
    query.restype=ctypes.c_int
    if query(signal.SIGCHLD,None,ctypes.byref(action))!=0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    if action.handler is not None or action.flags&2:  # SA_NOCLDWAIT
        raise RuntimeError('Kernel SIGCHLD disposition can reap children without the Popen lock')


def _signal_direct_child(record,sig,process):
    """Signal only a retained unreaped Popen child, never a ledger-only PID.

    Holding the same lock used by Popen.poll/wait prevents child reaping between
    verification and signaling. With default SIGCHLD and no SA_NOCLDWAIT, even
    an exit during this interval leaves a zombie whose PID cannot be reused.
    This parent uses Popen for all child waits and installs no SIGCHLD handler.
    Descendants and recovered ledgers have no such handle and fail closed.
    """
    if (not isinstance(process,subprocess.Popen) or process.pid!=record['pid']
            or record['sid']!=record['pid'] or record['uid']!=os.getuid()):
        raise RuntimeError('No retained direct-child process authority')
    lock=getattr(process,'_waitpid_lock',None)
    if lock is None or not lock.acquire(False):
        raise RuntimeError('Direct-child wait lock is unavailable')
    try:
        _require_waitable_children()
        if process.returncode is not None:
            raise ProcessLookupError(errno.ESRCH,'Owned child was already reaped')
        waited,status=os.waitpid(process.pid,os.WNOHANG)
        if waited==process.pid:
            process._handle_exitstatus(status)
            raise ProcessLookupError(errno.ESRCH,'Owned child already exited')
        if waited!=0:
            raise RuntimeError('Unexpected direct-child wait result')
        current=identity(process.pid)
        assert all(current[key]==record[key] for key in ('pid','start_ticks','sid','uid')), \
            'Owned direct-child identity changed before signal'
        assert current['ppid']==os.getpid(), 'Owned process is no longer our direct child'
        # No poll/wait occurs until the signal is issued and the lock released.
        # This numeric PID is anchored by the unreaped-child kernel relation.
        os.kill(process.pid,sig)
    finally:lock.release()


def signal_owned(record,sig,*,process=None):
    try:
        return _signal_with_pidfd(record,sig)
    except OSError as error:
        if error.errno!=errno.ENOSYS or process is None:raise
        return _signal_direct_child(record,sig,process)


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
        self.path=Path(path);self.run_id=run_id;self.jobs={};self.handles={}
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
        self.handles[name]=process
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
            if same(leader) and identity(leader['pid'])['state']!='Z':
                try:signal_owned(leader,signal.SIGINT,process=self.handles.get(name))
                except ProcessLookupError:pass
            end=time.monotonic()+15
            while time.monotonic()<end and self.observe(name):time.sleep(.2)
            for sig in (signal.SIGTERM,signal.SIGKILL):
                for r in self.observe(name):
                    try:signal_owned(r,sig,process=self.handles.get(name) if r['pid']==leader['pid'] else None)
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

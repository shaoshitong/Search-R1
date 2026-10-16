"""Isolate prefix20 JIT cache writers on the shared HPC3 filesystem."""
import json
import os
from pathlib import Path
import re
import socket


def process_identity():
    pid=os.getpid()
    stat=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
    return dict(host=socket.gethostname(),pid=pid,start_ticks=int(stat[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uid=os.getuid())


def isolate(base, identity):
    base=Path(base)
    host=identity['host'];boot=identity['boot_id']
    if (not re.fullmatch('[A-Za-z0-9._-]+',host) or
            not re.fullmatch('[0-9a-f-]{36}',boot) or
            any(type(identity[k]) is not int or identity[k]<1 for k in ('pid','start_ticks'))):
        raise ValueError('PREFIX20_JIT_PROCESS_IDENTITY_INVALID')
    key='%s-%s-p%d-s%d'%(host,boot,identity['pid'],identity['start_ticks'])
    root=base/'cache/process-jit'/key
    if any(path.is_symlink() for path in (root,*root.parents)):
        raise ValueError('PREFIX20_JIT_CACHE_PATH_NOT_OWNED_REGULAR_DIRECTORY')
    root.mkdir(parents=True,exist_ok=True)
    if any(path.is_symlink() for path in (root,*root.parents)) or root.stat().st_uid!=identity['uid']:
        raise ValueError('PREFIX20_JIT_CACHE_PATH_NOT_OWNED_REGULAR_DIRECTORY')
    values={name:str(root/child) for name,child in (
        ('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'))}
    for path in values.values():
        directory=Path(path);directory.mkdir(exist_ok=True)
        if directory.is_symlink():raise ValueError('PREFIX20_JIT_CACHE_CHILD_SYMLINK')
    os.environ.update(values)
    proof=dict(process=identity,paths=values,policy='prefix20_process_identity_isolation_v1')
    path=root/'identity.json'
    if path.exists():
        if json.loads(path.read_text())!=proof:raise ValueError('PREFIX20_JIT_EXISTING_PROCESS_BINDING_DIFFERS')
    else:
        with path.open('x') as stream:json.dump(proof,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    return proof


def configure():
    value=os.environ.get('SEARCH_G1_BASE','')
    # Other algorithms and already-deployed checkouts retain their own policy.
    if not re.fullmatch('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-[A-Za-z0-9._-]+',value):
        return None
    base=Path(value)
    if base.resolve()!=base:raise ValueError('PREFIX20_JIT_DEDICATED_BASE_REQUIRED')
    return isolate(base,process_identity())

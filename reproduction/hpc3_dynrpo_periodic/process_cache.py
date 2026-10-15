"""Isolate Triton/Inductor metadata for each process in this experiment only."""
import os
from pathlib import Path
import re

PREFIX='/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016'


def cache_paths(base,pid):
    if re.fullmatch(re.escape(PREFIX)+r'(?:-r[1-9][0-9]*)?',str(base)) is None:
        return None
    if type(pid) is not int or pid<=0:
        raise ValueError('Invalid compilation-cache process identity')
    root=Path(base)/'cache'
    return dict(TRITON_CACHE_DIR=str(root/'triton'/('process-'+str(pid))),
                TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'/('process-'+str(pid))))


def bind_process_cache():
    values=cache_paths(os.environ.get('SEARCH_G1_BASE',''),os.getpid())
    if values is None:
        return None
    base=Path(os.environ['SEARCH_G1_BASE'])
    if base.resolve()!=base or base.is_symlink() or base.stat().st_uid!=os.getuid():
        raise ValueError('Compilation cache base is not this owned canonical experiment')
    for key,name in values.items():
        path=Path(name)
        path.mkdir(parents=True,exist_ok=True)
        if path.resolve()!=path or path.stat().st_uid!=os.getuid():
            raise ValueError('Compilation cache is not a private process directory')
        # Derive from this PID, never inherit a torchrun/Ray parent's namespace.
        os.environ[key]=name
    return values

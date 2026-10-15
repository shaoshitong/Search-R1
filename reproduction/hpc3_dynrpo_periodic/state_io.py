"""Durable state writes limited to this experiment or an explicit restart."""
import json
import os
from pathlib import Path
import re

from .common import BASE, DEFAULT_BASE


def active_base():
    base=Path(os.environ.get('SEARCH_G1_BASE',str(BASE)))
    if (base.resolve()!=base or base.is_symlink() or
            re.fullmatch(re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?',str(base)) is None or
            base.stat().st_uid != os.getuid()):
        raise ValueError('PERIODIC_STATE_OUTSIDE_OWN_EXPERIMENT_OR_RESTART')
    return base


def _target(path):
    path=Path(path)
    if not path.is_relative_to(active_base()) or path.is_symlink():
        raise ValueError('PERIODIC_STATE_OUTPUT_ESCAPES_ACTIVE_NAMESPACE')
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def save(path,value):
    import torch
    path=_target(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    temporary.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def publish(path,value):
    path=_target(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as f:
        json.dump(value,f,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

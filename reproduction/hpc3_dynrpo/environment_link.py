"""Complete a missing read-only package reference without changing live source."""
import json
import os
from pathlib import Path
import time


def main():
    base=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-native-thread5-20261015')
    source=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/env/lib/python3.10/site-packages')
    if base.resolve()!=base or base.stat().st_uid!=os.getuid():raise ValueError('Not the own independent environment')
    if not (source/'torch/lib').is_dir() or not (source/'nvidia').is_dir():raise ValueError('Verified CUDA library tree unavailable')
    target=base/'env/lib/python3.10/site-packages';target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=source.resolve():raise ValueError('Existing package reference differs')
    elif target.exists():raise ValueError('Do not replace an existing environment')
    else:target.symlink_to(source.resolve())
    receipt=dict(status='OWN_READONLY_CUDA_PACKAGE_REFERENCE_READY',path=str(target),resolved=str(target.resolve()),
        runtime_source_modified=False,shared_packages_modified=False,checked_unix=time.time())
    output=base/'preparation/environment-link.json'
    if not output.exists():
        with output.open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    print(json.dumps(receipt))


if __name__=='__main__':main()

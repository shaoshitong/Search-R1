"""Publish immutable A800 normal occurrence inputs, never score-only records."""
import argparse
import os
import fcntl
from pathlib import Path
from .core import atomic, canonical, digest, read, sha, utc

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--normal-root',required=True);p.add_argument('--count',required=True,type=int)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.normal_root);binding=read(root/'binding.json')
    complete=read(root/'normal-complete.json')
    if complete['target']<args.count or complete['completed']<args.count:
        raise ValueError('Normal worker has not completed requested occurrences')
    files=sorted((root/'records').glob('*.json'))[:args.count]
    if len(files)!=args.count:raise ValueError('Missing source occurrence files')
    records=[];seen=set()
    for f in files:
        row=read(f);checksum=row.pop('source_record_hash')
        if digest(row)!=checksum:raise ValueError('Normal row checksum mismatch')
        row['source_record_hash']=checksum
        key=tuple([row['data_source'],row['evaluation_ordinal'],row['qid']])
        if key in seen:raise ValueError('Repeated occurrence')
        if row['normal_config_sha256']!=binding['normal_config_sha256'] or row['checkpoint_id']!=binding['checkpoint_identity_sha256']:
            raise ValueError('Foreign model/config normal row')
        if any(k not in row for k in ('model_spans','seen_docs','actual_retrieval_calls','gold','question')):
            raise ValueError('Incomplete raw normal schema')
        seen.add(key);records.append(row)
    dest=Path(args.output);temp=dest.with_name(dest.name+'.tmp')
    lock=open(dest.with_name(dest.name+'.export.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if dest.exists():
        expected=b''.join(canonical(r)+b'\n' for r in records)
        if dest.read_bytes()!=expected:raise ValueError('Immutable export differs')
        return
    # Only this exporter owns this exact task temp path while holding its lock.
    # An interrupted partial is safely rewritten from immutable source records.
    with open(temp,'wb') as f:
        for r in records:f.write(canonical(r)+b'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,dest)
    if not (root/'normal-config.json').exists():atomic(root/'normal-config.json',binding['normal_config'])
    atomic(dest.with_suffix(dest.suffix+'.receipt.json'),dict(count=len(records),sha256=sha(dest),
        bytes=dest.stat().st_size,source_binding_sha256=sha(root/'binding.json'),utc=utc()))

if __name__=='__main__':main()

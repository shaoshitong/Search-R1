"""Complete a bounded du permission partition, without changing private modes."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha,atomic
from reproduction.prefix20_joint4.control import PERSONAL_ROOT,own_base,publish
from reproduction.prefix20_joint4.storage_observe import identity


KIND='permission_partition_allocated_byte_upper_bound'


def denied_paths(stderr):
    paths=[]
    for line in stderr.splitlines():
        match=re.fullmatch(r"du: cannot read directory '([^']+)': Permission denied",line)
        if match is None:raise ValueError('PREFIX20_UNEXPECTED_ROOT_DU_ERROR')
        path=Path(match.group(1))
        if not path.is_relative_to(PERSONAL_ROOT) or path==Path(PERSONAL_ROOT) or '..' in path.parts:
            raise ValueError('PREFIX20_DENIED_PATH_OUTSIDE_PERSONAL_ROOT')
        paths.append(str(path))
    if not paths or len(set(paths))!=len(paths):raise ValueError('PREFIX20_MISSING_OR_DUPLICATE_DENIED_PARTITION')
    return paths


def directory_identity(path):
    if not path.is_dir() or any(p.is_symlink() for p in [path,*path.parents]):
        raise ValueError('PREFIX20_STORAGE_PARTITION_MISSING_OR_SYMLINK')
    s=path.stat();return dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode)


def parsed_bytes(stdout,path):
    row=stdout.strip().split('\t')
    if len(row)!=2 or row[1]!=str(path) or not row[0].isdecimal():raise ValueError('PREFIX20_INVALID_DU_PARTITION_STDOUT')
    return int(row[0])


def verify_root(root):
    if (root['role']!='prefix20_prelaunch_personal_storage_observation' or root['account_experiment_root']!=PERSONAL_ROOT or
            root['returncode']!=1 or root['timed_out'] or root['measurement_complete'] or
            root['claim']['command']!=['du','-s','-B1',PERSONAL_ROOT]):
        raise ValueError('PREFIX20_ORIGINAL_ROOT_DU_RECEIPT_DIFFERS')
    for key in ('stdout','stderr'):
        path=Path(root[key+'_file'])
        if sha(path)!=root[key+'_sha256'] or path.read_text()!=root[key]:raise ValueError('PREFIX20_ROOT_DU_BYTES_CHANGED')
    return denied_paths(root['stderr'])


def validate_upper_bound(proof,base):
    if proof.get('measurement_kind')!=KIND:raise ValueError('PREFIX20_UNKNOWN_STORAGE_MEASUREMENT_KIND')
    if proof['aggregator_uid']!=14090 or proof['aggregator_source_sha256']!=sha(Path(__file__)):
        raise ValueError('PREFIX20_STORAGE_AGGREGATOR_SOURCE_OR_UID_DIFFERS')
    path=Path(proof['original_root_receipt_file'])
    if not path.is_relative_to(base/'preparation') or sha(path)!=proof['original_root_receipt_sha256']:
        raise ValueError('PREFIX20_ORIGINAL_ROOT_RECEIPT_SHA_DIFFERS')
    root=json.loads(path.read_text());expected=verify_root(root)
    if time.time()-root['checked_unix']>21600:raise ValueError('PREFIX20_ORIGINAL_ROOT_OBSERVATION_STALE')
    if proof['returncode']!=root['returncode'] or proof['stdout']!=root['stdout'] or proof['stderr']!=root['stderr']:
        raise ValueError('PREFIX20_ORIGINAL_FAILED_DU_WAS_RELABELED')
    parts=proof['permission_subtree_complements']
    if [p['path'] for p in parts]!=expected:raise ValueError('PREFIX20_STORAGE_PARTITION_COVERAGE_DIFFERS')
    total=parsed_bytes(root['stdout'],PERSONAL_ROOT)
    for part in parts:
        if part['returncode']!=0 or part['stderr'] or part['before']!=part['after']:
            raise ValueError('PREFIX20_STORAGE_COMPLEMENT_INCOMPLETE_OR_REPLACED')
        if part['command']!=['du','-s','-B1',part['path']] or part['observer_uid']!=14090:
            raise ValueError('PREFIX20_STORAGE_COMPLEMENT_COMMAND_OR_UID_DIFFERS')
        for key in ('stdout','stderr'):
            target=Path(part[key+'_file'])
            if not target.is_relative_to(base/'preparation') or target.is_symlink() or sha(target)!=part[key+'_sha256']:
                raise ValueError('PREFIX20_STORAGE_COMPLEMENT_FILE_SHA_DIFFERS')
            if target.read_text()!=part[key]:raise ValueError('PREFIX20_STORAGE_COMPLEMENT_BYTES_DIFFERS')
        total+=parsed_bytes(part['stdout'],part['path'])
    if total!=proof['allocated_bytes_upper_bound']:raise ValueError('PREFIX20_STORAGE_UPPER_BOUND_SUM_DIFFERS')
    return total


def complete(base,root_receipt,commit):
    os.umask(0o007);base=own_base(base);root_receipt=Path(root_receipt)
    if os.getuid()!=14090 or not root_receipt.is_relative_to(base/'preparation'):
        raise ValueError('PREFIX20_JXU_PERMISSION_COMPLEMENT_REQUIRED')
    root=json.loads(root_receipt.read_text());paths=verify_root(root)
    directory=base/'preparation'/('storage-complements-'+str(time.time_ns()));directory.mkdir()
    parts=[];total=parsed_bytes(root['stdout'],PERSONAL_ROOT)
    for ordinal,name in enumerate(paths):
        path=Path(name);before=directory_identity(path);stdout=directory/(str(ordinal)+'.stdout');stderr=directory/(str(ordinal)+'.stderr')
        started=time.time()
        with stdout.open('xb') as out,stderr.open('xb') as error:
            child=subprocess.Popen(['du','-s','-B1',name],stdout=out,stderr=error)
            child_identity=identity(child.pid)
            publish(directory/(str(ordinal)+'.claim.json'),dict(path=name,child=child_identity,started_unix=started))
            try:code=child.wait(timeout=180)
            except subprocess.TimeoutExpired:child.kill();child.wait();raise ValueError('PREFIX20_STORAGE_COMPLEMENT_TIMED_OUT') from None
            out.flush();error.flush();os.fsync(out.fileno());os.fsync(error.fileno())
        part=dict(path=name,command=['du','-s','-B1',name],observer_uid=os.getuid(),hostname=socket.gethostname(),child=child_identity,
            before=before,after=directory_identity(path),returncode=code,stdout=stdout.read_text(),stderr=stderr.read_text(),
            stdout_file=str(stdout),stderr_file=str(stderr),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),
            started_unix=started,checked_unix=time.time())
        publish(directory/(str(ordinal)+'.result.json'),part)
        if code or part['stderr'] or part['before']!=part['after']:raise ValueError('PREFIX20_STORAGE_COMPLEMENT_FAILED_OR_REPLACED')
        total+=parsed_bytes(part['stdout'],name);parts.append(part)
        atomic(directory/'progress.json',dict(completed_parts=len(parts),total_parts=len(paths),observed_upper_bound_so_far=total,checked_unix=time.time()))
    proof=dict(root,measurement_kind=KIND,measurement_complete=True,allocated_bytes_upper_bound=total,
        original_root_receipt_file=str(root_receipt),original_root_receipt_sha256=sha(root_receipt),
        original_root_checked_unix=root['checked_unix'],aggregator_uid=os.getuid(),aggregator_source_sha256=sha(Path(__file__)),
        aggregate_source_commit=commit,permission_subtree_complements=parts,checked_unix=time.time(),
        semantics='complete permission partition coverage; allocated-byte conservative upper bound, not a successful original root du or future hard quota guarantee')
    validate_upper_bound(proof,base);publish(directory/'result.json',proof)
    atomic(base/'preparation/personal-storage-observation.json',proof)
    print(json.dumps(dict(result_file=str(directory/'result.json'),allocated_bytes_upper_bound=total,completed_complements=len(parts),checked_unix=proof['checked_unix'])),flush=True)
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--root-receipt',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    complete(a.base,a.root_receipt,a.commit)


if __name__=='__main__':main()

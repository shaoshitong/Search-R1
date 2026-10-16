"""Exact-file retirement of the obsolete sigma3 checkpoint copies only.

The human authorized old failed experiments and checkpoints after native51713.
Keep the full evaluation, raw trajectories, teacher data, source and receipts.
This operation deliberately ends the old sigma3 checkpoint recovery chain.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4.control import own_base, publish

OLD = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015')
TRANSIT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/archives')
RUN = '70827e85fe584f0c84453aae907918a4'
SUFFIXES = {'.pt', '.pth', '.safetensors', '.bin', '.pkl', '.npy', '.npz'}


def targets():
    result = [OLD / 'checkpoints/search-g1-common8/actor' / ('global_step_' + str(s))
              for s in (5, 50, 100, 150, 200)]
    for step in (5, 50, 100, 150):
        for kind in ('archive', 'target'):
            parent = TRANSIT / RUN / ('global_step_' + str(step)) / kind / 'checkpoints'
            if parent.exists():
                result.append(parent)
    return result


def identity(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('RETIRE_SYMLINK_FORBIDDEN')
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, uid=s.st_uid,
                gid=s.st_gid, mode=s.st_mode, mtime_ns=s.st_mtime_ns, links=s.st_nlink,
                allocated_bytes=s.st_blocks * 512)


def final_proof():
    path = OLD / 'logs/final-identity.json'
    d = json.loads(path.read_text())
    counts = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
                  **{'2wikimultihopqa':12576, 'musique':2417, 'bamboogle':125})
    if (d['run_id'] != RUN or d['step'] != 200 or d['split'] != 'final' or
            d['final'] is not True or d['counts'] != counts or
            any(len(d[k]) != 51713 for k in ('question_ids', 'data_sources', 'em', 'f1')) or
            len(set(d['question_ids'])) != 51713 or
            sum(d['em']) != d['correct'] or abs(d['weighted_em'] - sum(d['em'])/51713) > 1e-12):
        raise ValueError('RETIRE_NATIVE51713_FINAL_IDENTITY_INVALID')
    from collections import Counter
    if Counter(d['data_sources']) != counts or any(x not in (0, 1) for x in d['em']):
        raise ValueError('RETIRE_FINAL_POPULATION_OR_EM_INVALID')
    manifest = json.loads((OLD / 'logs/run-manifest.json').read_text())
    pair = OLD / 'artifacts/question-probe/completed-pairs/global_step_200/source-receipt.json'
    p = json.loads(pair.read_text())
    if manifest['run_id'] != RUN or p['run_id'] != RUN or p['step'] != 200:
        raise ValueError('RETIRE_FINAL_CHECKPOINT_RUN_BINDING_DIFFERS')
    return dict(path=str(path), sha256=sha(path), run_id=RUN, step=200,
                native_rows=51713, correct=d['correct'], weighted_em=d['weighted_em'],
                counts=counts, validation_sha256=d['validation_sha256'],
                source_manifest=str(OLD/'logs/run-manifest.json'),
                source_manifest_sha256=sha(OLD/'logs/run-manifest.json'),
                pair_receipt=str(pair), pair_sha256=sha(pair))


def no_active_jobs():
    args = ['squeue', '-h', '-u', 'jxu957,mou176', '-o', '%i|%u|%T|%j|%N']
    p = subprocess.run(args, capture_output=True, text=True, timeout=20)
    if p.returncode or p.stdout.strip():
        raise ValueError('RETIRE_OWNER_SLURM_QUEUE_NOT_EMPTY')
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def inventory():
    files, roots = [], []
    for root in targets():
        if not root.is_dir():
            raise ValueError('RETIRE_EXPECTED_CHECKPOINT_ROOT_MISSING')
        roots.append(dict(path=str(root), identity=identity(root)))
        for directory, dirs, names in os.walk(root, followlinks=False):
            for name in dirs:
                identity(Path(directory) / name)
            for name in names:
                path = Path(directory) / name
                rec = identity(path)
                if not stat.S_ISREG(rec['mode']):
                    raise ValueError('RETIRE_NONREGULAR_CHECKPOINT_MEMBER')
                if path.suffix not in SUFFIXES:
                    continue
                if rec['uid'] not in (14090, 21930):
                    raise ValueError('RETIRE_CHECKPOINT_UNEXPECTED_OWNER')
                files.append(dict(path=str(path), identity=rec))
    if not files:
        raise ValueError('RETIRE_NO_OLD_CHECKPOINT_TENSORS')
    return roots, sorted(files, key=lambda x:x['path'])


def guard(base, proof_path):
    proof_path = Path(proof_path)
    if not proof_path.is_relative_to(base/'preparation'):
        raise ValueError('RETIRE_QUIESCENCE_PROOF_OUTSIDE_OWN_PREPARATION')
    d = json.loads(proof_path.read_text())
    if (d['scope'] != 'obsolete_sigma3_checkpoint_retirement' or
            time.time()-d['checked_unix'] > 900 or d['checked_unix'] > time.time()+30 or
            not d['jxulogin_no_checkpoint_reader'] or not d['moulogin_no_checkpoint_reader'] or
            not d['hkust3_no_qprobe_process'] or not d['actual_compute_ledger_births_absent']):
        raise ValueError('RETIRE_FRESH_CROSS_OWNER_QUIESCENCE_REQUIRED')
    return dict(path=str(proof_path), sha256=sha(proof_path), **d)


def plan(base, proof_path):
    base = own_base(base)
    queue = no_active_jobs(); final = final_proof(); quiescent = guard(base, proof_path)
    roots, files = inventory()
    value = dict(scope='obsolete_sigma3_checkpoint_retirement', user_authorized=True,
        authorization='2026-10-16 human: old failed/previous experiments checkpoints may be deleted; all native51713-evaluated checkpoints may be deleted',
        final=final, quiescence=quiescent, queue=queue, roots=roots, files=files,
        logical_bytes=sum(x['identity']['bytes'] for x in files),
        allocated_file_bytes=sum(x['identity']['allocated_bytes'] for x in files),
        preserved='all non-tensor checkpoint metadata; all evaluations/logs/raw trajectories/teacher/source/model/cache/retrieval',
        old_recovery_chain_intentionally_retired=True, checked_unix=time.time())
    path=base/'preparation'/('old-checkpoint-retirement-plan-'+str(time.time_ns())+'.json')
    publish(path,value)
    return dict(plan_file=str(path), plan_sha256=sha(path), files=len(files),
                logical_bytes=value['logical_bytes'], allocated_file_bytes=value['allocated_file_bytes'])


def retire(base, plan_path, expected_sha, proof_path):
    base=own_base(base);plan_path=Path(plan_path)
    if (not plan_path.is_relative_to(base/'preparation') or sha(plan_path)!=expected_sha or
            os.getuid()!=14090):
        raise ValueError('RETIRE_EXACT_APPROVED_PLAN_REQUIRED')
    with (base/'preparation/old-checkpoint-retirement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        d=json.loads(plan_path.read_text());guard(base,proof_path);no_active_jobs()
        if d['final']!=final_proof():raise ValueError('RETIRE_FINAL_PROOF_CHANGED')
        roots,files=inventory()
        if roots!=d['roots'] or files!=d['files']:
            raise ValueError('RETIRE_EXACT_CHECKPOINT_INVENTORY_CHANGED')
        claim=plan_path.with_suffix('.delete-claim.json')
        publish(claim,dict(plan_sha256=expected_sha,pid=os.getpid(),started_unix=time.time()))
        journal=plan_path.with_suffix('.deleted.jsonl')
        with journal.open('x') as out:
            for row in files:
                path=Path(row['path'])
                if identity(path)!=row['identity']:
                    raise ValueError('RETIRE_FILE_CHANGED_BEFORE_UNLINK')
                path.unlink()
                out.write(json.dumps(row)+'\n');out.flush();os.fsync(out.fileno())
        if any(Path(x['path']).exists() for x in files):
            raise ValueError('RETIRE_UNLINK_READBACK_FAILED')
        for root in roots:
            fd=os.open(root['path'],os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        value=dict(status='OLD_SIGMA3_EXACT_CHECKPOINT_TENSORS_RETIRED',
            plan_file=str(plan_path),plan_sha256=expected_sha,deleted_files=len(files),
            deleted_logical_bytes=d['logical_bytes'],deleted_allocated_file_bytes=d['allocated_file_bytes'],
            journal=str(journal),journal_sha256=sha(journal),all_deleted_paths_absent=True,
            retained_final=final_proof(),checked_unix=time.time())
        publish(plan_path.with_suffix('.deleted-complete.json'),value)
        return value


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('plan','retire'))
    p.add_argument('--base',required=True);p.add_argument('--quiescence',required=True)
    p.add_argument('--plan');p.add_argument('--plan-sha');a=p.parse_args()
    value=plan(a.base,a.quiescence) if a.mode=='plan' else retire(a.base,a.plan,a.plan_sha,a.quiescence)
    print(json.dumps(value))


if __name__=='__main__':main()

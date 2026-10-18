#!/usr/bin/env python3
"""Read one progressive three-epoch P10 probe's full SHA/model/Adam/four RNG on HPC3 CPU.

Run only on HPC3 with this experiment's env/bin/python. This does not validate
the actor payload or replace milestone QA/full-state/recovery qualification.
Training remains live; immutable successful proofs use stat seals on reuse.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
import traceback

BASE = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
RUNTIME = '89c97ff2a5a60c7c3d6e1b0bf825d122f0ff874c'
RUN_ID = 'cc0de51f2d68495c9664aa888074839c'
PASS = 'PERIODIC_PROBE_FULL_SHA_CPU_READBACK_PASS'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def publish(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def memory_headroom():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, group = line.split(':', 2)
        if 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory') / group.lstrip('/')
            maximum, current = root / 'memory.limit_in_bytes', root / 'memory.usage_in_bytes'
        elif not controllers:
            root = Path('/sys/fs/cgroup') / group.lstrip('/')
            maximum, current = root / 'memory.max', root / 'memory.current'
        else:
            continue
        if maximum.exists() and current.exists():
            limit = maximum.read_text().strip()
            if limit == 'max':
                continue
            limit = int(limit); used = int(current.read_text())
            if limit < 2 ** 60:
                return dict(limit_bytes=limit,used_bytes=used,headroom_bytes=limit-used,cgroup=str(root))
    raise RuntimeError('Cannot establish the real execution cgroup memory bound')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=range(60, 200, 10), required=True)
    parser.add_argument('--attempt', default='v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9-]{1,32}', args.attempt):
        raise ValueError('Require a bounded explicit audit attempt name')
    assert BASE.resolve(strict=True) == BASE
    joint_path = BASE / 'outers' / ('outer-%06d' % args.outer) / 'complete.json'
    if not joint_path.is_file():
        print(json.dumps(dict(status='PENDING_NO_COMMITTED_P10_JOINT',outer=args.outer)))
        return
    proof = BASE / 'preparation' / ('live-probe-%06d-%s-full-sha-cpu.json' % (args.outer,args.attempt))
    progress = proof.with_name(proof.name + '.progress.json')
    intent = proof.with_name(proof.name + '.intent.json')
    with proof.with_name(proof.name + '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='EXISTING_PROBE_READBACK_RUNNING',progress=str(progress))))
            return
        script_sha,joint_sha = sha(__file__),sha(joint_path)
        if proof.exists():
            old=json.loads(proof.read_text())
            assert old['status']==PASS and old['source_joint_sha256']==joint_sha
            assert old['operator_script_sha256']==script_sha
            for row in old['verified_files']:
                stat=Path(row['path']).stat()
                assert (stat.st_ino,stat.st_size,stat.st_mtime_ns,int(os.statvfs(row['path']).f_fsid)) == (
                    row['inode'],row['bytes'],row['mtime_ns'],row['shared_filesystem_fsid'])
            print(json.dumps(dict(status='EXISTING_SUCCESSFUL_PROBE_PROOF_STAT_SEALS_REUSED',proof=str(proof),sha256=sha(proof))))
            return
        if intent.exists():
            raise RuntimeError('Inspect interrupted audit evidence before using an explicit new attempt')
        memory=memory_headroom()
        if memory['headroom_bytes'] < 3*1024**3:
            print(json.dumps(dict(status='PENDING_LOGIN_MEMORY_HEADROOM',memory=memory)))
            return
        context=dict(pid=os.getpid(),host=socket.gethostname(),started_unix=time.time(),outer=args.outer,
                     base=str(BASE),source_joint_sha256=joint_sha,operator_script_sha256=script_sha,memory=memory)
        publish(intent,dict(context,status='READ_ONLY_PROBE_CPU_AUDIT_STARTED'))
        try:
            publish(progress,dict(context,stage='IMPORT_EXISTING_P10_VALIDATOR',checked_unix=time.time()))
            os.environ['CUDA_VISIBLE_DEVICES']=''
            sys.path.insert(0,str(BASE/'repo'))
            from reproduction.prefix_selection200.progressive_prefix import MODE,after_boundary,before_fit,boundary
            os.environ['DYNRPO_PROGRESSIVE_PREFIX']=MODE
            os.environ['DYNRPO_PREFIX_TOKENS']=str(after_boundary(args.outer))
            assert json.loads((BASE/'preparation/source-publication.json').read_text())['commit']==RUNTIME
            import torch
            torch.set_num_threads(2)
            from reproduction.prefix_selection200.checkpoint_native_pending_probe import Seals
            from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import probe_state
            joint=json.loads(joint_path.read_text())
            config=json.loads((BASE/'preparation/launch-config.json').read_text())
            assert joint['outer']==args.outer and joint['periodic_probe_update']
            assert joint['run_id']==config['run_id']==RUN_ID and joint['runtime_source_commit']==RUNTIME
            metrics=joint['probe_metrics']
            assert metrics['epochs']==3 and metrics['new_adam_steps_per_rank']==150
            before=before_fit(args.outer)['actual_adam_per_rank']
            expected=boundary(args.outer)['actual_adam_per_rank']
            epoch_root=joint_path.parent/'probe-update'
            epoch_proofs=[json.loads((epoch_root/('epoch-%03d.json'%epoch)).read_text()) for epoch in (1,2,3)]
            assert [r['actual_adam_steps_per_rank'] for r in epoch_proofs]==[[before+50*epoch]*4 for epoch in (1,2,3)]
            assert joint['probe']['actual_adam_steps_per_rank']==[expected]*4
            snapshot=Path(joint['probe_snapshot'])
            seals=Seals(BASE,allowed=[snapshot])
            seals.file(joint_path);seals.json(BASE/'preparation/launch-config.json')
            publish(progress,dict(context,stage='FULL_PROBE_SHA_AND_CPU_MMAP',checked_unix=time.time()))
            head,file=seals.mmap_torch(snapshot,joint['probe'])
            result=probe_state(head,joint['probe'],config,args.outer,{args.outer:joint})
            del head;gc.collect()
            files=seals.finish()
            for row in files:row['shared_filesystem_fsid']=int(os.statvfs(row['path']).f_fsid)
            assert sha(joint_path)==joint_sha
            payload=dict(context,status=PASS,completed_unix=time.time(),runtime_commit=RUNTIME,
                         probe=dict(result,file=file),verified_files=files,
                         prefix_tokens=after_boundary(args.outer),epochs=3,new_probe_adam_per_rank=150,
                         epoch_boundary_steps=[before+50*epoch for epoch in (1,2,3)],
                         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         actor_payload_readback_performed=False,stopped_writer_recovery_proof=False,
                         cuda_initialized=torch.cuda.is_initialized(),model_forward_performed=False)
            assert not payload['cuda_initialized']
            publish(proof,payload)
            publish(progress,dict(context,status=PASS,stage='COMPLETE',checked_unix=time.time(),proof=str(proof)))
            print(json.dumps(dict(status=PASS,proof=str(proof),sha256=sha(proof),
                                 actual_adam_steps_per_rank=result['actual_adam_steps_per_rank'],
                                 peak_rss_bytes=payload['peak_rss_bytes'])))
        except BaseException as error:
            publish(proof.with_name(proof.name+'.failure.json'),dict(context,error_type=type(error).__name__,
                    error=str(error),traceback=traceback.format_exc(),checked_unix=time.time()))
            raise


if __name__=='__main__':main()

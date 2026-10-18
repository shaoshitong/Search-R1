#!/usr/bin/env python3
"""CPU/full-SHA readback of one immutable milestone while training continues.

Run on HPC3 with the existing remote environment, never with Mac model data.
This produces a snapshot-readback proof, NOT a stopped-writer recovery proof.
It uses the existing four-rank/P10 validators without their terminal barrier.
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

FAMILY = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
VALIDATOR_BASE = Path(str(FAMILY) + '-r3')
VALIDATOR_COMMIT = None  # Exact pinned source-publication at this isolated execution
RUN_ID = 'cc0de51f2d68495c9664aa888074839c'
PASS = 'LIVE_IMMUTABLE_MILESTONE_FULL_SHA_CPU_READBACK_PASS'


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=str(VALIDATOR_BASE))
    parser.add_argument('--outer', type=int, choices=(5, 50, 100, 150, 200), required=True)
    parser.add_argument('--attempt', default='v1')
    args = parser.parse_args()
    base = Path(args.base)
    if (base.parent != FAMILY.parent or not re.fullmatch(re.escape(FAMILY.name) + r'(?:-r[0-9]+)?', base.name)
            or base.resolve(strict=True) != base or not re.fullmatch(r'[a-z0-9-]{1,32}', args.attempt)):
        raise ValueError('Require this exact experiment family and a bounded audit attempt name')
    joint_path = base / 'outers' / ('outer-%06d' % args.outer) / 'complete.json'
    if not joint_path.is_file():
        print(json.dumps(dict(status='PENDING_NO_COMMITTED_MILESTONE', outer=args.outer)))
        return
    proof = base / 'preparation' / ('live-milestone-%06d-%s-full-sha-cpu.json' % (args.outer, args.attempt))
    progress = proof.with_name(proof.name + '.progress.json')
    intent = proof.with_name(proof.name + '.intent.json')
    lock = proof.with_name(proof.name + '.lock')
    script_sha = sha(__file__)
    joint_sha = sha(joint_path)
    with lock.open('a+') as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='EXISTING_READBACK_RUNNING', progress=str(progress))))
            return
        if proof.exists():
            old = json.loads(proof.read_text())
            assert old['status'] == PASS and old['source_joint_sha256'] == joint_sha
            assert old['operator_script_sha256'] == script_sha and old['base'] == str(base)
            for row in old['verified_files']:
                stat = Path(row['path']).stat()
                assert (stat.st_ino, stat.st_size, stat.st_mtime_ns, int(os.statvfs(row['path']).f_fsid)) == (
                    row['inode'], row['bytes'], row['mtime_ns'], row['shared_filesystem_fsid'])
            print(json.dumps(dict(status='EXISTING_SUCCESSFUL_PROOF_STAT_SEALS_REUSED', proof=str(proof), sha256=sha(proof))))
            return
        if intent.exists():
            raise RuntimeError('Inspect interrupted audit evidence; use an explicit new attempt after fixing its cause')
        context = dict(pid=os.getpid(), host=socket.gethostname(), started_unix=time.time(), base=str(base),
                       outer=args.outer, operator_script_sha256=script_sha, source_joint_sha256=joint_sha)
        publish(intent, dict(context, status='READ_ONLY_MILESTONE_AUDIT_STARTED'))

        def stage(name, **extra):
            publish(progress, dict(context, checked_unix=time.time(), stage=name, **extra))

        try:
            stage('IMPORT_EXISTING_FOUR_RANK_VALIDATORS')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            sys.path.insert(0, str(VALIDATOR_BASE / 'repo'))
            pub = json.loads((VALIDATOR_BASE / 'preparation/source-publication.json').read_text())
            assert pub['commit']==pub['remote_sha']
            from reproduction.prefix_selection200.progressive_prefix import MODE,after_boundary
            os.environ['DYNRPO_PROGRESSIVE_PREFIX']=MODE
            os.environ['DYNRPO_PREFIX_TOKENS']=str(after_boundary(args.outer))
            import torch
            torch.set_num_threads(2)
            from reproduction.prefix_selection200.checkpoint_native_pending_probe import Seals, _hf
            from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import actor_state, actor_ledger, probe_state, selection_probe_state
            from reproduction.prefix_selection200.i1_i6_joint_resume import _joint
            from reproduction.prefix_selection200.checkpoint_cpu_once import validate_rng
            from reproduction.prefix_selection200.candidate_schedule import consumed_candidates_through
            from reproduction.prefix20_online.pool import canonical as pool_canonical

            config = json.loads((base / 'preparation/launch-config.json').read_text())
            joint = json.loads(joint_path.read_text()); actor = joint['actor']; state = joint['grounding']
            assert config['run_id'] == joint['run_id'] == RUN_ID and joint['outer'] == args.outer
            # The validators use only the prior actor counter and the last
            # probe-update's batch binding. Do not retain200 recursively embedded
            # grounding journals alongside the actual multi-GB CPU tensors.
            history = {}
            for n in {args.outer - 1, joint['probe']['completed_outer']} - {0}:
                item = joint if n == args.outer else json.loads(
                    (base / 'outers' / ('outer-%06d' % n) / 'complete.json').read_text())
                history[n] = dict(actor=dict(actual_adam_steps_per_rank=item['actor']['actual_adam_steps_per_rank']),
                    probe=item['probe'], rollout=dict(actor_version=item['rollout']['actor_version']),
                    probe_metrics={key: item['probe_metrics'].get(key) for key in
                                   ('scored_features_manifest_sha256', 'own_em_sha256', 'actor_version')})
                del item
            _joint(config, joint, args.outer, previous_actual=history[args.outer - 1]['actor']['actual_adam_steps_per_rank'])
            assert len(actor['rank_states']) == len(actor['rank_receipts']) == 4
            allowed = [Path(config['official_model_dir']) / 'config.json', Path(actor['hf_model_dir']),
                       Path(actor['driver_state']['path']), Path(joint['probe_snapshot']),
                       Path(joint['pool_snapshot']), Path(joint['grounding_snapshot'])]
            allowed += [Path(actor['hf_model_dir']) / name for name in actor['hf_files']]
            allowed += [Path(row['path']) for row in actor['rank_states']]
            allowed += [Path(row['ledger_path']) for row in actor['rank_receipts']]
            allowed += [Path(state[key]) for key in ('manifest_path', 'commit', 'boundary', 'adoption_receipt')]
            seals = Seals(base, allowed=allowed)
            seals.json(base / 'preparation/launch-config.json'); seals.file(joint_path)
            stage('FULL_SHA_AND_FINITE_FP32_HF_TENSORS')
            hf = _hf(seals, actor, config)
            ranks = []
            for rank, producer in enumerate(actor['rank_states']):
                stage('FULL_SHA_AND_CPU_ACTOR_RANK', rank=rank, bytes_verified=sum(x['bytes'] for x in seals.files.values()))
                saved, file = seals.mmap_torch(producer['path'], producer)
                ranks.append(dict(actor_state(saved, actor, config, args.outer, rank), file=file,
                                  ledger=actor_ledger(seals, actor, args.outer, rank)))
                del saved; gc.collect()
            pool = seals.json(joint['pool_snapshot']); size = len(pool['order'])
            consumed = consumed_candidates_through(config, args.outer)
            assert pool['next_outer'] == args.outer + 1 and pool['consumed_candidates'] == consumed
            assert pool['cycle'] * size + pool['cursor'] == consumed and sorted(pool['order']) == list(range(size))
            assert sha(joint['pool_snapshot']) == joint['pool_sha256']
            stage('CPU_DRIVER_RNG_POOL_AND_PRE_UPDATE_SELECTION_HEAD')
            driver, driver_file = seals.mmap_torch(actor['driver_state']['path'], actor['driver_state'])
            assert driver['run_id'] == RUN_ID and driver['completed_outer'] == args.outer
            assert driver['actor_version'] == actor['actor_version'] and driver['grounding_state'] == state
            assert driver['grounding_probe_version'] == state['version'] and driver['candidate_pool_state']['value'] == pool
            assert driver['candidate_pool_state']['sha256'] == hashlib.sha256(pool_canonical(pool) + b'\n').hexdigest()
            manifest = seals.json(joint_path.parent / 'probe-score/manifest.json')
            selection_probe_state(driver['prefix_probe_receipt'], manifest, config, args.outer)
            driver_rng = validate_rng(driver['rng']); del driver; gc.collect()
            stage('FULL_SHA_AND_CPU_PROBE_MODEL_ADAM_FOUR_RNG')
            head, head_file = seals.mmap_torch(joint['probe_snapshot'], joint['probe'])
            probe = dict(probe_state(head, joint['probe'], config, args.outer, history), file=head_file)
            del head; gc.collect()
            assert seals.json(joint['grounding_snapshot']) == state
            seals.json(state['manifest_path']); seals.json(state['adoption_receipt'])
            for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
                path = Path(state[key]); seals.file(path, dict(bytes=path.stat().st_size, sha256=state['head_sha256'][name]))
            files = seals.finish()
            for row in files:
                row['shared_filesystem_fsid'] = int(os.statvfs(row['path']).f_fsid)
            assert sha(joint_path) == joint_sha
            result = dict(context, status=PASS, finished_unix=time.time(), validator_runtime_commit=pub['commit'],
                run_id=RUN_ID, source_config_sha256=sha(base / 'preparation/launch-config.json'),
                actor_hf=hf, actor_rank_states=ranks, actor_driver_file=driver_file, actor_driver_rng=driver_rng,
                actor_actual_adam_steps_per_rank=actor['actual_adam_steps_per_rank'], scheduler_completed_outer=args.outer,
                probe=probe, pool_next_outer=pool['next_outer'], pool_consumed_candidates=consumed,
                actor_used_grounding_round=state['round'], verified_files=files,
                maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                actual_cpu_deserialization=True, gpu_forward_performed=False, cuda_rng_applied=False,
                training_stopped=False, stopped_writer_recovery_qualification=False,
                due_post_outer_grounding_refit_requires_separate_acceptance=True)
            publish(proof, result); stage('PASS', proof=str(proof), sha256=sha(proof))
            print(json.dumps(dict(status=PASS, proof=str(proof), sha256=sha(proof), outer=args.outer,
                                 verified_files=len(files), actor_adam=actor['actual_adam_steps_per_rank'],
                                 maximum_resident_bytes=result['maximum_resident_bytes'])))
        except BaseException as error:
            failed = dict(context, checked_unix=time.time(), status='READBACK_FAILED_NOT_TRAINING_FAILURE',
                          error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc(limit=12))
            publish(proof.with_name(proof.name + '.failure.json'), failed); stage('FAILED', error=failed['error'])
            raise


if __name__ == '__main__':
    main()

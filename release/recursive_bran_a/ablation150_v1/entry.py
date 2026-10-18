"""Fresh independent H100 run; or explicit complete native state of that same group."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

from .contracts import read_receipt, sha


def execute(args):
    receipt_path = Path(args.receipt).resolve()
    record = read_receipt(receipt_path)
    base = Path(record['base'])
    from search_r1.baselines.execution_identity import shared_h100, execution_id
    shared = shared_h100()
    if not shared and (not os.environ.get('SLURM_JOB_ID') or os.environ['SLURM_JOB_ID'] != os.environ.get('RBA_H100_ALLOCATION_ID')):
        raise ValueError('Inherited task-owned H100 allocation is required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('All eight authorized physical cards are required')
    previous = base / 'receipts' / 'current-entry.json'
    retry_proof = None
    zero_write_manifest = getattr(args, 'zero_write_manifest', None)
    zero_write_sha = getattr(args, 'zero_write_sha256', None)
    if zero_write_manifest:
        if not zero_write_sha or args.resume_manifest or args.retry_unstarted:
            raise ValueError('Zero-write batch recovery requires only its explicit failed manifest and SHA')
    elif zero_write_sha:
        raise ValueError('No zero-write SHA without its failed manifest')
    if previous.exists():
        old = json.loads(previous.read_text())
        proc = Path('/proc') / str(old['pid'])
        if proc.exists():
            stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[19]) == old['tick'] and proc.stat().st_uid == old['uid'] and stat[0] != 'Z':
                raise ValueError('This group still has a live owner; do not submit a duplicate')
        if args.retry_unstarted:
            if args.resume_manifest:
                raise ValueError('Pre-initialization retry cannot also claim native resume')
            from .unstarted_retry import check
            retry_proof = check(base, old)
        elif not args.resume_manifest and not zero_write_manifest:
            raise ValueError('Existing group needs an explicit native checkpoint; do not reset Adam')
    elif args.retry_unstarted:
        raise ValueError('No previous exact failed entry exists to retry')
    output = base / 'output' / ('attempt-' + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(RECURSIVE_ABLATION_RECEIPT=str(receipt_path),
        SEARCH_R1_BASELINE_RUN_ID=record['run_id'], E1_RUN_ID=record['run_id'],
        SEARCH_R1_EXECUTION_BACKEND=('h100_shared' if shared else 'slurm_h100'),
        SEARCH_R1_BASELINE_COMMIT=record['source_commit'],
        SEARCH_R1_BASELINE_CONFIG_SHA256=record['config_sha256'],
        SEARCH_R1_BASELINE_DATA_IDENTITY=record['data_identity'],
        SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
        SEARCH_R1_BASELINE_OUTPUT=str(output), SEARCH_R1_BASELINE_AUDIT_DIR=str(output / 'audit'),
        SEARCH_R1_TREE_RANK_AUDIT_DIR=str(output / 'rank-audit'),
        SEARCH_R1_TRAJECTORY_DIR=str(output / 'trajectories'),
        TREE_COMMON_DATA=record['data'], TREE_INSTRUCT_MODEL=record['model'],
        RECURSIVE_A_H100_FRESH_PROFILE='1', RECURSIVE_A_OMIT_FULL_BETA='1',
        RECURSIVE_A_GUARD_ROUND_PROGRESS='1', TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',
        RECURSIVE_A_SPARSE_DETAILS='1', PYTHONDONTWRITEBYTECODE='1')
    if zero_write_manifest:
        os.environ.update(RECURSIVE_ABLATION_ZERO_WRITE_MANIFEST=str(Path(zero_write_manifest).resolve()),
                          RECURSIVE_ABLATION_ZERO_WRITE_SHA256=zero_write_sha)
    if args.resume_manifest:
        if not args.resume_sha256:
            raise ValueError('Explicit resume requires exact manifest SHA')
        os.environ.update(RECURSIVE_ABLATION_RESUME_MANIFEST=str(Path(args.resume_manifest).resolve()),
                          RECURSIVE_ABLATION_RESUME_SHA256=args.resume_sha256)
    elif args.resume_sha256:
        raise ValueError('No resume SHA without a manifest')
    for name in tuple(os.environ):
        if name.startswith(('TREE_RECOVERY_', 'TREE_CURRENT_CHAIN_RESUME_', 'RECURSIVE_A_CONTINUATION_')):
            raise ValueError('Historical A/fixed-PID recovery scope must not enter this fresh ablation')
    from .bootstrap import install, worker_setup
    install()
    from .progress import replace_json
    entry = dict(time=time.time(), pid=os.getpid(), uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        run_id=record['run_id'], group=record['group'], output=str(output),
        config_sha256=record['config_sha256'], source_commit=record['source_commit'],
        entry_source_commit=os.environ.get('RBA_ABLATION_ENTRY_COMMIT', record['source_commit']),
        resume_manifest=args.resume_manifest, retry_unstarted_proof=retry_proof,
        zero_write_manifest=zero_write_manifest, zero_write_sha256=zero_write_sha,
        allocation=(None if shared else os.environ['SLURM_JOB_ID']), execution_id=execution_id(), execution_backend=('h100_shared' if shared else 'slurm_h100'))
    replace_json(previous, entry)
    print('ABLATION_ENTRY_START=' + json.dumps(entry, sort_keys=True), flush=True)
    from omegaconf import OmegaConf
    config = OmegaConf.load(record['config'])
    from search_r1.baselines.preflight import validate_gpu_lease
    validate_gpu_lease(config)
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    import ray
    env = {key: value for key, value in os.environ.items() if key.startswith(('RECURSIVE_ABLATION_', 'SEARCH_R1_', 'RECURSIVE_A_', 'TREE_'))}
    for key in ('RBA_H100_SHARED_EXECUTION_ID', 'SLURM_JOB_ID', 'CPATH', 'TRITON_LIBCUDA_PATH', 'LIBRARY_PATH', 'PYTHONPATH'):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(TOKENIZERS_PARALLELISM='false', NCCL_DEBUG='WARN')
    # Ray creates a session/socket suffix. Keep its root short enough for the
    # kernel AF_UNIX107-byte address limit on this H100 runtime.
    ray_root = '/tmp/ra-' + record['run_id'][:10]
    context = ray.init(address='local', num_gpus=8, num_cpus=96, include_dashboard=False,
        object_store_memory=8589934592, _node_ip_address='127.0.0.1', _temp_dir=ray_root,
        runtime_env={'env_vars': env, 'worker_process_setup_hook': worker_setup})
    os.environ['RAY_ADDRESS'] = context.address_info['gcs_address']
    exit_code, error = 0, None
    try:
        from search_r1.baselines.main import main
        main.__wrapped__(config)
    except BaseException as failure:
        exit_code, error = 1, repr(failure)
        raise
    finally:
        ray.shutdown()
        replace_json(output / 'entry-exit.json', dict(time=time.time(), pid=os.getpid(),
            uid=os.getuid(), tick=entry['tick'], run_id=record['run_id'], group=record['group'],
            rc=exit_code, error=error, scope='Actual native main return and this entry exception; parent wait recorded separately'))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--resume-manifest')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--retry-unstarted', action='store_true')
    parser.add_argument('--zero-write-manifest')
    parser.add_argument('--zero-write-sha256')
    raise SystemExit(execute(parser.parse_args()))


if __name__ == '__main__':
    main()

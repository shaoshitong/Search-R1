"""Once-only CPU savepoint and exit audit after replay70 has actually finished.

Run remotely with CUDA_VISIBLE_DEVICES='' and the existing CPU-capable training
environment. This controller neither launches work nor changes the frozen run.
Only a small final summary is printed; complete proofs and metrics stay remote.
"""
import argparse
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time


PREFIX = '/mnt/data/shaoshitong/search-g1-question20-replay70-'
RUNTIME = '5fa565b576811a5ddec7bdc3edcb384bd05f608c'
TOTAL = 70
LR = 3e-6


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def stat_record(info):
    return dict(dev=info.st_dev, inode=info.st_ino, bytes=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def stream_sha(path):
    """Hash the original open inode and prove the path still names that inode."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Expected a regular retained file: ' + str(path))
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError('Original file inode changed during audit: ' + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), **stat_record(before), stat_stable=True)


def once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def progress(out, **value):
    temporary = out/'PROGRESS.json.partial'
    temporary.write_bytes(canonical(dict(observed_utc=utc(), **value)) + b'\n')
    os.replace(temporary, out/'PROGRESS.json')


def expected_steps(iteration):
    return [7100 + 100 * iteration] * 8


def validate_mixed(group, count):
    if group['count'] != count or group['threshold'] != .5:
        raise ValueError('Mixed diagnostic question count or threshold changed')
    confusion = [group[key] for key in ('tp', 'fp', 'fn', 'tn')]
    if any(type(n) is not int or n < 0 for n in confusion) or sum(confusion) != count:
        raise ValueError('Mixed diagnostic confusion matrix is incomplete')
    if group['accuracy'] != (group['tp'] + group['tn']) / count:
        raise ValueError('Mixed diagnostic accuracy differs from actual decisions')
    if group['mixed_count'] != group['tp'] + group['fn']:
        raise ValueError('Mixed diagnostic true labels differ')
    if group['nonmixed_count'] != group['fp'] + group['tn']:
        raise ValueError('Mixed diagnostic nonmixed labels differ')
    return group['accuracy']


def native_state(base, terminal_evidence=None, terminal_evidence_sha256=None):
    from reproduction.common8.native_allocation import fields
    receipt = read(base/'preparation/submit-receipt.json')
    job = str(receipt['job_id'])
    try:
        raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'],
                                      text=True, stderr=subprocess.PIPE, timeout=30)
        source='current_scontrol_record'
    except subprocess.CalledProcessError as error:
        stderr=error.stderr.decode(errors='replace')if isinstance(error.stderr,bytes)else str(error.stderr)
        if (not terminal_evidence or not terminal_evidence_sha256
                or 'invalid job id'not in stderr.lower()):
            raise
        evidence_path=Path(terminal_evidence)
        if stream_sha(evidence_path)['sha256']!=terminal_evidence_sha256:
            raise ValueError('Pinned original native terminal evidence bytes differ')
        evidence=read(evidence_path)
        config=read(base/'preparation/launch-config.json')
        if (evidence['base']!=str(base)or str(evidence['job_id'])!=job
                or evidence['run_id']!=config['run_id']
                or evidence['owned_births_present']!=0
                or evidence['native_cgroup_exists']is not False
                or evidence['owned_gpu_pids']
                or evidence['latest']['replay_iteration']!=70
                or evidence['latest']['actual_adam_steps_per_rank']!=[14100]*8):
            raise ValueError('Pinned original native evidence belongs to another run or incomplete boundary')
        raw=evidence['native_raw']
        source='pinned_original_native_record_after_scontrol_purge'
    native = fields(raw)
    if native['JobId'] != job or native['Command'] != str(base/'preparation/native-launch.sh'):
        raise ValueError('Native job does not belong to this frozen namespace')
    return dict(job_id=job, state=native.get('JobState'), exit_code=native.get('ExitCode'), raw=raw,
                evidence_source=source,terminal_evidence_sha256=terminal_evidence_sha256
                if source!='current_scontrol_record'else None)


def exit_resources(base, config, native):
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    ledgers, all_births, cgroups = [], [], []
    bases = [base, Path(config['resume_replay']['source_base'])]
    for owner in bases:
        ledger_path = owner/'logs/process-ledger.json'
        ledger = read(ledger_path)
        births = [row for job in ledger['jobs'].values() for row in job['ledger'].values()]
        present = birth_states(births)
        if present:
            raise ValueError('Original owned process births remain: ' + str(owner))
        owner_job = str(read(owner/'preparation/submit-receipt.json')['job_id'])
        matches = list(Path('/sys/fs/cgroup').glob('**/job_' + owner_job))
        matches += list(Path('/sys/fs/cgroup').glob('**/job' + owner_job))
        if matches:
            raise ValueError('Original native job cgroup remains: ' + owner_job)
        cgroups.append(dict(job_id=owner_job, native_cgroup_exists=False))
        ledgers.append(dict(base=str(owner), ledger_sha256=stream_sha(ledger_path)['sha256'],
                            owned_birth_count=len(births), owned_births_present=0))
        all_births.extend(births)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=30).split()
    # A numeric PID can have been reused by a foreign task; ownership is its
    # original birth identity, not its number alone.
    gpu_candidates = [row for row in all_births if str(row['pid']) in set(gpu_rows)]
    if birth_states(gpu_candidates):
        raise ValueError('Original owned GPU process births remain')
    cleanup = read(base/'logs/cleanup-status.json')
    if cleanup.get('owned_births_all_gone') is not True or cleanup.get('remaining_owned_births'):
        raise ValueError('Pipeline cleanup receipt is incomplete')
    return dict(native=native, ledgers=ledgers, cgroups=cgroups,
                owned_gpu_process_births_present=0, cleanup=cleanup)


def iteration_evidence(base, source_base, test_sha):
    rows, evidence = [], []
    for iteration in range(1, TOTAL + 1):
        owner = source_base if iteration <= 6 else base
        root = owner/'train/iterations'/('iteration-%04d' % iteration)
        metrics, complete = read(root/'metrics.json'), read(root/'complete.json')
        expected = expected_steps(iteration)
        if (metrics['replay_iteration'] != iteration or complete['replay_iteration'] != iteration
                or metrics['actual_adam_steps_per_rank'] != expected
                or complete['actual_adam_steps_per_rank'] != expected
                or metrics['new_adam_steps_per_rank'] != 100
                or complete['new_adam_steps_per_rank'] != 100
                or metrics['actual_optimizer_learning_rates'] != [LR]
                or metrics['test_manifest_sha256'] != test_sha
                or metrics['train_unique_trajectories'] != 25600
                or complete['status'] != 'QUESTION_PLUS20_REPLAY_ITERATION_COMPLETE'):
            raise ValueError('Incomplete actual replay/Adam/test boundary: ' + str(iteration))
        test_group = metrics.get('test_mixed_group', metrics['mixed_group'])
        test_acc = validate_mixed(test_group, 512)
        if (test_group['mixed_count'], test_group['nonmixed_count']) != (183, 329):
            raise ValueError('Fixed original test G5 labels changed')
        if iteration <= 6:
            if 'train_mixed_group' in metrics or 'train_mixed_group_accuracy' in metrics:
                raise ValueError('Do not invent retrospectively measured first-six train results')
            train_acc = None
        else:
            train_acc = validate_mixed(metrics['train_mixed_group'], 5120)
            if (metrics['train_mixed_group_accuracy'] != train_acc
                    or metrics['test_mixed_group_accuracy'] != test_acc
                    or metrics['mixed_metric_semantics'] !=
                    'complete train and fixed test eval/no_grad passes of the same post-update head'):
                raise ValueError('Paired metrics do not use the same post-update head')
        rank_summary = []
        for rank in range(8):
            path = root/('rank-%d-updates.jsonl' % rank)
            updates = [json.loads(line) for line in path.read_text().splitlines() if line]
            if len(updates) != 100:
                raise ValueError('Each original rank ledger must contain exactly100 writes')
            for step, update in enumerate(updates, 1):
                if (update['iteration_step'] != step or update['replay_iteration'] != iteration
                        or update['actual_adam_steps'] != 7100 + (iteration - 1) * 100 + step
                        or update['local_trajectories'] != 32
                        or update['actual_optimizer_learning_rates'] != [LR]):
                    raise ValueError('Actual original rank Adam write trajectory differs')
            if (root/('rank-%d-error.json' % rank)).exists():
                raise ValueError('Original rank error receipt remains in completed iteration')
            rank_summary.append(dict(rank=rank, writes=100, actual_adam_steps=expected[rank],
                                     ledger_sha256=stream_sha(path)['sha256']))
        bundle_path = owner/'iterations'/('iteration-%04d' % iteration)/'data-bundle.json'
        bundle = read(bundle_path)
        if (bundle['checkpoint']['sha256'] != complete['sha256']
                or bundle['source_data_epoch'] != iteration or bundle['question_max_tokens'] != 273
                or bundle['test_manifest_sha256'] != test_sha):
            raise ValueError('Persisted replay data bundle differs from the committed head')
        rows.append(dict(replay_iteration=iteration, train_mixed_accuracy=train_acc,
                         test_mixed_accuracy=test_acc, train_question_groups=5120 if train_acc is not None else None,
                         test_question_groups=512, actual_adam_steps_per_rank=expected,
                         owner_base=str(owner), test_only_original_first_six=iteration <= 6))
        evidence.append(dict(replay_iteration=iteration, complete_sha256=stream_sha(root/'complete.json')['sha256'],
                             metrics_sha256=stream_sha(root/'metrics.json')['sha256'],
                             data_bundle_sha256=stream_sha(bundle_path)['sha256'], ranks=rank_summary))
    return rows, evidence


def retained_source_inventory(base):
    inventory_path = base/'preparation/replay-sources.json'
    inventory = read(inventory_path)
    sources = {row['epoch']: row for row in inventory['sources'] if row['split'] == 'train'}
    if set(sources) != set(range(1, 71)):
        raise ValueError('All seventy original replay source batches must be retained')
    records = []
    for iteration, source in sorted(sources.items()):
        if 'inputs' in source:
            root = Path(source['inputs'])
            scan = read(root/'scan-complete.json')
            files = []
            for name, producer in scan['files'].items():
                path = root/name
                info = path.stat()
                if info.st_size != producer['bytes']:
                    raise ValueError('Original cached input file size changed: ' + str(path))
                files.append(dict(path=str(path), bytes=info.st_size, producer_sha256=producer['sha256']))
            records.append(dict(source_data_epoch=iteration, kind='cached_inputs', files=files,
                                payload_hash_repeated=False))
        else:
            path = Path(source['rollout_complete'])
            complete = read(path)
            complete_sha = stream_sha(path)['sha256']
            if (complete_sha != source['sha256'] or complete['epoch'] != iteration
                    or complete['questions'] != 5120 or complete['trajectories'] != 25600):
                raise ValueError('Original compact rollout inventory metadata differs')
            questions = Path(complete['questions_path'])
            question_info = questions.stat()
            shards = []
            for producer in complete['shards']:
                item = Path(producer['path'])
                info = item.stat()
                if info.st_size != producer['bytes']:
                    raise ValueError('Required compact original rollout is missing or changed in size')
                shards.append(dict(path=str(item), bytes=info.st_size,
                                   producer_sha256=producer['sha256'], count=producer['count']))
            if len(shards) != 8 or sum(row['count'] for row in shards) != 25600:
                raise ValueError('Incomplete original compact G5 shard inventory')
            records.append(dict(source_data_epoch=iteration, kind='compact_original_rollout',
                complete_sha256=complete_sha, question_path=str(questions), question_bytes=question_info.st_size,
                question_producer_sha256=complete['questions_sha256'], compact_shards=shards,
                payload_hash_repeated=False))
    return dict(inventory_sha256=stream_sha(inventory_path)['sha256'], batches=records,
                original_question_and_compact_payloads_rehashed=False)


def checkpoint_audit(base, iteration, binding, complete):
    import numpy as np
    import torch
    path = base/'train'/('head-replay-%04d.pt' % iteration)
    hashed = stream_sha(path)
    if hashed['sha256'] != complete['sha256'] or hashed['bytes'] != complete['bytes']:
        raise ValueError('Independent immutable checkpoint fullSHA differs from producer')
    before = path.stat()
    if any(getattr(before,'st_'+key) != hashed[field] for key,field in
           (('dev','dev'),('ino','inode'),('size','bytes'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns'))):
        raise ValueError('Immutable checkpoint inode changed between fullSHA and CPU load')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    expected = expected_steps(iteration)
    if (saved['replay_iteration'] != iteration or saved['source_epoch'] != 71
            or saved['source_data_epoch'] != iteration
            or saved['actual_adam_steps_per_rank'] != expected
            or saved['source_adam_steps_per_rank'] != [7100]*8
            or saved['binding'] != binding or saved['binding_sha256'] != hashlib.sha256(canonical(binding)).hexdigest()
            or saved['actual_optimizer_learning_rates'] != [LR]
            or saved['test_manifest_sha256'] != binding['test_manifest_sha256']):
        raise ValueError('CPU-loaded immutable head boundary or scientific binding differs')
    shapes = {
        'block.self_attn.q_proj.weight': (2048,2048), 'block.self_attn.q_proj.bias': (2048,),
        'block.self_attn.k_proj.weight': (256,2048), 'block.self_attn.k_proj.bias': (256,),
        'block.self_attn.v_proj.weight': (256,2048), 'block.self_attn.v_proj.bias': (256,),
        'block.self_attn.o_proj.weight': (2048,2048), 'block.mlp.gate_proj.weight': (11008,2048),
        'block.mlp.up_proj.weight': (11008,2048), 'block.mlp.down_proj.weight': (2048,11008),
        'block.input_layernorm.weight': (2048,), 'block.post_attention_layernorm.weight': (2048,),
        'output.weight': (1,600064), 'output.bias': (1,)}
    model, optimizer = saved['model'], saved['optimizer']
    if set(model) != set(shapes) or len(model) != 14:
        raise ValueError('Actual saved head tensor inventory differs')
    for name, value in model.items():
        if tuple(value.shape) != shapes[name] or value.dtype != torch.float32 or value.device.type != 'cpu':
            raise ValueError('Actual FP32 head tensor shape/dtype differs: ' + name)
    ids = [identity for group in optimizer['param_groups'] for identity in group['params']]
    states = optimizer['state']
    if len(ids) != 14 or len(set(ids)) != 14 or set(ids) != set(states):
        raise ValueError('All fourteen actual Adam states are required')
    if any(group['lr'] != LR for group in optimizer['param_groups']):
        raise ValueError('Actual saved optimizer LR differs')
    for identity, parameter in zip(ids, model.values()):
        state = states[identity]
        if int(state['step']) != expected[0]:
            raise ValueError('Actual Adam moment step differs from the complete replay boundary')
        for key in ('exp_avg','exp_avg_sq'):
            value = state[key]
            if value.shape != parameter.shape or value.dtype != torch.float32 or value.device.type != 'cpu':
                raise ValueError('Actual FP32 Adam moment shape/dtype differs')
    rng = saved['rank_rng']
    if len(rng) != 8 or [row['rank'] for row in rng] != list(range(8)):
        raise ValueError('All eight original rank RNG states are required')
    for row in rng:
        if row['actual_adam_steps'] != expected[0]:
            raise ValueError('Actual saved RNG belongs to another Adam boundary')
        random.Random().setstate(row['python'])
        np.random.RandomState().set_state(row['numpy'])
        torch.Generator(device='cpu').set_state(row['torch_cpu'])
        for key in ('torch_cpu','torch_cuda'):
            value = row[key]
            if value.dtype != torch.uint8 or value.device.type != 'cpu' or value.numel() < 1:
                raise ValueError('Saved CPU/CUDA RNG bytes are incomplete')
    if signature(before) != signature(path.stat()):
        raise ValueError('Immutable checkpoint changed during actual CPU load')
    del saved, model, optimizer, states, rng
    gc.collect()
    return dict(replay_iteration=iteration, **hashed, independent_full_sha_performed=True,
                cpu_load_performed=True, model_fp32_tensors=14, adam_fp32_states=14,
                actual_adam_steps_per_rank=expected, rank_rng_count=8,
                cuda_rng_applied=False, gpu_forward_performed=False)


def run(base, output=None, terminal_evidence=None, terminal_evidence_sha256=None):
    base = Path(base).resolve()
    if not str(base).startswith(PREFIX):
        raise ValueError('Run only against the explicit remote replay namespace')
    sys.path.insert(0,str(base/'repo'))
    native = native_state(base,terminal_evidence,terminal_evidence_sha256)
    latest = read(base/'train/latest.json')
    pipeline = base/'logs/pipeline-complete.json'
    if (native['state'] != 'COMPLETED' or native['exit_code'] != '0:0'
            or latest.get('replay_iteration') != TOTAL or not pipeline.is_file()):
        return dict(status='PENDING_NATIVE_COMPLETE70_EXIT0_NO_AUDIT_STARTED', native=native,
                    latest_replay_iteration=latest.get('replay_iteration'), once_intent_created=False)
    output = Path(output).resolve() if output else base/'preparation/final-replay70-audit-once-v1'
    if base/'preparation' not in output.parents:
        raise ValueError('Final complete evidence must stay in this remote preparation namespace')
    output.mkdir(exist_ok=True)
    with (output/'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path, intent_path = output/'RESULT.json', output/'INTENT.json'
        if result_path.exists():
            result = read(result_path)
            return dict(status='EXISTING_FINAL_AUDIT_RESULT_NO_REPLAY',
                        proof_path=str(result_path), proof_sha256=stream_sha(result_path)['sha256'],
                        summary=result['summary'])
        if intent_path.exists():
            return dict(status='EXISTING_FINAL_AUDIT_INTENT_NO_BLIND_REPLAY', intent_path=str(intent_path))
        once(intent_path, dict(base=str(base), native=native, latest=latest,
                              audit_source_sha256=stream_sha(Path(__file__))['sha256'], created_utc=utc()))
        started = time.monotonic()
        try:
            config = read(base/'preparation/launch-config.json')
            claim = read(base/'train/run-claim.json')
            binding = claim['binding']
            if (config['source_commit'] != RUNTIME or config['source_epoch'] != 71
                    or config['iterations'] != 70 or config['learning_rate'] != LR
                    or config['source_adam_steps_per_rank'] != [7100]*8
                    or config['resume_replay']['completed_iteration'] != 6
                    or binding['question_max_tokens'] != 273 or binding['head_max_tokens'] != 293
                    or binding['question_hidden_passed_to_head'] is not True
                    or binding['source_epoch'] != 71
                    or binding['learning_rate'] != LR or binding['global_batch'] != 256
                    or binding['world_size'] != 8 or binding['accumulation'] != 1
                    or latest['actual_adam_steps_per_rank'] != [14100]*8
                    or read(pipeline)['latest'] != latest):
                raise ValueError('Final replay recipe or complete70 actual boundary differs')
            for owner in (base,Path(config['resume_replay']['source_base'])):
                if (owner/'train/iterations/iteration-0071').exists() or (owner/'iterations/iteration-0071').exists():
                    raise ValueError('Unexpected replay iteration71 exists')
            resources = exit_resources(base,config,native)
            progress(output,stage='NATIVE_EXIT0_AND_OWN_RESOURCES_VERIFIED')
            test_sha = binding['test_manifest_sha256']
            test = Path(config['resume_replay']['fixed_test_features_path'])
            if stream_sha(test/'manifest.json')['sha256'] != test_sha:
                raise ValueError('Fixed diagnostic feature identity changed')
            rows, iteration_proofs = iteration_evidence(base,Path(config['resume_replay']['source_base']),test_sha)
            checkpoints = []
            for iteration in range(10,71,10):
                complete = read(base/'train/iterations'/('iteration-%04d'%iteration)/'complete.json')
                checkpoints.append(checkpoint_audit(base,iteration,binding,complete))
                progress(output,stage='IMMUTABLE_CP_CPU_ACCEPTED',checkpoint_iteration=iteration,
                         verified_checkpoint_count=len(checkpoints))
            latest_path, final_path = base/'train/latest.pt',base/'train/head-replay-0070.pt'
            if (latest['sha256']!=checkpoints[-1]['sha256']
                    or latest['bytes']!=checkpoints[-1]['bytes']):
                raise ValueError('Latest70 producer receipt differs from the accepted immutable CP70')
            if latest_path.samefile(final_path):
                latest_proof = dict(checkpoints[-1],path=str(latest_path),same_original_inode_as_cp70=True)
            else:
                latest_proof = stream_sha(latest_path)
                if latest_proof['sha256'] != latest['sha256']:
                    raise ValueError('Final latest checkpoint bytes differ')
            inventory = retained_source_inventory(base)
            logs = [stream_sha(path)for path in sorted((base/'logs').rglob('*'))
                    if path.is_file()and not path.is_symlink()]
            once(output/'ITERATION-METRICS.json',dict(rows=rows,first_six_train_results_available=False))
            metrics_path=output/'ITERATION-METRICS.json'
            summary=dict(status='FINAL_REPLAY70_CPU_SAVEPOINT_AND_EXIT0_PASS',run_id=config['run_id'],
                base=str(base),native_job_id=native['job_id'],native_state='COMPLETED',native_exit_code='0:0',
                complete_replay_iterations=70,actual_adam_steps_per_rank=[14100]*8,
                new_adam_updates_per_rank=7000,immutable_checkpoint_count=7,
                immutable_checkpoint_iterations=list(range(10,71,10)),
                paired_train_test_iterations=64,original_test_only_iterations=6,
                final_train_mixed_accuracy=rows[-1]['train_mixed_accuracy'],
                final_test_mixed_accuracy=rows[-1]['test_mixed_accuracy'],
                original_log_files_full_sha=len(logs),owned_process_births_present=0,
                owned_gpu_process_births_present=0,native_cgroups_present=0,
                original_source_batches_retained=70,original_trajectory_payloads_rehashed=False,
                gpu_forward_performed=False,backup_performed=False,semantic_reload_claimed=False,
                model_cpu_load_performed=True,cuda_rng_applied=False,large_payload_transferred_to_Mac=False,
                metrics_path=str(metrics_path),metrics_sha256=stream_sha(metrics_path)['sha256'],
                elapsed_seconds=time.monotonic()-started,observed_utc=utc())
            result=dict(summary=summary,native_and_resources=resources,checkpoints=checkpoints,
                        latest_checkpoint=latest_proof,original_terminal_log_inodes=logs,
                        iteration_evidence=iteration_proofs,retained_source_inventory=inventory)
        except BaseException as error:
            result=dict(summary=dict(status='FINAL_REPLAY70_AUDIT_FAILED',base=str(base),
                error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,
                observed_utc=utc(),training_modified=False,large_payload_transferred_to_Mac=False))
        once(result_path,result)
        return dict(status=result['summary']['status'],proof_path=str(result_path),
                    proof_sha256=stream_sha(result_path)['sha256'],summary=result['summary'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--output')
    parser.add_argument('--terminal-evidence')
    parser.add_argument('--terminal-evidence-sha256')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('This final audit must run with CUDA_VISIBLE_DEVICES empty')
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    if bool(args.terminal_evidence)!=bool(args.terminal_evidence_sha256):
        parser.error('Both pinned terminal evidence options are required together')
    result=run(args.base,args.output,args.terminal_evidence,args.terminal_evidence_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    if result['status']=='FINAL_REPLAY70_AUDIT_FAILED':
        raise SystemExit(2)


if __name__=='__main__':
    main()

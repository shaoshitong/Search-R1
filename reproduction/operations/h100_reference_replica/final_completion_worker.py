"""CPU-only final retention and actual owned-exit evidence, never training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def metadata(path, expected=None):
    path = Path(path); value = path.stat()
    need(stat.S_ISREG(value.st_mode), 'Retained payload is not regular: ' + str(path))
    need(expected is None or value.st_size == expected, 'Retained payload size differs: ' + str(path))
    return dict(path=str(path), bytes=value.st_size, dev=value.st_dev,
                inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def inspect(base, run_id, config_sha256):
    base = Path(base); prep = base / 'preparation'
    proof = prep / 'final-completion-retention-owned-release-once.json'
    if proof.exists():
        return dict(status='EXISTING_FINAL_CLOSURE_NO_REPLAY', proof=str(proof), sha256=sha(proof))
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Final closure is CPU-only')
    need(sha(prep / 'launch-config.json') == config_sha256, 'Actual launch config changed')
    config = read(prep / 'launch-config.json')
    cp = read(prep / 'ab-joint-0200-independent-full-sha-cpu.json')
    qa = read(prep / 'final51713-original-output-cpu-once.json')
    terminal_path = prep / 'native-actual-waitpid-exit.json'
    terminal = read(terminal_path); submitted = read(prep / 'submit-receipt.json')
    armed = read(prep / 'native-wait-supervisor-armed.json')
    need(config['run_id'] == run_id == cp['run_id'] == qa['run_id'], 'Final proof run differs')
    need(cp['status'] == 'AB_JOINT_FULL_SHA_CPU_PASS' and cp['outer'] == 200, 'Final state CPU proof missing')
    need(qa['status'] == 'FINAL51713_ORIGINAL_OUTPUT_SCORE_READBACK_PASS'
         and qa['actual_qa_count'] == qa['actual_unique_qids'] == 51713
         and qa['actual_full_ordinal_coverage'], 'Final original-output CPU proof incomplete')
    need(terminal['status'] == 'ACTUAL_OWN_CHILD_WAITPID_EXIT_CAPTURED'
         and terminal['actual_returncode'] == 0 and terminal['actual_native_exit_zero']
         and terminal['native_leader'] == submitted['leader']
         and terminal['native_receipt'] == submitted == armed['native_receipt']
         and submitted['config_sha256'] == config_sha256
         and armed['submitted_child_parent'] == armed['supervisor']['pid']
         and sha(prep / 'native-wait-supervisor.py') == armed['source_sha256'],
         'Actual direct-parent native exit0 proof differs')
    need(qa['final_receipt_sha256'] == sha(base / 'final-evaluation/complete.json')
         and qa['evaluation_config_sha256'] == sha(prep / 'final-evaluation-config.json'),
         'Accepted final-output binding changed')
    sys.path.insert(0, str(base / 'repo'))
    from reproduction.a800.owned_processes import same
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    ledger_path = base / 'logs/process-ledger.json'; ledger = read(ledger_path)
    records = [record for job in ledger['jobs'].values() for record in job.get('ledger', {}).values()]
    pids = {record['pid'] for record in records}; checks = []
    for _ in range(2):
        remaining = birth_states(records)
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True)
        owned_rows = [line for line in rows.splitlines()
                      if line.split(',', 1)[0].strip().isdigit()
                      and int(line.split(',', 1)[0].strip()) in pids]
        check = dict(checked_unix=time.time(), native_same_birth=same(submitted['leader']),
                     supervisor_same_birth=same(armed['supervisor']),
                     remaining_owned_births=remaining, owned_gpu_pid_rows=owned_rows)
        checks.append(check)
        need(not remaining and not owned_rows and not check['native_same_birth']
             and not check['supervisor_same_birth'], 'Current execution still owns processes/GPU contexts')
        time.sleep(.25)
    cleanup_path = base / 'logs/cleanup-status.json'; cleanup = read(cleanup_path)
    need(cleanup['run_id'] == run_id and cleanup.get('cleaned')
         and cleanup.get('owned_births_all_gone'), 'Current native cleanup incomplete')
    intent = proof.with_suffix('.intent.json')
    with intent.open('x') as stream:
        json.dump(dict(run_id=run_id, created_unix=time.time(), cpu_only=True), stream)
    milestones = []; data = []
    for outer in range(1, 201):
        path = base / 'outers' / ('outer-%06d' % outer) / 'complete.json'; joint = read(path)
        need(joint['outer'] == outer and joint['run_id'] == run_id, 'Committed outer identity differs')
        rollout = joint['rollout']
        need(rollout['questions'] == 512 and rollout['trajectories'] == 2560,
             'Retained selected trajectory budget differs')
        files = [metadata(row['path'], row['bytes']) for row in rollout['shards']]
        files.append(metadata(rollout['own_em']['path'], rollout['own_em']['bytes']))
        selection = Path(joint['actor_selection_path'])
        files.append(metadata(selection)); files.append(metadata(selection.parent.parent / 'questions.json'))
        data.append(dict(outer=outer, selected_questions=512, selected_trajectories=2560, files=files))
        if outer in (5, 50, 100, 150, 200):
            actor = joint['actor']; probe = joint['probe']
            need(len(actor['rank_states']) == 4, 'Milestone rank state count differs')
            payloads = [metadata(row['path'], row['bytes']) for row in actor['rank_states']]
            payloads.append(metadata(actor['driver_state']['path'], actor['driver_state']['bytes']))
            payloads.append(metadata(joint['probe_snapshot'], probe['bytes']))
            payloads.extend(metadata(Path(actor['hf_model_dir']) / name, value['bytes'])
                            for name, value in actor['hf_files'].items())
            payloads.extend(metadata(joint[name]) for name in ('pool_snapshot', 'grounding_snapshot'))
            milestones.append(dict(outer=outer, files=payloads,
                actor_actual_adam_per_rank=actor['actual_adam_steps_per_rank'],
                probe_actual_adam_per_rank=probe['actual_adam_steps_per_rank']))
        del joint
    above = [str(path) for path in (base / 'outers').glob('outer-*')
             if path.is_dir() and path.name[6:].isdigit() and int(path.name[6:]) > 200]
    above += [str(path) for path in (base / 'actor/outers').glob('outer-0201-attempt-*')]
    need(not above, 'Unauthorized outer above200 exists')
    result = dict(status='FINAL_RETENTION_AND_ACTUAL_OWNED_EXIT0_PASS', run_id=run_id,
        config_sha256=config_sha256, actual_native_returncode=0,
        native_terminal_sha256=sha(terminal_path), submit_receipt_sha256=sha(prep / 'submit-receipt.json'),
        owned_release_checks=checks, process_ledger_sha256=sha(ledger_path),
        cleanup_status_sha256=sha(cleanup_path),
        final_checkpoint_cpu_proof_sha256=sha(prep / 'ab-joint-0200-independent-full-sha-cpu.json'),
        final_qa_cpu_proof_sha256=sha(prep / 'final51713-original-output-cpu-once.json'),
        complete_outers=200, no_outer201=True, milestones=milestones,
        retained_selected_data=data, retained_selected_question_groups=200 * 512,
        retained_selected_trajectories=200 * 2560,
        historical_other_execution_residue_is_separate=True,
        retained_payload_checks_are_metadata_only=True, repeated_large_tensor_hash_or_load=False,
        no_new_inference_or_gpu_test=True, foreign_processes_modified=False,
        original_models_and_outputs_copied_to_mac=False, completed_unix=time.time())
    with proof.open('x') as stream:
        json.dump(result, stream, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
    return dict(status=result['status'], proof=str(proof), sha256=sha(proof),
                complete_outers=200, retained_selected_question_groups=200 * 512,
                retained_selected_trajectories=200 * 2560)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--run-id', required=True)
    parser.add_argument('--config-sha256', required=True); args = parser.parse_args()
    print(json.dumps(inspect(args.base, args.run_id, args.config_sha256)))

"""Read-only current four-arm observer; never submits, stops, or edits state."""
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
terminal_samples = 0
for sample in range(180):
    control = json.loads((root / 'prefix-only-four-current-control.json').read_text())
    command = ['python3', '-m', 'reproduction.prefix_only_correctness_ab.control', 'observe']
    for key in ('commit', 'source_root', 'target_inputs', 'base', 'run_id'):
        command += ['--' + key.replace('_', '-'), control[key]]
    command += ['--reference-metadata', str(root / 'prefix10-20-a800-reference-source-metadata.json'),
                '--management-receipt', str(root / 'qprobe3-management-current.json'),
                '--receipts', str(root)]
    try:
        result = subprocess.run(command, cwd=control['worktree'], capture_output=True,
                                text=True, timeout=300)
        if result.returncode:
            print(json.dumps({'sample': sample, 'observation_channel_error':
                              result.stderr[-1600:], 'training_exit_inferred': False}), flush=True)
        else:
            actual = json.loads(result.stdout.strip().splitlines()[-1])
            groups = {}
            for prefix, group in actual.get('groups', {}).items():
                progress = group.get('train_progress') or {}
                metric = group.get('latest_epoch_metrics') or {}
                groups[prefix] = dict(
                    feature_progress=group.get('feature_progress'),
                    feature_complete=group.get('feature_complete'),
                    epoch=progress.get('epoch'), stage=progress.get('stage'),
                    adam=progress.get('actual_adam_steps_per_rank'),
                    train_accuracy=progress.get('sampled_train_accuracy'),
                    train_bce=progress.get('sampled_train_bce'),
                    diagnostic_epoch=metric.get('epoch'),
                    diagnostic_raw={key: (metric.get('raw') or {}).get(key)
                                    for key in ('accuracy', 'balanced_accuracy', 'bce')},
                    result=group.get('train_result'),
                    errors=dict(feature=group.get('feature_errors'),
                                train=group.get('train_error'),
                                ranks=group.get('train_rank_errors')))
            compact = {key: actual.get(key) for key in
                       ('receipt', 'checked_unix', 'native', 'owned_active_births',
                        'owned_births_recorded', 'owned_zombie_births', 'cleanup',
                        'native_actual_exit0_verified', 'native_resource_closure_verified')}
            compact['groups'] = groups
            print(json.dumps(compact, ensure_ascii=False), flush=True)
            state = (actual.get('native') or {}).get('state')
            if state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'):
                terminal_samples += 1
                if terminal_samples >= 2 or state != 'COMPLETED':
                    break
            else:
                terminal_samples = 0
    except Exception as error:
        print(json.dumps({'observation_channel_error': repr(error),
                          'training_exit_inferred': False}), flush=True)
    time.sleep(30)

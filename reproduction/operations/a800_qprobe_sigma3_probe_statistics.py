"""User-requested CPU statistics from immutable warmup Head and existing predictions.

This is a manual analysis entry, separate from the periodic metadata observer.
It never trains, generates trajectories, loads an actor, or changes the runtime.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
OPS = Path(__file__).resolve().parents[2]
RUN = 'f84749ca43be42bd91f2bc63d73347dd'
EXECUTION = 'ffdc8bb4690d4b8793335d9a718e3dcb'
SOURCE = 'c200f9ebcaba6fd08039f24df8e7ffc55cf19c40'
HANDLE = ROOT/'preparation/qprobe-a800-sigma3-teacher-v4-deployment.json'
RUNTIME = ROOT/'qprobe-a800-sigma3-teacher-v4'
OBSERVATION = ROOT/'preparation/qprobe-sigma3-v4-progress-observation5.json'
REMOTE_PATH = OPS/'reproduction/operations/qprobe_sigma3_probe_statistics_remote.py'
RECEIPT = ROOT/'preparation/qprobe-sigma3-f847-manual-probe-statistics.json'


def main():
    commit = subprocess.check_output(['git', '-C', str(OPS), 'rev-parse', 'HEAD']).decode().strip()
    for path in (Path(__file__).resolve(), REMOTE_PATH):
        assert subprocess.check_output(['git', '-C', str(OPS), 'show', commit+':'+str(path.relative_to(OPS))]) == path.read_bytes()
    publication = json.loads((ROOT/'preparation/qprobe-sigma3-f847-manual-statistics-operator-publication.json').read_text())
    assert publication['operations_commit'] == publication['remote_commit'] == commit
    assert publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    for path in (Path(__file__).resolve(), REMOTE_PATH):
        row = next(row for row in publication['files'] if row['path'] == str(path.relative_to(OPS)))
        assert row['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    remote = REMOTE_PATH.read_text()
    ast.parse(remote)
    spec = importlib.util.spec_from_file_location('published_sigma3_primary', OPS/'reproduction/operations/a800_qprobe_sigma3_fresh_observer.py')
    primary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(primary)
    flex, _, _, _ = primary.load_flex()
    intent = flex.read(HANDLE)
    authority_sha = flex.authority(HANDLE, intent, RUN, EXECUTION)
    assert intent['source_commit'] == SOURCE
    observation = json.loads(OBSERVATION.read_text())['observation']
    assert (observation['run_id'], observation['execution_id'], observation['source_commit']) == (RUN, EXECUTION, SOURCE)
    assert observation['reward_window']['status'] == 'observed'
    steps = [dict(step=row['step'], attempt_id=row['attempt_id']) for row in observation['reward_window']['new_complete_rows']]
    assert [row['step'] for row in steps] == [1, 2, 3]
    module = flex.runtime_module(RUNTIME, intent)
    route = module.ssh_command(remote)
    assert route[-1].startswith('python3 -c ')
    route[-1] = '/data/search-g1/env/bin/python -c '+route[-1][len('python3 -c '):]
    assert not RECEIPT.exists()
    payload = dict(base=intent['base'], run=RUN, execution=EXECUTION, source=SOURCE,
        expected_initial_head_sha256='8a34948bbb115213be405b192f4b40d04bfc25a7f35f7ce1d02d389c845ae641',
        expected_warmup_ready_sha256='0fb27f558bd7e4c7b36e5c52a74df44da257e8715cdd33553e19455839fd5fa8',
        completed_steps=steps)
    result = dict(role='user_requested_saved_head_cpu_statistics_receipt', operator_commit=commit,
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_code_sha256=hashlib.sha256(remote.encode()).hexdigest(), authority_sha256=authority_sha,
        run_id=RUN, execution_id=EXECUTION, source_commit=SOURCE, input_observation_sha256=hashlib.sha256(OBSERVATION.read_bytes()).hexdigest(),
        user_requested_scope='probe loss and mixed proportion, initial5120 cold/after-warmup binary accuracy',
        cpu_only_saved_question_head_analysis=True, no_actor_load=True, no_training_updates=True,
        no_new_rollouts=True, no_gpu_evaluation=True, no_remote_mutation=True,
        one_manual_ssh_query=True, no_automatic_retry=True, observation=None)
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            completed = subprocess.run(route, input=json.dumps(payload).encode(), stdout=out, stderr=err, timeout=150)
            result.update(exit_code=completed.returncode, status='READ_RETURNED')
        except (OSError, subprocess.TimeoutExpired) as failure:
            result.update(exit_code=None, status='TRANSPORT_OR_ANALYSIS_UNKNOWN', failure_type=type(failure).__name__)
        result.update(stdout_bytes=out.tell(), stderr_bytes=err.tell())
        out.seek(0); raw = out.read(128*1024+1)
        err.seek(0); result['bounded_stderr_sha256'] = hashlib.sha256(err.read(64*1024)).hexdigest()
    if result['exit_code'] == 0 and len(raw) <= 128*1024:
        rows = [json.loads(line.split(':', 1)[1]) for line in raw.decode().splitlines() if line.startswith('PROBE_STATS_JSON:')]
        assert len(rows) == 1
        value = rows[0]
        assert (value['run_id'], value['execution_id'], value['source_commit']) == (RUN, EXECUTION, SOURCE)
        result['observation'] = value
    result['checked_unix'] = time.time()
    flex.once(RECEIPT, result)
    print(json.dumps(dict(receipt=str(RECEIPT), status=result['status'], exit_code=result['exit_code'], observation=result['observation']), allow_nan=False))
    return 0 if result['observation'] is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())

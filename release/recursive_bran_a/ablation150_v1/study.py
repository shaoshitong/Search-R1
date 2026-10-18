"""Sequential task-owned execution; matched conditions reuse accepted outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .contracts import planned_conditions, read_receipt, sha
from .prepare import prepare
from .progress import replace_json


def reusable_original(path):
    """Bind A reuse to its native completed150 actual result, not its200 EM."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    identity = value.get('identity', {})
    if (value.get('status') != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or identity.get('completed_step') != 150
            or identity.get('code_commit') != 'a1ec638575268747468b1a14ff8fd5f858ca0e1a'
            or identity.get('config_sha256') != '43e28cc7ecc0d38a93324bf480aafb23a0c1ecdb979e9ea587d0358ae095233d'
            or value.get('boundary_phase') != 'post-evaluation'):
        raise ValueError('A reuse needs the actual accepted scheduled150 boundary')
    return dict(path=str(path), sha256=sha(path), completed_outer=150,
                role='A-original', scope='Matched original150 schedule; aggregate EM reported separately')


def wait_gpu_release(timeout=90):
    """No signals. Only advance when the allocated task’s GPU contexts disappear."""
    started = time.monotonic()
    while True:
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=15)
        if query.returncode == 0 and not query.stdout.strip():
            return dict(time=time.time(), complete=True, applications=[], query_rc=0)
        if time.monotonic()-started > timeout:
            raise RuntimeError('Previous stage GPU context release remains unconfirmed; no duplicate stage')
        time.sleep(3)


def run(args):
    base = Path(args.base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[3]
    original = reusable_original(args.original150_manifest)
    conditions = planned_conditions()
    known = {'A-original': original}
    state_path = base / 'study-progress.json'
    state = dict(started=time.time(), status='ACTIVE', source_commit=args.source_commit,
                 entry_source_commit=os.environ.get('RBA_ABLATION_ENTRY_COMMIT', args.source_commit),
                 requested_labels=len(conditions), conditions=conditions, complete={}, current=None)
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old['source_commit'] != args.source_commit or old['conditions'] != conditions:
            raise ValueError('Existing sequential study belongs to different frozen conditions/source')
        state = old
        state['entry_source_commit'] = os.environ.get('RBA_ABLATION_ENTRY_COMMIT', args.source_commit)
        known.update(state['complete'])
    start = dict(time=time.time(), pid=os.getpid(), uid=os.getuid(),
        tick=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        source_commit=args.source_commit, allocation=os.environ.get('SLURM_JOB_ID'),
        execution_id=os.environ.get('RBA_H100_SHARED_EXECUTION_ID', os.environ.get('SLURM_JOB_ID')))
    start['entry_source_commit'] = os.environ.get('RBA_ABLATION_ENTRY_COMMIT', args.source_commit)
    previous = base / 'study-start.json'
    if previous.exists():
        owner = json.loads(previous.read_text())
        proc = Path('/proc', str(owner['pid']))
        if proc.exists() and int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]) == owner['tick']:
            raise ValueError('Another matching study owner still exists')
    replace_json(previous, start)
    for row in conditions:
        group = row['group']
        if group in state['complete']:
            known[group] = state['complete'][group]
            continue
        if row.get('reuse'):
            accepted = known[row['reuse']]
            reused = dict(reuse_of=row['reuse'], accepted=accepted,
                condition=row['options'], generated_new_training=False)
            state['complete'][group] = reused
            known[group] = reused
            replace_json(state_path, state)
            continue
        group_base = base / group
        receipt_path = group_base / 'receipts' / 'configuration.json'
        if not receipt_path.exists():
            prepare(args.template, group_base, args.source_commit, group, row['options'], args.control_token_file)
        record = read_receipt(receipt_path)
        if record['condition'] != row['options'] or record['source_commit'] != args.source_commit:
            raise ValueError('Existing group config is not this frozen single-factor condition')
        entry_flags = []
        if (group_base / 'receipts' / 'current-entry.json').exists():
            old = json.loads((group_base / 'receipts' / 'current-entry.json').read_text())
            completed = Path(old['output']) / 'training-complete.json'
            terminal = Path(old['output']) / 'entry-exit.json'
            if completed.exists() and terminal.exists() and json.loads(terminal.read_text()).get('rc') == 0:
                release = wait_gpu_release()
                accepted = dict(training_complete=str(completed), complete_sha256=sha(completed),
                    entry_exit=str(terminal), entry_exit_sha256=sha(terminal), owned_gpu_release=release)
                state['complete'][group] = accepted
                known[group] = accepted
                replace_json(state_path, state)
                continue
            request_path = group_base / 'receipts' / 'explicit-recovery.json'
            request = json.loads(request_path.read_text()) if request_path.is_file() else None
            if request is not None:
                if (request['previous_pid'], request['previous_tick'], request['previous_uid']) != (old['pid'], old['tick'], old['uid']):
                    raise ValueError('Explicit recovery is not the current exact failed owner')
                if request['kind'] == 'ZERO_WRITE_SAVED_BATCH':
                    entry_flags = ['--zero-write-manifest', request['manifest'], '--zero-write-sha256', request['sha256']]
                elif request['kind'] == 'COMPLETE_NATIVE_BOUNDARY':
                    entry_flags = ['--resume-manifest', request['manifest'], '--resume-sha256', request['sha256']]
                else:
                    raise ValueError('Unknown explicit recovery kind')
                retry_proof = None
                state['explicit_recovery'] = dict(group=group, request=request, time=time.time())
            else:
                from .unstarted_retry import check
                retry_proof = check(group_base, old)
                entry_flags = ['--retry-unstarted']
                state['unstarted_retry'] = dict(group=group, proof=retry_proof, time=time.time())
            preserved_wait = group_base / 'receipts' / ('actual-parent-wait-pid-' + str(old['pid']) + '.json')
            if not preserved_wait.exists():
                preserved_wait.write_bytes((group_base / 'receipts' / 'actual-parent-wait.json').read_bytes())
            preserved_entry = group_base / 'receipts' / ('entry-pid-' + str(old['pid']) + '.json')
            if not preserved_entry.exists():
                preserved_entry.write_bytes((group_base / 'receipts' / 'current-entry.json').read_bytes())
        state['current'] = group
        state['status'] = 'ACTIVE'
        replace_json(state_path, state)
        env = dict(os.environ, RBA_ABLATION_BASE=str(group_base), RBA_ABLATION_SOURCE_COMMIT=args.source_commit)
        print('ABLATION_STUDY_START_GROUP=' + json.dumps(dict(group=group, time=time.time())), flush=True)
        log_path = group_base / 'logs' / 'training.log'
        if log_path.exists():
            log_path = group_base / 'logs' / ('training-attempt-' + str(time.time_ns()) + '.log')
        state['current_training_log'] = str(log_path)
        with log_path.open('xb') as log:
            child = subprocess.Popen(['bash', str(source / 'release' / 'recursive_bran_a' / 'ablation150_v1' / 'run.sh'), *entry_flags],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child'] = dict(pid=child.pid, group=group, started=time.time())
            replace_json(state_path, state)
            rc = child.wait()
        replace_json(group_base / 'receipts' / 'actual-parent-wait.json', dict(time=time.time(),
            pid=child.pid, rc=rc, group=group, source_commit=args.source_commit,
            entry_source_commit=os.environ.get('RBA_ABLATION_ENTRY_COMMIT', args.source_commit)))
        if rc != 0:
            state['status'] = 'PROGRAMMING_OR_RUNTIME_FAILURE_REQUIRES_NATIVE_RECOVERY'
            state['actual_parent_wait_rc'] = rc
            replace_json(state_path, state)
            raise RuntimeError(f'Group {group} returned actual wait {rc}; committed state retained')
        entry = json.loads((group_base / 'receipts' / 'current-entry.json').read_text())
        complete = Path(entry['output']) / 'training-complete.json'
        value = json.loads(complete.read_text())
        if value.get('completed_outer') != 150 or value.get('evaluation_steps') != [5, 50, 100, 150]:
            raise ValueError('A successful process return is not complete150/fourdev evidence')
        release = wait_gpu_release()
        accepted = dict(training_complete=str(complete), complete_sha256=sha(complete),
            actual_parent_wait=str(group_base / 'receipts' / 'actual-parent-wait.json'),
            actual_parent_wait_rc=rc, owned_gpu_release=release)
        state['complete'][group] = accepted
        known[group] = accepted
        replace_json(state_path, state)
    state.update(status='ALL_PLANNED_LABELS_150_COMPLETE_OR_MATCHED_REUSE', current=None,
                 finished=time.time(), analysis_delivery_complete=False)
    replace_json(state_path, state)
    print('ABLATION_STUDY_TRAINING_SEQUENCE_COMPLETE=' + json.dumps(state, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--template', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--control-token-file', required=True)
    parser.add_argument('--original150-manifest', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

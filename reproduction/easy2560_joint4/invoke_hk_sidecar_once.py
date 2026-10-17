"""Local, single-attempt HK invocation of the published CPU sidecar command.

Default dry-run reads only the pinned public command backup. Explicit execution
reads the authorized jxu957 credential into RAM and sends JSON over SSH stdin.
No campus failover, retry, watcher, raw SSH output, or credential receipt exists.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

PUBLICATION = '77ca9a0268eb0b6901508ccd3da7c7fb3602b3ce'
COMMAND_JSON_SHA = '3c1e881c0323c40ae65689aac4f26468dd14a7b2f33916dd820caa156e41cc5d'
COMMAND_CODE_SHA = '06505ce5a30653f4575ed80b28f1c5bff5e0747fe9edc3fb9df8cad5c525db09'
COMMAND_BACKUP = Path(__file__).parent / 'code_backup/micro16-bound-acceptance-and-storage-20261018/cpu-acceptance-micro16-HK-sidecar-sync-command-20261018.json'
PRIVATE_CREDENTIAL = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md')
TRAINING_COMMIT = '8a5bac94b8230a99b81977f5290589e3d3708dfe'
RUN_ID = 'dfe9c94084bb48b2a9bb7b60f9d882da'
SOURCE_JOB = '730226'
STEPS = (5, 50, 100, 150, 200)


def command_document(step):
    if type(step) is not int or step not in STEPS:
        raise ValueError('FIXED_CHECKPOINT_REQUIRED')
    raw = COMMAND_BACKUP.read_bytes()
    if hashlib.sha256(raw).hexdigest() != COMMAND_JSON_SHA:
        raise ValueError('PUBLISHED_COMMAND_JSON_DIFFERS')
    document = json.loads(raw)
    argv = document['argv']
    if (len(argv) != 5 or argv[:3] != ['/usr/bin/python3', '-B', '-c'] or argv[-1] != '5'
            or hashlib.sha256(argv[3].encode()).hexdigest() != COMMAND_CODE_SHA
            or (document['training_commit'], document['run_id'], document['source_job']) !=
            (TRAINING_COMMIT, RUN_ID, SOURCE_JOB)):
        raise ValueError('PUBLISHED_COMMAND_BINDING_DIFFERS')
    argv = [*argv[:-1], str(step)]
    destination = str(Path(document['HK_destination']).parent / ('global_step_' + str(step)))
    return document, argv, destination


def read_jxu957_password():
    # Same authorized private source/parser as local_account_bridge, without
    # invoking its HK->HPC command route. No value is returned by a dry-run.
    secret = PRIVATE_CREDENTIAL.read_text()
    line = next(line for line in secret.splitlines() if line.startswith('账号 jxu957'))
    password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
    secret = line = None
    if not password:
        raise ValueError('EMPTY_AUTHORIZED_CREDENTIAL')
    return password


def run_once(step=5, *, execute=False, receipt=None, runner=subprocess.run,
             credential_reader=read_jxu957_password):
    document, argv, destination = command_document(step)
    result = dict(schema_version=1, role='local_easy2560_HK_sidecar_single_attempt',
                  status='DRY_RUN_NO_CREDENTIAL_READ_NO_SSH', publication_commit=PUBLICATION,
                  command_json_sha256=COMMAND_JSON_SHA, command_code_sha256=COMMAND_CODE_SHA,
                  base=document['base'], training_commit=TRAINING_COMMIT,
                  operator_commit=document['operator_commit'], run_id=RUN_ID, source_job=SOURCE_JOB,
                  step=step, host='hkust3', expected_host='xielab3', expected_uid=1002,
                  destination=destination, execute=execute, ssh_attempts=0,
                  credential_read_attempted=False, credential_payload_persisted=False,
                  automatic_failover=False, automatic_retry=False, watcher_started=False,
                  actual_cuda_restore_accepted=False, checked_unix=time.time())
    if execute and receipt is None:
        raise ValueError('EXECUTION_REQUIRES_NEW_LOCAL_RECEIPT_PATH')
    stream = None
    try:
        if receipt is not None:
            receipt = Path(receipt)
            if not receipt.is_absolute():
                raise ValueError('ABSOLUTE_LOCAL_RECEIPT_REQUIRED')
            # Exclusive creation prevents accidental reuse of the same dispatch
            # receipt. A killed caller leaves an unresolved prepared receipt.
            stream = receipt.open('x+', encoding='utf-8')
            prepared = dict(result, status='PREPARED_NOT_DISPATCHED' if execute else result['status'])
            stream.write(json.dumps(prepared, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
        if execute:
            password = payload = None
            try:
                result['credential_read_attempted'] = True
                password = credential_reader()
                payload = json.dumps(dict(password=password))
                password = None
                # Exactly campus()'s primary HK SSH route, intentionally without
                # that helper's returncode255 fallback/replay branch.
                ssh = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
                       '-oConnectTimeout=8', 'hkust3', shlex.join(argv)]
                result['ssh_attempts'] = 1
                completed = runner(ssh, input=payload, capture_output=True, text=True, timeout=900)
                result['ssh_returncode'] = completed.returncode
                result['status'] = 'EXECUTION_UNCERTAIN_INSPECT_BEFORE_ANY_REDISPATCH'
                if completed.returncode == 0:
                    remote = json.loads(completed.stdout)
                    if (remote.get('status') == 'EASY2560_INDEPENDENT_SMALL_SIDECAR_SHA_FSYNC_PASS'
                            and remote.get('directory') == destination
                            and remote.get('receipt') == destination + '/sidecar-archive-receipt.json'
                            and type(remote.get('file_count')) is int and remote['file_count'] > 0
                            and type(remote.get('total_bytes')) is int and remote['total_bytes'] > 0
                            and remote.get('actual_cuda_restore_accepted') is False
                            and remote.get('post_callback_rng_captured') is False):
                        result.update(status='HK_SIDECAR_SINGLE_ATTEMPT_CONFIRMED',
                                      file_count=remote['file_count'], total_bytes=remote['total_bytes'],
                                      remote_receipt=remote['receipt'])
                # Never retain or print raw stdout/stderr: even an unexpected
                # server response must not echo the secret into a local receipt.
            except subprocess.TimeoutExpired:
                result['status'] = 'EXECUTION_UNCERTAIN_TIMEOUT_INSPECT_BEFORE_ANY_REDISPATCH'
            except Exception:
                result['status'] = ('EXECUTION_UNCERTAIN_INSPECT_BEFORE_ANY_REDISPATCH'
                                    if result['ssh_attempts'] else 'LOCAL_CREDENTIAL_READ_FAILED_NO_SSH')
            finally:
                password = payload = None
        result['checked_unix'] = time.time()
        if stream is not None:
            stream.seek(0); stream.truncate()
            stream.write(json.dumps(result, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
        return result
    finally:
        if stream is not None:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=STEPS, default=5)
    parser.add_argument('--execute', action='store_true', help='one SSH attempt; no automatic failover/retry')
    parser.add_argument('--receipt', help='new absolute local small JSON receipt; required for --execute')
    args = parser.parse_args()
    try:
        result = run_once(args.step, execute=args.execute, receipt=args.receipt)
    except Exception:
        # An output/receipt error could happen after the sole SSH call. Never
        # turn that into a false "not dispatched" claim or a retry suggestion.
        print(json.dumps(dict(status=('EXECUTE_REQUESTED_WRAPPER_ERROR_INSPECT_BEFORE_ANY_REDISPATCH'
                                      if args.execute else 'LOCAL_DRY_RUN_FAILED_NO_DISPATCH'),
                              credentials_in_output=False, automatic_retry=False)))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] in ('DRY_RUN_NO_CREDENTIAL_READ_NO_SSH', 'HK_SIDECAR_SINGLE_ATTEMPT_CONFIRMED') else 2


if __name__ == '__main__':
    raise SystemExit(main())

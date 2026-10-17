"""One delayed, bounded metadata read for the active goal; never trains or notifies.

The local root remains the sole primary-state writer. A new human instruction
or a newer root observation supersedes this wait before any network request.
An existing result/claim is consumed rather than replayed. This is one local
wait, not a recurring automation or a GPU admission/inspection loop.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

from reproduction.h1008_qprobe.management_transport import remote
from .watch_native_actual import REMOTE


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def still_current(spec, state):
    return (all(state.get(key) == spec[key] for key in
                ('run_id', 'current_job_id', 'base', 'current_runtime_commit'))
            and state.get('last_actual_observation') == spec['baseline_actual']
            and state.get('root_pending_normal_read', {}).get('token') == spec['token'])


def run(path):
    path = Path(path); spec = read(path)
    result_path = Path(str(path) + '.result.json')
    if result_path.exists():
        return dict(status='EXISTING_RESULT_NO_REPLAY', result=str(result_path))
    claim = Path(str(path) + '.started.json')
    if claim.exists():
        return dict(status='EXISTING_CLAIM_RECONCILE_NO_REPLAY', claim=str(claim))
    write_once(claim, dict(token=spec['token'], started_unix=time.time(),
                           not_before_unix=spec['not_before_unix'], remote_mutations=False))
    result = None
    while True:
        state = read(spec['state_path'])
        if not still_current(spec, state):
            result = dict(status='SUPERSEDED_NO_NETWORK_READ', token=spec['token'],
                          checked_unix=time.time(), training_modified=False)
            break
        remaining = spec['not_before_unix'] - time.time()
        if remaining <= 0:
            break
        time.sleep(min(30, remaining))
    if still_current(spec, read(spec['state_path'])):
        tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory = Path(spec['directory'])
        transport = directory / ('actorhidden-normal-delayed-' + tag + '-transport.json')
        try:
            control = state.get('management_transport', {}).get('control_path')
            if control:
                check = subprocess.run(['/usr/bin/ssh', '-S', control, '-O', 'check', 'mgmt-host'],
                                       capture_output=True, timeout=10)
                if check.returncode:
                    control = None
            code = REMOTE.replace('BASE', repr(spec['base'])).replace('VERIFIED', str(spec['verified_outer']))
            if state.get('management_transport', {}).get('use_established_controller_route'):
                from reproduction.h1008_qprobe.established_controller_transport import remote as established_remote
                value = json.loads(established_remote(code, transport, target='h100-8', control_path=control))
            else:
                value = json.loads(remote(code, transport, control_path=control, target='h100-8',
                    source_bound_tcp=state.get('management_transport', {}).get('source_bound_direct_tcp_enabled', False),
                    source_bound_interface=state.get('management_transport', {}).get('source_bound_interface')))
            actual = directory / ('actorhidden-normal-delayed-' + tag + '-actual.json')
            write_once(actual, value)
            result = dict(status='ACTUAL_METADATA_READ_COMPLETE', token=spec['token'],
                          checked_unix=time.time(), actual=str(actual),
                          actual_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
                          complete_outer=value['complete_outer'],
                          native=(value.get('native') or {}).get('JobState', 'UNKNOWN'),
                          stage=value.get('stage'), training_modified=False,
                          primary_state_modified=False)
        except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
            result = dict(status='OBSERVATION_ERROR_NOT_NATIVE_EXIT', token=spec['token'],
                          checked_unix=time.time(), error=repr(error),
                          transport=str(transport), training_modified=False,
                          primary_state_modified=False)
    if result is None:
        result = dict(status='SUPERSEDED_NO_NETWORK_READ', token=spec['token'],
                      checked_unix=time.time(), training_modified=False)
    write_once(result_path, result)
    return dict(result, result=str(result_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    print(json.dumps(run(parser.parse_args().spec), ensure_ascii=False), flush=True)

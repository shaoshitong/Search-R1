"""One explicit Python3.8 accept recovery, followed by a fresh CPU companion.

The original failed companion and accept records are immutable evidence. The
already received stage is accepted once with the published compatibility fix;
no snapshot, emit, receive, training operation or Slurm request is replayed.
The new companion runs in a clean Python child with only its private ROOT moved.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid


HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
PARENT = HK/'post-callback-boundary-control/job-745371/future-50-200'
ROOT = PARENT/'recovery-python38-r1'
OLD_COMMIT = '6cbafbb2d602b567a21da6f6fce36af767157f48'
OLD_PID = 3844281
OLD_ERROR = "'PosixPath' object has no attribute 'is_relative_to'"
OLD_FAILURE_SHA = '03df604d14fa957ff9238e6fa955fa07342323bd6d5a2c5d349eaf3a0a4deee2'
ACCEPT_FAILURE_SHA = 'bdbebd720a16ab80c095fbf42ee1b5bad385b113ff10032c4c19a7a06f353f31'
ACCEPT_CLAIM_SHA = 'f38b3f100544026588c795ca95bf7236fbb9e27642853de38908c16f61b65ac2'
CORE = HK/'checkpoints/232f20e112ac4a978c5aba52a0367195/global_step_5'
STAGE = CORE/'.post-callback-dependencies.partial-bbbd605edd41455a864b3c0a3a86e6a1'
ACCEPT_CONTROL = CORE.parent/('.post-callback-dependencies-'+CORE.name)
TOOLS_ROOT = HK/'post-callback-future-recovery-tools'
NAMES = {'archive_post_callback_future_hk.py', 'observe_post_callback_future_hk.py',
         'recover_post_callback_future_hk.py', 'observe_post_callback_future_recovery_hk.py'}
MAX_ENVELOPE = 384*1024


def need(condition, message):
    if not condition: raise ValueError('POST_CALLBACK_PYTHON38_RECOVERY_' + message)


def load_future():
    path = Path(__file__).resolve().with_name('archive_post_callback_future_hk.py')
    need(path.parent.parent == TOOLS_ROOT and re.fullmatch('[0-9a-f]{40}', path.parent.name) is not None and
         path.resolve() == path and not path.is_symlink() and path.is_file() and path.stat().st_size <= 128*1024,
         'PUBLISHED_SAVED_RUNNER_REQUIRED')
    spec = importlib.util.spec_from_file_location('_python38_recovery_private_future_' + uuid.uuid4().hex, str(path))
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    need(loaded.ROOT == PARENT and loaded.STEPS == (50,100,150,200), 'PUBLISHED_FUTURE_IDENTITY_DIFFERS')
    return loaded


def record(helper, path):
    path = helper.canonical(path)
    need(path.is_file() and path.stat().st_size <= 32*1024**2, 'SMALL_PRESERVED_FILE_REQUIRED')
    return dict(bytes=path.stat().st_size, sha256=helper.sha(path))


def preserved_unchanged(helper, records):
    need(all(record(helper, Path(path)) == expected for path, expected in records.items()),
         'ORIGINAL_FAILURE_EVIDENCE_CHANGED')


def recovery_prerequisites():
    """Local read-only proof; called before the recovery launcher claims/writes."""
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    future = load_future(); helper = future.frozen_helper()
    claim = helper.read_json(PARENT/'claim.json'); launch = helper.read_json(PARENT/'launch-receipt.json')
    failed = helper.read_json(PARENT/'failure.json')
    need(helper.sha(PARENT/'failure.json') == OLD_FAILURE_SHA and
         helper.sha(ACCEPT_CONTROL/'accept-failure.json') == ACCEPT_FAILURE_SHA and
         helper.sha(ACCEPT_CONTROL/'accept-claim.json') == ACCEPT_CLAIM_SHA,
         'ACTUALLY_OBSERVED_FAILURE_SHA_DIFFERS')
    for value in (claim, launch):
        need(all(value.get(k) == v for k,v in dict(base=str(helper.BASE), source_job=helper.JOB,
            run_id=helper.RUN, training_commit=helper.TRAINING, dependencies_commit=OLD_COMMIT).items()),
            'ORIGINAL_COMPANION_IDENTITY_DIFFERS')
    need(launch.get('launcher_commit') == OLD_COMMIT and launch.get('root') == str(PARENT) and
         claim.get('requested_steps') == [50,100,150,200] and
         failed.get('status') == 'FAILED' and failed.get('error_type') == 'AttributeError' and
         failed.get('error') == OLD_ERROR and failed.get('completed_steps') == [] and
         failed.get('source_job') == helper.JOB and failed.get('training_failure_inferred') is False,
         'EXACT_ORIGINAL_PYTHON38_FAILURE_REQUIRED')
    birth = dict(launch['birth'])
    need(birth.get('pid') == OLD_PID and birth.get('start_ticks') == 1038489500 and birth.get('uid') == 1002 and
         birth.get('boot_id') == '34b77aa6-ff04-435f-88be-ff4b0cc3d0ae' and birth.get('hostname') == 'xielab3' and
         all(claim.get('birth', {}).get(k) == birth[k] for k in ('pid','start_ticks','uid','boot_id')),
         'ORIGINAL_BIRTH_DIFFERS')
    actual = future.process_observation(birth)
    need(actual['state'] == 'TERMINAL', 'ORIGINAL_BIRTH_NOT_PROVED_TERMINAL')
    accept_claim = helper.read_json(ACCEPT_CONTROL/'accept-claim.json')
    accept_failure = helper.read_json(ACCEPT_CONTROL/'accept-failure.json')
    transfer = helper.read_json(PARENT/'qualification-global_step_5/dependencies-transfer.json')
    need(accept_claim.get('pid') == OLD_PID and accept_claim.get('stage') == str(STAGE) and
         accept_failure.get('error_type') == 'AttributeError' and accept_failure.get('error') == OLD_ERROR and
         transfer.get('stage') == str(STAGE) and transfer.get('emit_returncode') == 0 and
         transfer.get('source_emit_completed') is True and
         transfer.get('status') == 'EASY2560_POST_CALLBACK_DEPENDENCIES_TRANSFER_SHA_PASS' and
         transfer.get('transfer_file_count') == 6 and transfer.get('transfer_bytes') == 50580970 and
         STAGE.is_dir() and not (CORE/'post-callback-dependencies').exists(), 'EXACT_RECEIVED_STAGE_REQUIRED')
    names = ('claim.json','launch-claim.json','launch-receipt.json','failure.json','runner.stdout.log',
        'completed-five-read-only.json','qualification-global_step_5/claim.json',
        'qualification-global_step_5/source-tools.json','qualification-global_step_5/dependencies-source-snapshot.json',
        'qualification-global_step_5/dependencies-transfer.json','qualification-global_step_5/dependencies-emit.stderr.log')
    preserved = {str(PARENT/name):record(helper, PARENT/name) for name in names}
    for name in ('accept-claim.json','accept-failure.json','receive-claim.json','receive-result.json'):
        preserved[str(ACCEPT_CONTROL/name)] = record(helper, ACCEPT_CONTROL/name)
    expected_code = {'archive_post_callback_future_hk.py':'57deae5ce35f54506bf962534f4965172082d9752d51b151f2b72b7a21f90d17',
        'observe_post_callback_future_hk.py':'5a91270fc0beead0f7724f0fee29cd598981a2a73fc97bf2ce445df83d73dfd4'}
    need(set(launch.get('public_code',{})) == set(expected_code), 'ORIGINAL_PUBLIC_MEMBER_SET_DIFFERS')
    for name, row in launch['public_code'].items():
        path = HK/'post-callback-future-launch-tools'/OLD_COMMIT/name
        need(row.get('path') == str(path) and row.get('sha256') == expected_code[name] and
             record(helper,path) == dict(bytes=row['bytes'],sha256=row['sha256']),
             'ORIGINAL_PUBLIC_SOURCE_DIFFERS')
        preserved[str(path)] = record(helper,path)
    old_dependency = HK/'post-callback-dependency-tools'/OLD_COMMIT/'post_callback_dependencies.py'
    need(helper.sha(old_dependency) == claim['dependencies_sha256'] ==
         'fab9806d9fb38dd8a1c1ece00a2fd04bb5889a7ff74ea929d8e361339ee289fe', 'ORIGINAL_DEPENDENCY_SOURCE_DIFFERS')
    preserved[str(old_dependency)] = record(helper,old_dependency)
    return dict(base=str(helper.BASE), source_job=helper.JOB, run_id=helper.RUN, training_commit=helper.TRAINING,
        parent=str(PARENT), recovery_root=str(ROOT), stage=str(STAGE), parent_birth=birth,
        original_process=actual, observed_terminal=True, exact_python38_failure=True,
        accept_claim_sha256=helper.sha(ACCEPT_CONTROL/'accept-claim.json'),
        accept_failure_sha256=helper.sha(ACCEPT_CONTROL/'accept-failure.json'), preserved_files=preserved,
        stage_metadata={name:record(helper,STAGE/name) for name in ('source-snapshot.json','transfer-receipt.json')},
        core5_actual_ack=helper.ready_core(), source_ssh_called=False, training_modified=False,
        healthy_controller_modified=False, jobs_submitted=0, checked_unix=time.time())


def published_sources(helper, data):
    launch = helper.read_json(ROOT/'launch-receipt.json'); commit = launch.get('launcher_commit')
    need(isinstance(commit,str) and re.fullmatch('[0-9a-f]{40}',commit) is not None and
         launch.get('root') == str(ROOT) and launch.get('dependencies_commit') == data['commit'] == commit and
         launch.get('dependencies_sha256') == data['sha256'] and
         all(launch.get(k) == v for k,v in dict(base=str(helper.BASE),source_job=helper.JOB,
             run_id=helper.RUN,training_commit=helper.TRAINING).items()),
         'RECOVERY_LAUNCH_BINDING_DIFFERS')
    tools = TOOLS_ROOT/commit
    need(launch.get('tools') == str(tools) and set(launch.get('public_code',{})) == NAMES and
         Path(__file__).resolve() == tools/'recover_post_callback_future_hk.py', 'RECOVERY_PUBLIC_CODE_SCOPE_DIFFERS')
    for name, row in launch['public_code'].items():
        need(row.get('path') == str(tools/name) and record(helper,tools/name) ==
             dict(bytes=row['bytes'],sha256=row['sha256']), 'RECOVERY_PUBLISHED_SOURCE_SHA_DIFFERS')
    return launch


def run(envelope):
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    future = load_future(); helper = future.frozen_helper()
    password, raw, data = future.validate_envelope(helper,envelope)
    launch = published_sources(helper,data); evidence = recovery_prerequisites()
    helper.canonical(ROOT); child = None
    with (ROOT/'recovery-operation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not any((ROOT/name).exists() for name in ('recovery-claim.json','recovery-result.json',
            'recovery-failure.json','claim.json','result.json','failure.json')), 'RECOVERY_ALREADY_ATTEMPTED_NO_REPLAY')
        helper.write_json(ROOT/'recovery-claim.json',dict(evidence, birth=future.current_birth(),
            recovery_commit=launch['launcher_commit'], dependencies_sha256=data['sha256'],
            credentials_persisted=False, automatic_retry=False))
        helper.write_json(ROOT/'recovery-prerequisites.json',evidence)
        try:
            helper.bootstrap(); helper.install_immutable(helper.HK_TOOL,raw); helper.load_module()
            future.install_dependency(helper,data); dependencies = future.load_dependencies(helper,data)
            proof = dict(schema_version=1, role='explicit_failed_accept_parent_terminal_proof', stage=str(STAGE),
                accept_claim_sha256=evidence['accept_claim_sha256'], accept_failure_sha256=evidence['accept_failure_sha256'],
                parent_birth={key:evidence['parent_birth'][key] for key in ('pid','start_ticks','uid','boot_id','hostname')},
                observed_terminal=True, checked_unix=time.time())
            helper.write_json(ACCEPT_CONTROL/'recover-python38-terminal-proof.json',proof)
            receipt = dependencies.recover_accept(CORE,STAGE,evidence['accept_failure_sha256'])
            closure = future.inspect_closure(helper,dependencies)
            need(receipt.get('status') == dependencies.STATUS and receipt.get('actual_cuda_restore_accepted') is False and
                 helper.ready_core() == evidence['core5_actual_ack'], 'RECOVERED_ACCEPTANCE_DIFFERS')
            preserved_unchanged(helper,evidence['preserved_files'])
            helper.write_json(ROOT/'recovery-accept-result.json',dict(status='EASY7B_PYTHON38_STAGING_ACCEPT_RECOVERED',
                receipt=receipt, dependency_closure=closure, original_files_preserved=evidence['preserved_files'],
                transfer_replayed=False, source_ssh_called=False, actual_cuda_restore_accepted=False,
                credentials_persisted=False, checked_unix=time.time()))
            # Start a clean process. Reusing this process would retain the b87/
            # dependency imports from recover_accept and violate their once-only
            # import guards in normal companion.run.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            env.pop('SSHPASS',None)
            command = [sys.executable,'-B',str(Path(__file__).resolve()),'--companion']
            child = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent), env=env,
                stdin=subprocess.PIPE, start_new_session=True)
            process = Path('/proc')/str(child.pid); fields = (process/'stat').read_text().rsplit(')',1)[1].split()
            birth = dict(pid=child.pid,start_ticks=int(fields[19]),state=fields[0],uid=process.stat().st_uid,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),hostname=socket.gethostname(),sid=int(fields[3]))
            payload = json.dumps(envelope).encode(); need(0<len(payload)<=MAX_ENVELOPE,'SMALL_RAM_ENVELOPE_REQUIRED')
            child.stdin.write(payload); child.stdin.close(); payload=None; envelope.clear(); password=None
            helper.write_json(ROOT/'companion-start-receipt.json',dict(command=command,birth=birth,
                companion_root=str(ROOT), original_parent=str(PARENT), staging_recovered=True,
                clean_python_process=True, training_modified=False, credentials_persisted=False,
                source_job=helper.JOB, jobs_submitted=0, checked_unix=time.time()))
            rc = child.wait()
            need(rc == 0, 'NEW_COMPANION_EXIT_NONZERO_%d' % rc)
            done = helper.read_json(ROOT/'result.json')
            need(done.get('status') == 'EASY7B_FUTURE_POST_CALLBACK_COMPANION_COMPLETE' and
                 done.get('completed_steps') == [50,100,150,200], 'NEW_COMPANION_TERMINAL_RECEIPT_DIFFERS')
            preserved_unchanged(helper,evidence['preserved_files'])
            result = dict(status='EASY7B_PYTHON38_RECOVERY_COMPANION_COMPLETE', companion_returncode=rc,
                completed_steps=done['completed_steps'], original_files_preserved=True, training_modified=False,
                credentials_persisted=False, actual_cuda_restore_accepted=False, checked_unix=time.time())
            helper.write_json(ROOT/'recovery-result.json',result); return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)
            failure = dict(status='FAILED', error_type=type(error).__name__, error=helper.redact(error,password)[-2000:],
                original_parent=str(PARENT), original_failure_preserved=True, source_job=helper.JOB,
                automatic_writer_replay=False, training_failure_inferred=False, credentials_persisted=False,
                training_modified=False, jobs_submitted=0, checked_unix=time.time())
            helper.write_json(ROOT/'recovery-failure.json',failure)
            if not (ROOT/'failure.json').exists(): helper.write_json(ROOT/'failure.json',failure)
            raise RuntimeError('POST_CALLBACK_PYTHON38_RECOVERY_FAILED_INSPECT_PRESERVED_FAILURE') from None
        finally:
            envelope.clear(); password=None


def companion(envelope):
    future = load_future(); helper = future.frozen_helper()
    _,_,data = future.validate_envelope(helper,envelope); published_sources(helper,data)
    need((CORE/'post-callback-dependencies/dependency-archive-receipt.json').is_file(), 'CP5_RECOVERED_CHILD_REQUIRED')
    future.ROOT = ROOT  # Only this isolated module instance changes.
    return future.run(envelope)


def main():
    need(sys.argv[1:] in ([],['--companion']), 'FIXED_ENTRY_MODE_REQUIRED')
    raw = sys.stdin.buffer.read(MAX_ENVELOPE+1)
    need(0<len(raw)<=MAX_ENVELOPE, 'SMALL_STDIN_ENVELOPE_REQUIRED')
    envelope=json.loads(raw); raw=None
    print(json.dumps(companion(envelope) if sys.argv[1:] else run(envelope),sort_keys=True),flush=True)


if __name__=='__main__': main()

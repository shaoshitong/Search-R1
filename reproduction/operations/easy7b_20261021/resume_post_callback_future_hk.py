"""Resume from the already accepted CP5 dependency child, without archive replay."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import uuid

HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
ORIGINAL = HK/'post-callback-boundary-control/job-745371/future-50-200'
FAILED = ORIGINAL/'recovery-python38-r1'
ROOT = ORIGINAL/'recovery-accepted-r2'
TOOLS_ROOT = HK/'post-callback-future-resume-tools'
CORE = HK/'checkpoints/232f20e112ac4a978c5aba52a0367195/global_step_5'
FAILURE_SHA = '87f2a6600e62302504302c6b42529209fd684dc443d1726d7170d44e846f1a56'
LAUNCH_SHA = '7bad86298c5b0aac5cc07af23ec3ae83e2d91ed94d3b79b3a8684dc9116a4993'
DEPENDENCY_RECEIPT_SHA = 'db349c30ea0519e367ce44ec25f6092875656a0f79f94aa0b6e3bee7b0559506'
RECOVERY_RESULT_SHA = 'a6e04baa157effaac601fcf97fc3f2250b4a77a626d176f0ad97bfa25b08a45a'
ERROR = 'POST_CALLBACK_FUTURE_GROUNDING_DEPENDENCY_CLOSURE_NOT_ACTUALLY_ACCEPTED'
NAMES = {'archive_post_callback_future_hk.py','observe_post_callback_future_hk.py',
         'resume_post_callback_future_hk.py','observe_post_callback_future_resume_hk.py'}
MAX_ENVELOPE = 384*1024


def need(value, message):
    if not value: raise ValueError('POST_CALLBACK_ACCEPTED_RESUME_' + message)


def load_future():
    path = Path(__file__).resolve().with_name('archive_post_callback_future_hk.py')
    need(path.parent.parent == TOOLS_ROOT and re.fullmatch('[0-9a-f]{40}',path.parent.name) is not None and
         path.resolve() == path and not path.is_symlink() and path.is_file(), 'PUBLISHED_RUNNER_REQUIRED')
    spec = importlib.util.spec_from_file_location('_accepted_resume_future_'+uuid.uuid4().hex,str(path))
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    need(loaded.ROOT == ORIGINAL and loaded.STEPS == (50,100,150,200),'FUTURE_IDENTITY_DIFFERS')
    return loaded


def preserve(helper, paths):
    return {str(path):dict(bytes=path.stat().st_size,sha256=helper.sha(path)) for path in paths}


def unchanged(helper, proof):
    need(preserve(helper,[Path(path) for path in proof]) == proof,'ORIGINAL_RECORDS_CHANGED')


def resume_prerequisites():
    """Only local evidence and accepted receipts; no archive writer is called."""
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002,'HKUST3_OWNER_REQUIRED')
    future = load_future(); helper = future.frozen_helper()
    launch = helper.read_json(FAILED/'launch-receipt.json')
    claim = helper.read_json(FAILED/'recovery-claim.json'); failure = helper.read_json(FAILED/'failure.json')
    need(helper.sha(FAILED/'failure.json') == FAILURE_SHA == helper.sha(FAILED/'recovery-failure.json') and
         helper.sha(FAILED/'launch-receipt.json') == LAUNCH_SHA and failure.get('error_type') == 'ValueError' and
         failure.get('error') == ERROR and failure.get('status') == 'FAILED' and
         failure.get('source_job') == helper.JOB and failure.get('training_failure_inferred') is False,
         'EXACT_R1_BINDING_FAILURE_REQUIRED')
    for value in (launch,claim):
        need(all(value.get(k) == v for k,v in dict(base=str(helper.BASE),source_job=helper.JOB,
            run_id=helper.RUN,training_commit=helper.TRAINING).items()),'R1_IDENTITY_DIFFERS')
    birth = launch['birth']
    need(birth.get('pid') == 3867549 and birth.get('start_ticks') == 1038542290 and birth.get('uid') == 1002 and
         birth.get('boot_id') == '34b77aa6-ff04-435f-88be-ff4b0cc3d0ae' and birth.get('hostname') == 'xielab3' and
         all(claim.get('birth',{}).get(k) == birth[k] for k in ('pid','start_ticks','uid','boot_id')),
         'R1_BIRTH_DIFFERS')
    actual = future.process_observation(birth)
    need(actual.get('state') == 'TERMINAL','R1_BIRTH_NOT_PROVED_TERMINAL')
    receipt_path = CORE/'post-callback-dependencies/dependency-archive-receipt.json'
    receipt = helper.read_json(receipt_path); core_binding = receipt.get('binding',{}).get('core',{})
    need(helper.sha(receipt_path) == DEPENDENCY_RECEIPT_SHA and
         receipt.get('status') == 'EASY2560_POST_CALLBACK_DEPENDENCIES_SHA_FSYNC_PASS' and
         receipt.get('all_manifest_dependencies_covered') is True and receipt.get('destination_fsynced') is True and
         receipt.get('actual_cuda_restore_accepted') is False and receipt.get('transfer_file_count') == 6 and
         receipt.get('transfer_bytes') == 50580970 and all(core_binding.get(k) == v for k,v in dict(
             base=str(helper.BASE),run_id=helper.RUN,source_commit=helper.TRAINING,step=5,core_archive=str(CORE)).items()),
         'ACTUALLY_ACCEPTED_CP5_REQUIRED')
    control = CORE.parent/('.post-callback-dependencies-'+CORE.name)
    need(helper.sha(control/'recover-python38-result.json') == RECOVERY_RESULT_SHA,'ACTUAL_RECOVERY_RESULT_DIFFERS')
    paths = [FAILED/name for name in ('launch-claim.json','launch-receipt.json','recovery-claim.json',
        'recovery-prerequisites.json','failure.json','recovery-failure.json','runner.stdout.log')]
    paths += [ORIGINAL/name for name in ('claim.json','launch-receipt.json','failure.json','runner.stdout.log')]
    paths += [receipt_path,control/'recover-python38-result.json']
    return dict(base=str(helper.BASE),source_job=helper.JOB,run_id=helper.RUN,training_commit=helper.TRAINING,
        failed_root=str(FAILED),root=str(ROOT),birth=birth,actual_process=actual,accepted_cp5_receipt_sha256=DEPENDENCY_RECEIPT_SHA,
        core5=helper.ready_core(),preserved_files=preserve(helper,paths),fresh_payload_inspect_pending=True,
        archive_writers_called=False,source_ssh_called=False,training_modified=False,checked_unix=time.time())


def published_sources(helper,data):
    launch = helper.read_json(ROOT/'launch-receipt.json'); commit = launch.get('launcher_commit','')
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None and launch.get('root') == str(ROOT) and
         launch.get('dependencies_commit') == data['commit'] == commit and launch.get('dependencies_sha256') == data['sha256'],
         'PUBLISHED_RESUME_BINDING_DIFFERS')
    tools = TOOLS_ROOT/commit
    need(launch.get('tools') == str(tools) and set(launch.get('public_code',{})) == NAMES and
         Path(__file__).resolve() == tools/'resume_post_callback_future_hk.py','PUBLISHED_RESUME_SCOPE_DIFFERS')
    for name,row in launch['public_code'].items():
        path = helper.canonical(tools/name)
        need(row.get('path') == str(path) and path.stat().st_size == row['bytes'] and helper.sha(path) == row['sha256'],
             'PUBLISHED_RESUME_SOURCE_SHA_DIFFERS')
    return launch


def run(envelope):
    future=load_future();helper=future.frozen_helper()
    password,_,data=future.validate_envelope(helper,envelope)
    published_sources(helper,data);evidence=resume_prerequisites()
    with (ROOT/'resume-operation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not any((ROOT/name).exists() for name in ('resume-claim.json','resume-result.json','resume-failure.json',
            'claim.json','result.json','failure.json')),'RESUME_ALREADY_ATTEMPTED_NO_REPLAY')
        helper.write_json(ROOT/'resume-claim.json',dict(evidence,resume_birth=future.current_birth(),credentials_persisted=False))
        helper.write_json(ROOT/'resume-prerequisites.json',evidence)
        try:
            # No dependency module has been imported by this entry. The normal
            # accepted-CP5 branch performs the fresh inspect once, then WAIT50.
            # Only this private module instance receives the independent ROOT.
            future.ROOT=ROOT
            done=future.run(envelope)
            need(done.get('status')=='EASY7B_FUTURE_POST_CALLBACK_COMPANION_COMPLETE' and
                 done.get('completed_steps')==[50,100,150,200],'COMPANION_COMPLETION_DIFFERS')
            unchanged(helper,evidence['preserved_files'])
            result=dict(status='EASY7B_ACCEPTED_CP5_RESUME_COMPLETE',completed_steps=done['completed_steps'],
                cp5_archive_writer_replayed=False,credentials_persisted=False,training_modified=False,checked_unix=time.time())
            helper.write_json(ROOT/'resume-result.json',result);return result
        except BaseException as error:
            failure=dict(status='FAILED',error_type=type(error).__name__,error=helper.redact(error,password)[-2000:],
                original_failures_preserved=True,cp5_archive_writer_replayed=False,source_job=helper.JOB,
                training_failure_inferred=False,credentials_persisted=False,training_modified=False,checked_unix=time.time())
            helper.write_json(ROOT/'resume-failure.json',failure)
            if not (ROOT/'failure.json').exists():helper.write_json(ROOT/'failure.json',failure)
            raise RuntimeError('POST_CALLBACK_ACCEPTED_RESUME_FAILED_INSPECT_NEW_ROOT') from None
        finally:envelope.clear();password=None


def main():
    need(not sys.argv[1:],'FIXED_ENTRY_MODE_REQUIRED')
    raw=sys.stdin.buffer.read(MAX_ENVELOPE+1);need(0<len(raw)<=MAX_ENVELOPE,'SMALL_STDIN_ENVELOPE_REQUIRED')
    envelope=json.loads(raw);raw=None
    print(json.dumps(run(envelope),sort_keys=True),flush=True)


if __name__=='__main__':main()

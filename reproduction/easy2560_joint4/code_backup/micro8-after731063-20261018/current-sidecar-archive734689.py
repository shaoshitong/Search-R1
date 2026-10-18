"""Rebound existing remote sidecar transport; default dry-run makes no SSH call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time

BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-micro8-after731063-20261018'
TRAINING = '4cc07f3103917f5ac6bb81f6b022d454809c1054'
OPERATOR = '28e839d3649782598392d2c1424c9aa277d53733'
RUN = 'dfe9c94084bb48b2a9bb7b60f9d882da'
JOB = '734689'
HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
CODE = BASE + '/source/' + OPERATOR
OPERATOR_SHA = 'a503352c5113834be34735e3b8e7f8d0d071483d8edd6ebd7ae71ebd40166db9'
SSH_SOURCE_SHA = '3a40d7566ad390792ce1606518790bedfaf30a375c7f727e9ed87e71082fc55d'
STEPS = (50, 100, 150, 200)


def need(value, message):
    if not value: raise ValueError('SIDECAR734689_' + message)


def plan(step):
    need(type(step) is int and step in STEPS, 'FIXED_NEW_SUFFIX_MILESTONE_REQUIRED')
    return dict(status='DRY_RUN_NO_CREDENTIAL_READ_NO_SSH', base=BASE, training_commit=TRAINING,
        operator_commit=OPERATOR, source_job=JOB, run_id=RUN, step=step,
        source_directory=BASE+'/preparation/checkpoint-evidence/global_step_'+str(step),
        destination=str(HK/'checkpoint-evidence'/RUN/('global_step_'+str(step))),
        existing_core=str(HK/'checkpoints'/RUN/('global_step_'+str(step))),
        original_CP5_preserved=True, credentials_persisted=False, actual_cuda_restore_accepted=False)


def remote_program(step):
    p = plan(step)
    return ('import sys,json,hashlib;from pathlib import Path;'
        'code=Path('+repr(CODE)+');operator=code/"reproduction/easy2560_joint4/checkpoint_evidence.py";'
        'assert hashlib.sha256(operator.read_bytes()).hexdigest()=='+repr(OPERATOR_SHA)+';'
        'publication=json.loads((code/"source-publication.json").read_text());'
        'assert publication["source_commit"]=='+repr(OPERATOR)+' and publication["fresh_source_all_files_sha_verified"] is True;'
        'sys.path.insert(0,str(code));from reproduction.easy2560_joint4 import checkpoint_evidence as m;'
        'assert Path(m.__file__)==operator;'
        'value=m.verify('+repr(p['source_directory'])+');'
        'print(json.dumps(dict(operator_file=str(operator),operator_sha256='+repr(OPERATOR_SHA)+',verification=value)))')


def validate_document(document, step):
    p = plan(step); v = document['verification']
    need(document['operator_file'] == CODE+'/reproduction/easy2560_joint4/checkpoint_evidence.py' and
         document['operator_sha256'] == OPERATOR_SHA, 'VERIFIER_SHA_OR_IMPORT_DIFFERS')
    need(tuple(v[k] for k in ('source_base','source_commit','run_id','source_job','step')) ==
         (BASE,TRAINING,RUN,JOB,step), 'CURRENT_EXECUTION_BINDING_DIFFERS')
    need(v['status'] == 'EASY2560_CHECKPOINT_EVIDENCE_SEALED' and v['actual_cuda_restore_accepted'] is False and
         v['post_callback_rng_captured'] is False and v['single_original_job_prefix'] is False and
         v['inherited_history_assembled'] is True, 'RESTORED_SUFFIX_PHASE_DIFFERS')
    rows = v['files_to_archive']; files = {r['relative_path']:r for r in rows}
    need(len(files) == len(rows) and {'supplement.json','complete.json'} <= set(files), 'EXACT_FILE_SET_REQUIRED')
    for name,row in files.items():
        relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name and
             name not in ('','.','sidecar-archive-receipt.json') and row['path'] == p['source_directory']+'/'+name and
             type(row['bytes']) is int and 0 <= row['bytes'] <= 128*1024**2 and
             isinstance(row['sha256'],str) and len(row['sha256']) == 64 and
             set(row['sha256']) <= set('0123456789abcdef'), 'UNSAFE_OR_UNBOUND_SIDECAR_MEMBER')
    return files


def receive_members(files, stream, stage):
    seen = set()
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        for member in archive:
            name = member.name
            need(name in files and name not in seen and member.isfile() and member.size == files[name]['bytes'],
                 'TAR_MEMBER_DIFFERS')
            path = stage/name; path.parent.mkdir(parents=True,exist_ok=True)
            h = hashlib.sha256(); count = 0
            with path.open('xb') as output:
                source = archive.extractfile(member)
                for block in iter(lambda:source.read(1024**2),b''):
                    count += len(block); h.update(block); output.write(block)
                output.flush(); os.fsync(output.fileno())
            need(count == files[name]['bytes'] and h.hexdigest() == files[name]['sha256'], 'TRANSFER_MEMBER_SHA_DIFFERS')
            seen.add(name)
    need(seen == set(files), 'TRANSFER_FILE_SET_INCOMPLETE')


def execute(step):
    p = plan(step)
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'EXACT_HK_HOST_REQUIRED')
    helper = HK/'source'/TRAINING/'reproduction/prefix20_joint4/archive_supervisor.py'
    need(hashlib.sha256(helper.read_bytes()).hexdigest() == SSH_SOURCE_SHA, 'SSH_TRANSPORT_SOURCE_DIFFERS')
    sys.path.insert(0,str(HK/'source'/TRAINING))
    from reproduction.prefix20_joint4 import archive_supervisor
    need(Path(archive_supervisor.__file__) == helper, 'SSH_TRANSPORT_IMPORT_DIFFERS')
    payload = json.load(sys.stdin); password = payload.pop('password'); payload.clear(); payload = None
    env = dict(os.environ,SSHPASS=password); password = None
    def remote_verify():
        argv = [BASE+'/env/bin/python','-B','-c',remote_program(step)]
        child = subprocess.run(archive_supervisor.ssh_argv(shlex.join(argv)),env=env,
                               capture_output=True,text=True,timeout=180)
        need(child.returncode == 0, 'SOURCE_VERIFY_FAILED_INSPECT_SOURCE')
        document = json.loads(child.stdout)
        return document, validate_document(document,step)
    before, files = remote_verify()
    accepted = json.loads((Path(p['existing_core'])/'source-acceptance.json').read_text())
    need(tuple(accepted[k] for k in ('base','source_commit','run_id','step')) == (BASE,TRAINING,RUN,step) and
         accepted['status'] == 'EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS' and
         all(accepted[k] is True for k in ('fresh_source_after_archive_sha_verified','fresh_destination_sha_verified','destination_fsynced')) and
         accepted['actual_cuda_restore_accepted'] is False, 'ACCEPTED_CURRENT_CORE_REQUIRED')
    destination = Path(p['destination'])
    need(not destination.exists() and not destination.is_symlink(), 'DESTINATION_EXISTS_INSPECT_NO_OVERWRITE')
    destination.parent.mkdir(parents=True,exist_ok=True)
    need(destination.parent.resolve() == destination.parent, 'DESTINATION_PARENT_SYMLINK')
    stage = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-partial-',dir=destination.parent))
    command = shlex.join(['tar','-C',p['source_directory'],'-cf','-','--',*sorted(files)])
    child = subprocess.Popen(archive_supervisor.ssh_argv(command),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        receive_members(files,child.stdout,stage); child.stdout.close(); child.stderr.read()
        need(child.wait(timeout=30) == 0, 'TRANSFER_PROCESS_FAILED')
        after, second = remote_verify()
        need({n:(r['bytes'],r['sha256']) for n,r in files.items()} ==
             {n:(r['bytes'],r['sha256']) for n,r in second.items()}, 'SOURCE_AFTER_TRANSFER_DIFFERS')
        need(json.loads((stage/'identity/accepted-archive.json').read_text()) == accepted, 'COPIED_CORE_RECEIPT_DIFFERS')
        for name,row in files.items():
            path = stage/name
            need(path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],
                 'FRESH_DESTINATION_SHA_DIFFERS')
        result = dict(p,status='EASY2560_INDEPENDENT_SMALL_SIDECAR_SHA_FSYNC_PASS',files={n:dict(bytes=r['bytes'],sha256=r['sha256'])
            for n,r in sorted(files.items())},file_count=len(files),total_bytes=sum(r['bytes'] for r in files.values()),
            source_verify_before=before,source_verify_after=after,fresh_source_SHA_verified_twice=True,
            fresh_destination_SHA_verified=True,destination_fsynced=True,core_archive_unmodified=True,checked_unix=time.time())
        with (stage/'sidecar-archive-receipt.json').open('x') as output:
            json.dump(result,output,sort_keys=True,indent=2); output.write('\n'); output.flush(); os.fsync(output.fileno())
        for folder,_,_ in os.walk(stage,topdown=False):
            fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        need(not destination.exists(), 'DESTINATION_APPEARED_INSPECT_NO_OVERWRITE')
        os.rename(stage,destination)
        fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        return dict(status=result['status'],directory=str(destination),file_count=len(files),total_bytes=result['total_bytes'],
                    receipt=str(destination/'sidecar-archive-receipt.json'),actual_cuda_restore_accepted=False,post_callback_rng_captured=False)
    except BaseException:
        if child.poll() is None: child.terminate(); child.wait(timeout=15)
        print('SIDECAR_FAILED_STAGE_RETAINED='+str(stage),file=sys.stderr)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--step',type=int,required=True,choices=STEPS)
    parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    print(json.dumps(execute(args.step) if args.execute else plan(args.step),sort_keys=True))


if __name__ == '__main__': main()

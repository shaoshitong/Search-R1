"""Independent HK CPU consumers for an explicitly bound new FA3 training job.

Launch from the saved published file, not ``python -m``: the supplement helper
privately imports its frozen083 package. The core controller uses a separate
process and its own published source package. Passwords enter stdin/RAM only.
Old745371 consumers and accepted CP5 archives are never changed or replayed.
"""
import argparse
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

HK=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
RUN='232f20e112ac4a978c5aba52a0367195'
STEPS=(50,100,150,200)
PREFIX='reproduction/operations/easy7b_20261021/'
MAX_ENVELOPE=16*1024


def need(value,message):
    if not value:raise ValueError('FA3_ARCHIVE_CONSUMERS_'+message)


def canonical(path):
    path=Path(path);need(path.is_absolute() and path.resolve()==path and not path.is_symlink(),'CANONICAL_PATH_REQUIRED')
    return path


def digest(path):return hashlib.sha256(canonical(path).read_bytes()).hexdigest()


def document(path):
    path=canonical(path);need(path.is_file() and path.stat().st_size<=16*1024**2,'SMALL_DOCUMENT_REQUIRED')
    return json.loads(path.read_bytes())


def publish(path,value):
    path=canonical(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def arguments(base,training_commit,job):
    base=canonical(base)
    need(re.fullmatch(r'/data/user/jxu957/experiments/easy2560-corollout-joint4-[A-Za-z0-9._-]+',str(base)) is not None,
         'OWN_NEW_TRAINING_BASE_REQUIRED')
    need(isinstance(job,str) and job.isdecimal() and job!='745371','NEW_JOB_REQUIRED')
    need(isinstance(training_commit,str) and re.fullmatch('[0-9a-f]{40}',training_commit) is not None,
         'TRAINING_COMMIT_REQUIRED')
    return base,training_commit,job


def requested_steps(first_step=50):
    return STEPS[STEPS.index(first_step):]


def paths(job,first_step=50):
    return dict(bundle=HK/'fa3-archive-consumers'/('job-'+job),
        core=HK/'archive-controller'/('job-'+job),
        companion=HK/'post-callback-boundary-control'/('job-'+job)/('future-'+str(first_step)+'-200'))


def source(controller_commit):
    need(re.fullmatch('[0-9a-f]{40}',controller_commit) is not None,'CONTROLLER_COMMIT_REQUIRED')
    root=canonical(HK/'source'/controller_commit);proof=document(root/'source-publication.json')
    need(proof.get('source_commit')==controller_commit,'PUBLISHED_SOURCE_COMMIT_DIFFERS')
    names=(PREFIX+'fa3_archive_consumers_hk.py',PREFIX+'archive_post_callback_future_hk.py',
        'reproduction/easy2560_joint4/post_callback_dependencies.py',
        'reproduction/easy2560_joint4/archive_controller.py','reproduction/easy2560_joint4/checkpoint_archive.py',
        'reproduction/easy2560_joint4/archive_login_route.py',
        'reproduction/prefix20_joint4/archive_supervisor.py')
    for name in names:
        path=canonical(root/name);row=proof['files'][name]
        need(path.is_file() and path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],
             'PUBLIC_SOURCE_SHA_DIFFERS_'+name)
    return root,{name:proof['files'][name] for name in names}


def future_module(root):
    path=root/(PREFIX+'archive_post_callback_future_hk.py')
    spec=importlib.util.spec_from_file_location('_fa3_private_archive_future_'+uuid.uuid4().hex,str(path))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    need(result.STEPS==STEPS and result.HELPER_COMMIT=='9a5025a2471c5a3cf4476a0f5981e1c99af8dc06',
         'FROZEN_FUTURE_API_DIFFERS')
    return result


def route_module(root):
    """No package import: frozen helper.bootstrap must still load only083."""
    path=root/'reproduction/easy2560_joint4/archive_login_route.py'
    spec=importlib.util.spec_from_file_location('_fa3_private_archive_route_'+uuid.uuid4().hex,str(path))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def configured_route(root,login_host=None,login_identity_json=None):
    module=route_module(root)
    host,identity=module.resolve_login_route(login_host,login_identity_json)
    identity_json=json.dumps(identity,sort_keys=True)
    def transport(command):
        return module.ssh_argv(command,login_host=host,login_identity_json=identity_json)
    flags=[]
    # No new flags on the default launch; explicit routes are propagated as a
    # complete public pin, never via credentials or inherited environment.
    if login_host is not None or login_identity_json is not None:
        flags=['--login-host',host,'--login-identity-json',identity_json]
    return transport,module.route_record(host,identity_json),flags


def bind_private_installer_route(helper,route):
    """Change only new helper-generated code; never its frozen file/globals.

    The original installer independently checks uid and hostname after the
    transport has checked hostname, boot_id and pid namespace. Retain both.
    """
    original=helper.remote_install_code
    marker="socket.gethostname()=='ACD-Login-1'"
    def remote_install_code():
        code=original()
        need(code.count(marker)==1,'FROZEN_INSTALLER_HOST_ASSERTION_DIFFERS')
        return code.replace(marker,'socket.gethostname()=='+repr(route['hpc_identity']['hostname']))
    helper.remote_install_code=remote_install_code


def configure(future,helper,base,training_commit,job,first_step=50):
    """Change only this newly imported module instance, never an old consumer."""
    steps=requested_steps(first_step)
    locations=paths(job,first_step);future.ROOT=locations['companion'];future.CONTROLLER=locations['core']
    helper.BASE=base;helper.JOB=job;helper.TRAINING=training_commit;helper.RUN=RUN
    helper.HEALTHY_CLAIM=locations['core']/'claim.json'
    helper.HPC_TOOL=base/'preparation/post-callback-tools'/helper.MODULE_COMMIT/'post_callback_archive.py'
    original_observation=future.controller_observation
    def observation(current):
        value=original_observation(current);records=value.get('records',{});result=records.get('result.json',{})
        if (current.STEP==200 and value.get('state')=='TERMINAL' and 'failure.json' not in records and
                result.get('status')=='EASY2560_FULL_CHECKPOINT_SOURCE_ARCHIVE_SHA_PASS' and
                result.get('completed_steps')==list(steps) and all(result.get(k)==v for k,v in
                    dict(base=str(base),source_job=job,source_commit=training_commit).items())):
            # A normally completed core200 controller may exit before the
            # actual post-QA200 sidecar is visible. Its freshly verified ACK
            # is complete, while the read-only boundary observation may wait.
            ack=current.ready_core()
            return dict(value,state='COMPLETE',normal_core200_complete=True,core200_actual_ack=ack,
                        training_failure_inferred=False)
        return value
    future.controller_observation=observation
    future.configure_step(helper,first_step)
    return locations


def birth(pid):
    process=Path('/proc')/str(pid);fields=(process/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),uid=process.stat().st_uid,state=fields[0],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),hostname=socket.gethostname())


def _detach(command,cwd,env,password,receipt_path,stdout_path,identity):
    with stdout_path.open('xb') as output:
        child=subprocess.Popen(command,cwd=str(cwd),env=env,stdin=subprocess.PIPE,stdout=output,
                               stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(identity,command=command,birth=birth(child.pid),credentials_persisted=False,
                detached=True,automatic_retry=False,checked_unix=time.time())
    # Publish/fsync before the child receives any payload. Once a launch is
    # uncertain its unique claim remains; this function never replays a writer.
    publish(receipt_path,record)
    child.stdin.write(json.dumps(dict(password=password)).encode());child.stdin.close()
    return record


def launch(base,training_commit,job,password,controller_commit,*,first_step=50,login_host=None,login_identity_json=None):
    need(socket.gethostname()=='xielab3' and os.getuid()==1002,'HKUST3_OWNER_REQUIRED')
    base,training_commit,job=arguments(base,training_commit,job);root,code=source(controller_commit)
    _,route,route_flags=configured_route(root,login_host,login_identity_json)
    steps=requested_steps(first_step)
    locations=paths(job,first_step);bundle=locations['bundle'];bundle.mkdir(parents=True,exist_ok=True)
    identity=dict(base=str(base),training_commit=training_commit,source_job=job,run_id=RUN,
        controller_commit=controller_commit,source_root=str(root),requested_steps=list(steps),
        first_step=first_step,historical_CP5_replayed=False,training_jobs_submitted=0,training_modified=False,
        old_controllers_modified=False,credentials_persisted=False,public_source_files=code,source_ssh_route=route)
    with (bundle/'launch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not any((bundle/name).exists() for name in ('launch-claim.json','launch-receipt.json','failure.json')),
             'LAUNCH_ALREADY_ATTEMPTED_NO_REPLAY')
        need(not any((locations['core']/name).exists() for name in ('claim.json','progress.json','failure.json','result.json')) and
             not any((locations['companion']/name).exists() for name in ('claim.json','progress.json','failure.json','result.json')),
             'NEW_JOB_CONSUMER_ALREADY_EXISTS')
        publish(bundle/'launch-claim.json',dict(identity,created_unix=time.time()))
        env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        env.pop('SSHPASS',None)
        try:
            core_command=[sys.executable,'-B','-m','reproduction.easy2560_joint4.archive_controller',
                '--base',str(base),'--commit',training_commit,'--job',job,'--first-step',str(first_step)]+route_flags
            core=_detach(core_command,root,env,password,bundle/'core-launch-receipt.json',bundle/'core.stdout.log',identity)
            companion_command=[sys.executable,'-B',str(root/(PREFIX+'fa3_archive_consumers_hk.py')),
                'companion','--base',str(base),'--commit',training_commit,'--job',job,'--controller-commit',controller_commit,
                '--first-step',str(first_step)]+route_flags
            companion=_detach(companion_command,root,env,password,bundle/'companion-launch-receipt.json',
                              bundle/'companion.stdout.log',identity)
            result=dict(identity,status='NEW_FA3_CORE_AND_POST_CALLBACK_CONSUMERS_LAUNCHED',
                core=core,companion=companion,cpu_runners_started=2,gpu_allocations=0,checked_unix=time.time())
            publish(bundle/'launch-receipt.json',result);return result
        except BaseException as error:
            publish(bundle/'failure.json',dict(identity,error_type=type(error).__name__,
                error=str(error).replace(password,'[REDACTED]')[-2000:],staging_and_any_started_child_preserved=True,
                automatic_retry=False,training_failure_inferred=False,checked_unix=time.time()))
            raise RuntimeError('FA3_ARCHIVE_LAUNCH_UNCERTAIN_INSPECT_PRESERVED_BIRTHS') from None


def run_companion(base,training_commit,job,password,controller_commit,poll_seconds=60,
                  *,first_step=50,login_host=None,login_identity_json=None):
    need(socket.gethostname()=='xielab3' and os.getuid()==1002,'HKUST3_OWNER_REQUIRED')
    base,training_commit,job=arguments(base,training_commit,job);root,_=source(controller_commit)
    ssh_argv,route,_=configured_route(root,login_host,login_identity_json)
    future=future_module(root);helper=future.frozen_helper();helper.bootstrap()
    # bootstrap's frozen083 transport remains unchanged. This new process
    # uses the separately SHA-bound public route function for every operation.
    bind_private_installer_route(helper,route)
    # Reuse the already installed immutable boundary tool. Only its original
    # four-field source envelope is synthesized; no package or file is patched.
    boundary_raw=helper.HK_TOOL.read_text()
    dependency_raw=(root/'reproduction/easy2560_joint4/post_callback_dependencies.py').read_text()
    steps=requested_steps(first_step)
    locations=configure(future,helper,base,training_commit,job,first_step);control=locations['companion']
    envelope=dict(password=password,module_source=boundary_raw,module_sha256=helper.MODULE_SHA,
        published_commit=helper.MODULE_COMMIT,dependencies_source=dependency_raw,
        dependencies_sha256=hashlib.sha256(dependency_raw.encode()).hexdigest(),dependencies_commit=controller_commit)
    _,raw,data=future.validate_envelope(helper,envelope);envelope.clear()
    control.mkdir(parents=True,exist_ok=True);env=dict(os.environ,SSHPASS=password,CUDA_VISIBLE_DEVICES='')
    with (control/'operation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not any((control/name).exists() for name in ('claim.json','result.json','failure.json')),'COMPANION_ALREADY_ATTEMPTED')
        identity=dict(base=str(base),source_job=job,run_id=RUN,training_commit=training_commit,
            controller_commit=controller_commit,requested_steps=list(steps),first_step=first_step,birth=birth(os.getpid()),
            historical_CP5_replayed=False,credentials_persisted=False,training_modified=False,
            old_controllers_modified=False,jobs_submitted=0,source_ssh_route=route,checked_unix=time.time())
        helper.write_json(control/'claim.json',identity);completed=[]
        try:
            helper.install_immutable(helper.HK_TOOL,raw);future.install_dependency(helper,data)
            boundary=helper.load_module();dependencies=future.load_dependencies(helper,data)
            # This check is read-only and deliberately retains CP5's OLD source
            # identity. No helper.ready_core() is called for CP5 under new IDs.
            historical=future.verify_completed_five(helper,boundary)
            helper.write_json(control/'historical-CP5-read-only.json',historical)
            source_installed=False
            for step in steps:
                future.configure_step(helper,step)
                before,observed=future.wait_for_core(helper,completed,poll_seconds)
                helper.CONTROL.mkdir(parents=True,exist_ok=True);helper.canonical(helper.CONTROL)
                need(not any(helper.CONTROL.iterdir()),'STEP_ALREADY_ATTEMPTED_NO_REPLAY')
                helper.write_json(helper.CONTROL/'claim.json',dict(identity,step=step,core_before=before,
                    core_controller=observed,phase=boundary.phase(step),writer_attempt=1))
                # The first requested core ACK guarantees the new allocation exists.
                # No source install is attempted while its GPU job is pending.
                if not source_installed:
                    helper.write_json(control/'source-tools.json',future.install_source_tools(helper,ssh_argv,env,raw,data))
                    source_installed=True
                future.wait_for_boundary(helper,ssh_argv,env,completed,poll_seconds)
                future.progress(helper,phase='BOUNDARY_WRITER_ONCE',step=step,completed_steps=completed)
                boundary_receipt=future.transfer_once(helper,ssh_argv,env,boundary,'boundary')
                need(helper.ready_core()==before,'CORE_CHANGED_DURING_BOUNDARY')
                future.progress(helper,phase='DEPENDENCIES_WRITER_ONCE',step=step,completed_steps=completed)
                dependencies_receipt=future.transfer_once(helper,ssh_argv,env,dependencies,'dependencies',data)
                closure=future.inspect_closure(helper,dependencies)
                need(helper.ready_core()==before,'CORE_CHANGED_DURING_DEPENDENCIES')
                helper.write_json(helper.CONTROL/'result.json',dict(identity,step=step,
                    status='EASY7B_POST_CALLBACK_STATE_AND_DEPENDENCIES_ACCEPTED',phase=boundary.phase(step),
                    boundary_receipt=boundary_receipt,dependencies_receipt=dependencies_receipt,dependency_closure=closure,
                    actual_cuda_restore_accepted=False,actual_model_optimizer_rng_dataequal_accepted=False,
                    checked_unix=time.time()))
                completed.append(step);future.progress(helper,phase='MILESTONE_SUPPLEMENTS_ACCEPTED',step=step,completed_steps=completed)
            result=dict(identity,status='EASY7B_FUTURE_POST_CALLBACK_COMPANION_COMPLETE',completed_steps=completed,
                        actual_cuda_restore_accepted=False,checked_unix=time.time())
            helper.write_json(control/'result.json',result);return result
        except BaseException as error:
            helper.write_json(control/'failure.json',dict(identity,
                status='DEPENDENCY_TERMINAL' if 'DEPENDENCY_TERMINAL' in str(error) else 'FAILED',
                error_type=type(error).__name__,error=helper.redact(error,password)[-2000:],
                completed_steps=completed,automatic_writer_replay=False,training_failure_inferred=False,
                staging_preserved=True,checked_unix=time.time()))
            raise RuntimeError('FA3_POST_CALLBACK_COMPANION_FAILED_INSPECT_PRESERVED_FAILURE') from None
        finally:env.pop('SSHPASS',None);password=None


def observe(base,training_commit,job,controller_commit,first_step=50):
    base,training_commit,job=arguments(base,training_commit,job);root,_=source(controller_commit)
    future=future_module(root);records={};locations=paths(job,first_step)
    for role,folder in locations.items():
        values={name:document(folder/name) for name in ('launch-claim.json','launch-receipt.json','claim.json',
            'progress.json','failure.json','result.json') if (folder/name).is_file()}
        claim=values.get('claim.json') or values.get('launch-receipt.json')
        if role in ('core','companion') and (locations['bundle']/(role+'-launch-receipt.json')).is_file():
            values['detached_launch']=document(locations['bundle']/(role+'-launch-receipt.json'))
            claim=claim or values['detached_launch']
        if claim and isinstance(claim.get('birth'),dict):values['actual_process']=future.process_observation(claim['birth'])
        records[role]=dict(root=str(folder),records=values)
    return dict(base=str(base),training_commit=training_commit,source_job=job,run_id=RUN,
        controller_commit=controller_commit,first_step=first_step,requested_steps=list(requested_steps(first_step)),
        read_only=True,records=records,checked_unix=time.time())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('launch','companion','observe'))
    parser.add_argument('--base',required=True);parser.add_argument('--commit',required=True)
    parser.add_argument('--job',required=True);parser.add_argument('--controller-commit',required=True)
    parser.add_argument('--first-step',type=int,choices=STEPS,default=50)
    parser.add_argument('--login-host',help='Explicit source login host; default retains pinned Login1')
    parser.add_argument('--login-identity-json',help='Exact hostname, boot_id and pid_namespace pin')
    args=parser.parse_args(argv)
    if args.mode=='observe':result=observe(args.base,args.commit,args.job,args.controller_commit,args.first_step)
    else:
        raw=sys.stdin.buffer.read(MAX_ENVELOPE+1);need(0<len(raw)<=MAX_ENVELOPE,'SMALL_PASSWORD_STDIN_REQUIRED')
        envelope=json.loads(raw);raw=None;need(set(envelope)=={'password'},'PASSWORD_STDIN_SCHEMA_DIFFERS')
        password=envelope.pop('password');need(isinstance(password,str) and 0<len(password)<=4096 and
            '\n' not in password and '\x00' not in password,'RAM_PASSWORD_REQUIRED')
        result=(launch if args.mode=='launch' else run_companion)(args.base,args.commit,args.job,password,args.controller_commit,
            first_step=args.first_step,login_host=args.login_host,login_identity_json=args.login_identity_json)
        password=None
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()

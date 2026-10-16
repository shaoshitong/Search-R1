"""HKUST3-only archival of five prefix20 checkpoints, CPU/stdlib/SSH only.

Passwords arrive via stdin and are kept in RAM only. No Mac payload staging,
HPC3 checkpoint duplication, training launch/cancel, or model/GPU operations.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import shlex
import socket
import subprocess
import sys
import time

try:
    from . import checkpoint_archive as core
    from . import archive_supervisor as held
except ImportError:
    import checkpoint_archive as core
    import archive_supervisor as held


class ObservationUnknown(RuntimeError):
    pass


# This minimal installer runs before the source-side Python bundle exists.
# It is always wrapped by held.bound_command's exact host/boot/pidns check.
INSTALL_SOURCE = r'''
import base64,hashlib,json,os,re,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
p=json.load(sys.stdin);c=p['config'];b=Path(c['base'])
assert os.getuid()==14090
assert re.fullmatch('/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-[A-Za-z0-9._-]+',str(b))
assert b.resolve()==b and re.fullmatch('[0-9a-f]{40}',c['code_commit'])
names={'checkpoint_archive.py','archive_supervisor.py','archive_controller.py'}
assert set(p['contents'])==set(c['code_manifest'])==names
root=b/'preparation/archive-control/code'/c['code_commit']
assert not any(x.is_symlink() for x in (root,*root.parents))
if root.exists():
 assert (root/'code-publication.json').is_file(), 'UNKNOWN_PARTIAL_CODE_INSTALL_INSPECT'
else:
 root.parent.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=False)
 for name in sorted(names):
  raw=base64.b64decode(p['contents'][name],validate=True);row=c['code_manifest'][name]
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  with (root/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 with (root/'code-publication.json').open('x') as f:
  json.dump(dict(code_commit=c['code_commit'],files=c['code_manifest']),f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
sys.path.insert(0,str(root));import archive_supervisor as s
s.source_host();s.validate_config(c);s.verify_bundle(root,c)
print(json.dumps(dict(status='SIDECAR_FULL_SHA_PASS',code_root=str(root),checked_unix=time.time())),flush=True)
'''


def stream_download(command, payload, destination, row, environment, progress, *, stall_seconds=300, popen=subprocess.Popen):
    """Exact non-sparse received/written bytes; publish only after actual exit0.

    A transport timeout stops this *local SSH reader* and preserves its partial
    and exit receipt. It never replays a snapshot/accept or resumes a writer.
    """
    destination=Path(destination);core.safe_components(destination.parent)
    core.need(not destination.exists() and not destination.is_symlink(),'TRANSPORT_DESTINATION_ALREADY_EXISTS')
    partial=destination.with_name(destination.name+'.partial-'+row['sha256'])
    core.need(not partial.exists() and not partial.is_symlink(),'TRANSPORT_PARTIAL_REQUIRES_INSPECTION')
    error_path=destination.with_name(destination.name+'.ssh-stderr')
    result_path=destination.with_name(destination.name+'.transport-result.json')
    started=time.time();last_change=time.monotonic();last_report=0;count=0;digest=hashlib.sha256();timed_out=False
    with partial.open('xb') as out,error_path.open('xb') as error:
        child=popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,env=environment,close_fds=True)
        monitor=selectors.DefaultSelector()
        try:
            child.stdin.write(core.document_bytes(payload));child.stdin.flush();child.stdin.close()
            monitor.register(child.stdout,selectors.EVENT_READ)
            eof=False
            while not eof:
                events=monitor.select(timeout=2)
                if not events:
                    if time.monotonic()-last_change>stall_seconds:
                        timed_out=True;child.terminate()
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    continue
                for key,_ in events:
                    block=os.read(key.fileobj.fileno(),2**20)
                    if not block:eof=True;break
                    core.need(count+len(block)<=row['bytes'],'TRANSPORT_OVERSIZED_STREAM')
                    out.write(block);digest.update(block);count+=len(block);last_change=time.monotonic()
                    if time.monotonic()-last_report>=2:
                        out.flush();os.fsync(out.fileno())
                        progress(dict(received_bytes=count,total_bytes=row['bytes'],started_unix=started,checked_unix=time.time(),
                            partial_file=str(partial),scope='actual bytes received and written; no sparse size inference'))
                        last_report=time.monotonic()
            code=child.wait(timeout=90)
            out.flush();error.flush();os.fsync(out.fileno());os.fsync(error.fileno())
        except Exception as error_value:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            out.flush();error.flush();os.fsync(out.fileno());os.fsync(error.fileno())
            value=dict(returncode=child.returncode,received_bytes=count,total_bytes=row['bytes'],timed_out=timed_out,
                output_sha256=digest.hexdigest(),status='INCOMPLETE_STREAM_PRESERVED',error_type=type(error_value).__name__,
                started_unix=started,checked_unix=time.time())
            core.publish(result_path,value)
            raise
        finally:
            monitor.close();child.stdout.close()
            if not child.stdin.closed:child.stdin.close()
    value=dict(returncode=code,received_bytes=count,total_bytes=row['bytes'],timed_out=timed_out,
        output_sha256=digest.hexdigest(),stderr_sha256=core.sha(error_path),started_unix=started,checked_unix=time.time())
    good=code==0 and not timed_out and count==row['bytes'] and digest.hexdigest()==row['sha256']
    value['status']='ACTUAL_TRANSPORT_EXIT0_BYTES_SHA_PASS' if good else 'INCOMPLETE_STREAM_PRESERVED'
    core.publish(result_path,value)
    core.need(good,'TRANSPORT_NOT_ACCEPTED_PRESERVE_PARTIAL_AND_EXIT_RECEIPT')
    core.checked_file(partial,row['sha256'],expected_bytes=row['bytes'],fsync=True)
    os.link(str(partial),str(destination));core.fsync_directory(destination.parent)
    # Keep the verified partial hardlink as bounded failure/transport evidence;
    # no extra allocated bytes and no deletion of partial evidence here.
    progress(dict(received_bytes=count,total_bytes=row['bytes'],status=value['status'],checked_unix=time.time()))
    return value


class Controller:
    def __init__(self,config,password):
        self.config=held.validate_config(config);self.root=Path(config['controller_root']);self.password=password
        self.environment=dict(os.environ,SSHPASS=password,PYTHONDONTWRITEBYTECODE='1')
        self.events=self.root/'events.jsonl';self.completed=[];self.run_id=None
        self.started=time.time();self.unknown_observations=0

    def event(self,stage,**fields):
        value=dict(stage=stage,source_job=self.config['source_job'],run_id=self.run_id,
            completed_steps=self.completed,checked_unix=time.time(),**fields)
        held.atomic(self.root/'progress.json',value)
        with self.events.open('a') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(value,allow_nan=False),flush=True)
        return value

    def remote(self,mode,payload,*,extra=None,timeout=40):
        script=held.code_root(self.config,source=True)/'archive_supervisor.py'
        command=shlex.join(['python3','-B',str(script),mode]+(extra or []))
        try:
            result=subprocess.run(held.ssh_argv(command),input=core.document_bytes(payload),env=self.environment,
                capture_output=True,timeout=timeout)
        except subprocess.TimeoutExpired:
            self.unknown_observations+=1
            raise ObservationUnknown('SSH_OBSERVATION_TIMEOUT_NO_OPERATION_REPLAY') from None
        if result.returncode:
            self.unknown_observations+=1
            error=dict(mode=mode,returncode=result.returncode,stderr=result.stderr.decode(errors='replace')[-4000:],checked_unix=time.time())
            with (self.root/'observation-errors.jsonl').open('a') as stream:
                stream.write(json.dumps(error)+'\n');stream.flush();os.fsync(stream.fileno())
            raise ObservationUnknown('SSH_OBSERVATION_FAILED_'+str(result.returncode))
        try:return json.loads(result.stdout)
        except (ValueError,UnicodeDecodeError):raise ObservationUnknown('SSH_OBSERVATION_OUTPUT_INVALID') from None

    def deploy_source_sidecar(self):
        root=held.code_root(self.config);held.verify_bundle(root,self.config)
        packet=dict(config=self.config,contents={name:base64.b64encode((root/name).read_bytes()).decode() for name in held.BUNDLE_NAMES})
        command='python3 -B -c '+shlex.quote(INSTALL_SOURCE)
        try:
            result=subprocess.run(held.ssh_argv(command),input=core.document_bytes(packet),env=self.environment,capture_output=True,timeout=45)
        except subprocess.TimeoutExpired:
            raise ObservationUnknown('SIDECAR_INSTALL_OUTCOME_UNKNOWN_INSPECT_EXISTING_DIRECTORY') from None
        receipt=dict(returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace'),checked_unix=time.time())
        core.publish(self.root/'source-sidecar-deployment.json',receipt)
        core.need(result.returncode==0,'SIDECAR_INSTALL_NOT_ACCEPTED_INSPECT_EXISTING_DIRECTORY')
        value=json.loads(result.stdout)
        core.need(value.get('status')=='SIDECAR_FULL_SHA_PASS' and value['code_root']==str(held.code_root(self.config,source=True)),
            'SIDECAR_DEPLOYMENT_BINDING_DIFFERS')

    def probe(self):
        return self.remote('probe',self.config)

    def bind_run(self,observation):
        run_id=observation.get('run_id')
        if run_id is None:return False
        core.binding(self.config['base'],self.config['training_commit'],run_id,5)
        path=self.root/'run-binding.json'
        value=dict(run_id=run_id,source_job=self.config['source_job'],base=self.config['base'],
            training_commit=self.config['training_commit'],actual_prepared_sha256=observation['prepared_sha256'],checked_unix=time.time())
        if path.exists():
            old=core._read_json(path)
            core.need(all(old[k]==value[k] for k in ('run_id','source_job','base','training_commit','actual_prepared_sha256')),
                'CONTROLLER_RUN_BINDING_CHANGED')
        else:core.publish(path,value)
        if self.run_id is not None:core.need(self.run_id==run_id,'CONTROLLER_RUNTIME_RUN_ID_CHANGED')
        self.run_id=run_id;return True

    def wait_operation(self,plan):
        # One launch request only. On uncertainty, query the held operation and
        # source claim forever/bounded deadline; never call snapshot again.
        launched_unknown=False
        try:self.remote('launch-operation',plan)
        except ObservationUnknown:launched_unknown=True
        first=True;last_signature=None
        while True:
            core.need(time.time()-self.started<=self.config['source_wait_deadline_seconds'],'CONTROLLER_WAIT_DEADLINE_EXHAUSTED')
            try:observation=self.remote('operation-status',plan)
            except ObservationUnknown:
                if first:self.event('OPERATION_OBSERVATION_UNKNOWN_NO_REPLAY',step=plan['step'],kind=plan['kind'])
                first=False;time.sleep(self.config['poll_seconds']);continue
            status=observation.get('status');progress=observation.get('operation_progress') or observation.get('source_progress') or {}
            signature=(status,progress.get('verified_bytes'),progress.get('verified_files'))
            if signature!=last_signature:
                self.event('SOURCE_OPERATION_'+str(status),step=plan['step'],kind=plan['kind'],
                    operation_root=observation['operation_root'],actual_progress=progress,
                    launch_observation_was_unknown=launched_unknown)
                last_signature=signature
            if status=='COMPLETED':
                result=observation['operation_result']
                core.need(result['worker_returncode']==0 and result['accepted_output'] is True,'HELD_OPERATION_EXIT_NOT_ZERO')
                receipt=self.root/('step-%d-%s-operation.json'%(plan['step'],plan['kind']))
                core.publish(receipt,observation)
                return observation['output']['result']
            if status in ('FAILED','UNKNOWN_OPERATION_OUTCOME','UNKNOWN_PARTIAL_OPERATION_DIRECTORY','EXISTING_SOURCE_SNAPSHOT_CLAIM_REQUIRES_INSPECTION'):
                raise RuntimeError('HELD_OPERATION_'+status+'_NO_REPLAY')
            if status=='UNSTARTED':
                raise ObservationUnknown('LAUNCH_OUTCOME_UNKNOWN_NO_REPLAY_EVEN_IF_CLAIM_NOT_YET_VISIBLE')
            first=False;time.sleep(self.config['poll_seconds'])

    def stage_snapshot(self,step,wrapped):
        value=wrapped['snapshot'];core.validate_snapshot(value,base=self.config['base'],commit=self.config['training_commit'],run_id=self.run_id,step=step)
        digest=core.digest_text(wrapped['snapshot_sha256'])
        core.need(core.document_sha(value)==digest,'SNAPSHOT_TRANSFER_JSON_BYTES_SHA_DIFFERS')
        directory=core.ARCHIVES/self.run_id/('global_step_'+str(step))
        core.safe_components(directory.parent,require_exists=False);directory.parent.mkdir(parents=True,exist_ok=True)
        directory.mkdir(exist_ok=False)
        core.publish(directory/'source-snapshot.json',value)
        core.checked_file(directory/'source-snapshot.json',digest)
        core.publish(directory/'archive-claim.json',dict(run_id=self.run_id,step=step,source_snapshot_sha256=digest,
            controller=held.identity(),code_commit=self.config['code_commit'],created_unix=time.time()))
        return directory,value,digest

    def capacity(self,directory,snapshot):
        deps={row['source_path'] for row in snapshot['dependencies'].values()};unique=set();required=2**30
        blobs=core.ARCHIVES/self.run_id/'immutable-dependencies';blobs.mkdir(exist_ok=True)
        for row in snapshot['files']:
            if row['source_path'] in deps:
                if row['sha256'] in unique:continue
                unique.add(row['sha256'])
                if (blobs/row['sha256']).exists():continue
            required+=row['bytes']
        fs=os.statvfs(str(directory));available=fs.f_bavail*fs.f_frsize
        value=dict(available_bytes=available,new_payload_allocated_upper_bound_bytes=required,
            archive_host='xielab3',dependencies_shared_by_sha=True,checked_unix=time.time())
        core.publish(directory/'storage-admission.json',value)
        core.need(available>=required,'HKUST3_ARCHIVE_STORAGE_INSUFFICIENT')
        return blobs,deps

    def transfer(self,directory,snapshot,digest,plan):
        blobs,deps=self.capacity(directory,snapshot);staging=directory/'transport';staging.mkdir()
        completed=[];new_bytes=0
        for index,row in enumerate(snapshot['files']):
            payload=directory/row['payload_path'];core.safe_components(payload.parent,require_exists=False);payload.parent.mkdir(parents=True,exist_ok=True)
            dependency=row['source_path'] in deps
            blob=blobs/row['sha256']
            if dependency and blob.exists():
                core.checked_file(blob,row['sha256'],expected_bytes=row['bytes'],fsync=True)
                os.link(str(blob),str(payload));core.fsync_directory(payload.parent)
                receipt=dict(mode='VERIFIED_IMMUTABLE_DEPENDENCY_HARDLINK',sha256=row['sha256'],bytes=row['bytes'],
                    source_path=row['source_path'],payload_path=row['payload_path'],blob_path=str(blob),checked_unix=time.time())
                core.publish(staging/('%04d-reused.json'%index),receipt)
            else:
                destination=staging/('%04d-verified-source-bytes'%index)
                command=shlex.join(['python3','-B',str(held.code_root(self.config,source=True)/'archive_supervisor.py'),
                    'stream','--index',str(index),'--snapshot-sha',digest])
                def progress(value):
                    self.event('RECEIVING_EXACT_SNAPSHOT_FILE',step=plan['step'],file_index=index,
                        files_total=len(snapshot['files']),source_path=row['source_path'],accepted_new_transport_bytes=new_bytes,**value)
                receipt=stream_download(held.ssh_argv(command),plan,destination,row,self.environment,progress,
                    stall_seconds=self.config['transport_stall_seconds'])
                core.need(receipt['returncode']==0,'TRANSPORT_ACTUAL_EXIT_NONZERO')
                new_bytes+=row['bytes']
                if dependency:
                    core.need(not blob.exists(),'IMMUTABLE_DEPENDENCY_WRITER_CONFLICT')
                    os.link(str(destination),str(blob));core.fsync_directory(blobs)
                    os.chmod(str(blob),0o440)
                    core.checked_file(blob,row['sha256'],expected_bytes=row['bytes'],fsync=True)
                    os.link(str(blob),str(payload))
                else:os.link(str(destination),str(payload))
                core.fsync_directory(payload.parent)
            completed.append(dict(source_path=row['source_path'],payload_path=row['payload_path'],bytes=row['bytes'],sha256=row['sha256']))
            held.atomic(staging/'progress.json',dict(completed_files=len(completed),total_files=len(snapshot['files']),
                actual_new_received_bytes=new_bytes,accepted_payload_bytes=sum(r['bytes'] for r in completed),checked_unix=time.time()))
        core.need(completed==[{k:r[k] for k in ('source_path','payload_path','bytes','sha256')} for r in snapshot['files']],
            'TRANSPORT_EXACT_FILE_SET_DIFFERS')
        proof=dict(status='ALL_REQUIRED_STREAMS_EXIT0_AND_DEPENDENCY_SHA_PASS',files=completed,
            actual_new_received_bytes=new_bytes,accepted_payload_bytes=snapshot['bytes'],all_actual_stream_exit_codes_zero=True,
            dependency_links_fresh_sha_verified=True,checked_unix=time.time())
        core.publish(directory/'transport-complete.json',proof)
        self.event('ALL_REQUIRED_TRANSPORTS_EXIT0',step=plan['step'],actual_new_received_bytes=new_bytes,accepted_payload_bytes=snapshot['bytes'])

    def archive_step(self,step):
        plan=held.operation_plan(self.config,self.run_id,step,'snapshot')
        wrapped=self.wait_operation(plan);directory,snapshot,digest=self.stage_snapshot(step,wrapped)
        self.transfer(directory,snapshot,digest,plan)
        self.event('HKUST3_FRESH_FULL_SHA_AND_FSYNC',step=step,archive_root=str(directory))
        wrapper=core.archive_readback(directory,directory/'source-snapshot.json',digest,base=self.config['base'],
            commit=self.config['training_commit'],run_id=self.run_id,step=step)
        core.validate_archive_proof(wrapper,snapshot,digest)
        accept=held.operation_plan(self.config,self.run_id,step,'accept',snapshot_sha=digest,archive_wrapper=wrapper)
        complete=self.wait_operation(accept)
        core.need(complete.get('status')==core.STATUS and complete.get('step')==step and complete.get('actual_cuda_restore_accepted') is False,
            'SOURCE_ACCEPT_RESULT_DIFFERS')
        core.publish(directory/'source-acceptance.json',complete)
        self.completed.append(step)
        self.event('CHECKPOINT_SOURCE_ARCHIVE_SOURCE_FULL_SHA_PASS',step=step,archive_root=str(directory),source_acceptance=complete)

    def run(self):
        core.need(socket.gethostname()=='xielab3','CONTROLLER_ACTUAL_HKUST3_REQUIRED')
        held.verify_bundle(held.code_root(self.config),self.config)
        with (self.root/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            core.publish(self.root/'controller-birth.json',held.identity())
            self.event('CONTROLLER_STARTED',training_job_changed=False,credentials_persisted=False)
            try:
                self.deploy_source_sidecar();last_wait=None
                for step in core.STEPS:
                    while True:
                        core.need(time.time()-self.started<=self.config['source_wait_deadline_seconds'],'CONTROLLER_SOURCE_DEADLINE_EXHAUSTED')
                        try:observed=self.probe()
                        except ObservationUnknown:
                            if last_wait!='UNKNOWN':self.event('SOURCE_OBSERVATION_UNKNOWN');last_wait='UNKNOWN'
                            time.sleep(self.config['poll_seconds']);continue
                        bound=self.bind_run(observed)
                        if bound and observed['pair_receipts'].get(str(step)):break
                        if observed.get('source_terminal'):
                            raise RuntimeError('SOURCE_JOB_TERMINAL_WITHOUT_REQUIRED_COMPLETED_PAIR_'+str(step))
                        state=(step,bound,observed.get('run_id'))
                        if last_wait!=state:self.event('WAITING_ACTUAL_COMPLETED_PAIR',step=step,run_binding_ready=bound);last_wait=state
                        time.sleep(self.config['poll_seconds'])
                    self.archive_step(step)
                core.need(self.completed==list(core.STEPS) and self.run_id is not None and 200 in self.completed,
                    'ALL_FIVE_MILESTONES_NOT_ACCEPTED')
                value=dict(status='ALL_FIVE_CHECKPOINT_ARCHIVES_ACCEPTED',scope='checkpoint bytes and CPU archive transport only',
                    run_id=self.run_id,source_job=self.config['source_job'],base=self.config['base'],
                    training_commit=self.config['training_commit'],code_commit=self.config['code_commit'],steps=self.completed,
                    actual_cuda_restore_accepted=False,final_native51713_accepted=False,training_job_changed=False,
                    hpc3_archives_created=0,hpc3_restore_targets_created=0,checked_unix=time.time())
                core.publish(self.root/'controller-result.json',value);self.event(value['status']);return value
            except Exception as error:
                value=dict(status='ARCHIVE_CONTROLLER_INCOMPLETE',error_type=type(error).__name__,error=str(error),
                    completed_steps=self.completed,run_id=self.run_id,source_job=self.config['source_job'],
                    actual_cuda_restore_accepted=False,final_native51713_accepted=False,partial_files_retained=True,
                    unknown_ssh_observations=self.unknown_observations,training_job_changed=False,checked_unix=time.time())
                core.publish(self.root/'controller-result.json',value);self.event(value['status'],error_type=type(error).__name__,error=str(error));raise
            finally:
                self.environment.pop('SSHPASS',None);self.password=None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=core._read_json(args.config);secret=json.load(sys.stdin);password=secret.pop('password');secret=None
    core.need(isinstance(password,str) and password and '\n' not in password,'RAM_PASSWORD_REQUIRED')
    controller=Controller(config,password);password=None;controller.run()


if __name__=='__main__':
    main()

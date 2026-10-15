"""Fresh local A800 formal200 execution with explicitly assigned component GPUs.

The shared trainer performs the real behavior gate and inline first-five audit
in its single 0-to-200 process. Local startup does not repeat historical GPU
smoke tests, export H100 assets, or claim that those tests ran.
"""
import argparse
import fcntl
import importlib.util
import json
import os
import signal
import socket
import stat
import threading
import time
from pathlib import Path
from reproduction.a800.owned_processes import publish
from reproduction.common8.run_pipeline import Pipeline,request_interrupt,supported_execution_role,gpu_rows
from reproduction.common8.verify_data import digest
from reproduction.common8.formal_contract import assert_formal_run
from reproduction.common8.topology import execution_job_id,validate_base


def bound_ray_tmp_dir(run):
    path=Path(run['ray_tmp_dir'])
    assert path==Path('/tmp')/('g-'+run['execution_id']), 'Local Ray root differs from the bound fresh run'
    current=path.lstat()
    assert stat.S_ISDIR(current.st_mode) and path.resolve()==path and path.parent.resolve()==path.parent
    assert current.st_uid==os.getuid() and stat.S_IMODE(current.st_mode)==0o700
    assert run['ray_tmp_dir_identity']==dict(device=current.st_dev,inode=current.st_ino,
        uid=current.st_uid,mode=stat.S_IMODE(current.st_mode)), 'Private Ray root identity changed'
    assert len(os.fsencode(str(path/'session_0000-00-00_00-00-00_000000_2147483647'/'sockets/plasma_store')))<107
    return str(path)


class LocalPipeline(Pipeline):
    def __init__(self,base,run):
        ray_tmp_dir=bound_ray_tmp_dir(run)
        super().__init__(base,run)
        self.env['RAY_TMPDIR']=ray_tmp_dir

    def start(self,name,args,gpus,extra=None):
        if gpus:
            self.observe_all()
            from reproduction.common8.native_allocation import assert_local_training_idle,assert_local_component_gpus_available
            if name=='trainer':assert_local_training_idle(self.run)
            else:
                records=[record for job in self.owned.jobs.values() for record in job['ledger'].values()]
                assert_local_component_gpus_available(self.run,gpus,records)
        return super().start(name,args,gpus,extra)

    def initial_probe_calibration(self):
        source=self.run.get('initial_probe_adoption_source_base')
        if source is None:
            return super().initial_probe_calibration()
        assert isinstance(source,str) and source and not any(c in source for c in '\r\n\0')
        source_base=Path(source);validate_base(source_base,self.run)
        assert source_base!=self.base and source_base.is_dir()
        self.stage('AUTHENTICATED_COMPLETED_INITIAL_300_PROBE_ADOPTION',source_base=source)
        child=self.start('initial-probe-adoption',[self.python,'-m',
            'reproduction.a800_common.adopt_initial_probes','--base',str(self.base),
            '--source-base',source],[])
        self.wait('initial-probe-adoption',child)

    def execute(self):
        sampler=None;success=False;finished_stage=None
        try:
            assert self.local and supported_execution_role(self.run,self.base)=='production'
            sampler=threading.Thread(target=self.sample,daemon=True);sampler.start()
            self.stage('VERIFYING_EXISTING_LOCAL_ORIGINAL_RETRIEVAL')
            stage=self.start('retrieval-stage',[self.python,'-m','reproduction.common8.stage_retrieval','--base',str(self.base)],[])
            self.wait('retrieval-stage',stage)
            self.stage('STARTING_ORIGINAL_FLAT_FP16_TWO_SHARD_RETRIEVER')
            nvidia_spec=importlib.util.find_spec('nvidia');torch_spec=importlib.util.find_spec('torch')
            assert nvidia_spec is not None and nvidia_spec.submodule_search_locations, 'Installed NVIDIA library namespace is unavailable'
            assert torch_spec is not None and torch_spec.origin, 'Installed Torch package is unavailable'
            torch_lib=Path(torch_spec.origin).parent/'lib';assert torch_lib.is_dir()
            libs=list(dict.fromkeys(path for root in nvidia_spec.submodule_search_locations
                for path in sorted(Path(root).glob('*/lib')) if path.is_dir()))+[torch_lib]
            retriever=self.start('retriever',[self.python,'-m','reproduction.common8.retriever'],self.retriever_gpus,
                {'PYTHONPATH':self.run['retriever_gpu_lib']+':'+str(self.repo),
                 'LD_LIBRARY_PATH':':'.join(map(str,libs))})
            health=self.endpoint('retriever',retriever,'18010/health')
            assert health['ok'] is True and health['run_id']==self.run['run_id']
            assert health['shard_count']==len(self.retriever_gpus) and health['slurm_job_id'] is None
            assert health['execution_backend']=='ssh_local' and health['execution_id']==self.run['execution_id']
            assert health['admission_sha256']==digest(self.base/'logs/retrieval-admission.json')
            from reproduction.common8.http_retrieval import admit_http
            admit_http(self.base)
            self.run.update(probe_visible_device=str(self.probe_gpus[0]),probe_logical_ordinal=0,
                retrieval_placement='original full Flat FP16 shards on physical GPUs5,6; separate four-GPU training; actual CPU/GPU and HTTP64 equality admitted')
            publish(self.base/'logs/run-manifest.json',self.run)
            self.initial_probe_calibration()
            self.stage('STARTING_REAL_FROZEN_PROBE_ENDPOINT')
            probe=self.start('probe-server',[self.python,'probe_server.py','--manifest',str(self.base/'artifacts/live/manifest.json'),'--port','18012'],self.probe_gpus)
            health=self.endpoint('probe-server',probe,'18012/health');assert health['ok'] is True
            released=self.endpoint('probe-server',probe,'18012/suspend','post',{'version':health['version']})
            assert released['suspended'] is True and released['allocated_bytes_after_release']<256*1024*1024
            resumed=self.endpoint('probe-server',probe,'18012/resume','post',{'version':health['version']})
            assert resumed['resumed'] is True and resumed['version']==health['version']
            publish(self.base/'logs/evaluator-release-admission.json',dict(run_id=self.run['run_id'],
                execution_backend='ssh_local',execution_id=self.run['execution_id'],slurm_job_id=None,
                released=released,resumed=resumed))
            self.observe_all()
            totals={int(r[0]):int(r[1]) for r in gpu_rows('index,memory.total')}
            assert set(self.max_used)==self.physical_indices and all(0<=v<=totals[i] for i,v in self.max_used.items())
            publish(self.base/'logs/local-resource-admission.json',dict(accepted=True,
                role='actual_local_initial_service_resource_observation',run_id=self.run['run_id'],
                execution_backend='ssh_local',execution_id=self.run['execution_id'],slurm_job_id=None,
                host=socket.gethostname(),training_world_size=self.world,
                training_physical_indices=self.run['training_physical_indices'],
                auxiliary_physical_indices=self.run['auxiliary_physical_indices'],
                allocation_sha256=digest(self.base/'logs/native-allocation-admission.json'),
                retrieval_admission_sha256=digest(self.base/'logs/retrieval-admission.json'),
                retrieval_http_admission_sha256=digest(self.base/'logs/retrieval-http-admission.json'),
                fitted_probe_manifest_sha256=digest(self.base/'artifacts/live/manifest.json'),
                evaluator_release_admission_sha256=digest(self.base/'logs/evaluator-release-admission.json'),
                sampled_peak_mib=self.max_used,sampling_interval_seconds=2,
                occupancy_policy='monitor_only',sampled_peak_fraction_stop_limit=None,
                historical_gpu_smoke_executed=False,
                scope='actual assigned-GPU sampling during local original retrieval and real initial300 calibration; full2560 behavior gate and first-five health remain prospective'))
            self.stage('REAL_FITTED_PROBE_TRAINING_ADMISSION')
            admission=self.start('probe-admission',[self.python,'-m','reproduction.common8.probe_admission'],self.probe_gpus)
            self.wait('probe-admission',admission,300)
            finished_stage=self.train_and_verify();success=True
        except Exception as error:
            self.stage('FAILED',error=repr(error));raise
        finally:
            try:
                self.owned.stop_all()
                if (self.base/'logs/corpus-cache-claim.json').exists():
                    from reproduction.a800_common.cache import cleanup
                    cleanup(self.base)
                recorded={r['pid'] for job in self.owned.jobs.values() for r in job['ledger'].values()}
                deadline=time.monotonic()+30
                while True:
                    remaining=[r for r in gpu_rows('gpu_uuid,pid','compute-apps') if int(r[1]) in recorded]
                    if not remaining:break
                    assert time.monotonic()<deadline, 'Owned CUDA contexts remain after bounded release wait'
                    time.sleep(.5)
                publish(self.base/'logs/cleanup-status.json',dict(run_id=self.run['run_id'],
                    execution_backend='ssh_local',execution_id=self.run['execution_id'],slurm_job_id=None,
                    cleaned=True,checked_unix=time.time(),scope='only recorded project PID births and private local corpus cache'))
            except Exception as error:
                self.stage('OWNED_CLEANUP_FAILED',error=repr(error));raise
            finally:self.finish_sampling(sampler)
        if success:self.stage(finished_stage)


def launch(base,run):
    base=Path(base).resolve();validate_base(base,run);assert_formal_run(run)
    assert os.environ.get('RAY_TMPDIR')==bound_ray_tmp_dir(run), 'Launcher Ray root differs from the bound fresh run'
    assert execution_job_id(run) is None and run['execution_host']==socket.gethostname()
    assert supported_execution_role(run,base)=='production'
    assert run['completed_updates']==0
    if run.get('full_state_restore'):
        from search_g1.qprobe_checkpoint import read_source,digest as restore_digest,SAVED_CHECKPOINT_STATUS
        binding=run['full_state_restore'];path,receipt,_,_,_=read_source(binding['source_receipt'])
        assert receipt['status']==SAVED_CHECKPOINT_STATUS and binding['kind']=='question_probe_saved_checkpoint'
        assert restore_digest(path)==binding['source_receipt_sha256'] and run['initial_probe_version']==receipt['reward_probe_version']
    else:assert run['initial_probe_version'] is None
    assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE')=='4'
    for name in ('logs','tmp','ray','cache/triton','cache/inductor','cache/hf','artifacts'):
        (base/name).mkdir(parents=True,exist_ok=True)
    lock=(base/'gpu-job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,request_interrupt);signal.signal(signal.SIGINT,request_interrupt)
    assert not (base/'logs/process-ledger.json').exists(), 'Inspect/clean this attempt; never duplicate launch'
    assert not (base/'logs/pipeline-launch-claim.json').exists(), 'Fresh local execution has already been claimed'
    for name,sha in run['source_sha256'].items():assert digest(base/'repo'/name)==sha
    for port in (18010,18012):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    from reproduction.common8.native_allocation import admit,bind_idle_local_placement,local_placement
    bound=bind_idle_local_placement(base)
    assert bound['run_id']==run['run_id'] and bound['execution_id']==run['execution_id']
    run.clear();run.update(bound)
    training,_,_,_=local_placement(run)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,training))
    allocation=admit(base)
    claim=dict(run_id=run['run_id'],execution_backend='ssh_local',execution_id=run['execution_id'],
        slurm_job_id=None,node=allocation['node'],pipeline_identity=allocation['pipeline_identity'],
        training_physical_indices=allocation['training_physical_indices'],
        auxiliary_physical_indices=allocation['auxiliary_physical_indices'],
        excluded_physical_indices=allocation['excluded_physical_indices'],
        gpu_placement_binding_sha256=allocation['gpu_placement_binding_sha256'],
        assigned_gpu_uuids=allocation['assigned_gpu_uuids'],
        allocation_sha256=digest(base/'logs/native-allocation-admission.json'),claimed_unix=time.time(),
        semantics='exclusive actual local pipeline startup, before owned service/trainer launches; no training completion claim')
    with (base/'logs/pipeline-launch-claim.json').open('x') as f:
        json.dump(claim,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(base/'logs',os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    run['pipeline_identity']=allocation['pipeline_identity'];publish(base/'logs/run-manifest.json',run)
    from reproduction.common8.disk_admission import admit as admit_disk
    admit_disk(base)
    LocalPipeline(base,run).execute()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    base=a.base.resolve();run=json.loads((base/'logs/run-manifest.json').read_text())
    launch(base,run)


if __name__=='__main__':main()

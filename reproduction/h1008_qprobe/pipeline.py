"""One actual native allocation: official feature precompute, fresh8 training200."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time


def once(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


class UUIDLocks:
    def __init__(self,uuids):
        root=Path('/mnt/data/shaoshitong/.search-g1-gpu-uuid-locks');root.mkdir(exist_ok=True);self.fds=[]
        try:
            for uuid in sorted(uuids):
                fd=os.open(root/(uuid+'.lock'),os.O_CREAT|os.O_RDWR,0o600);self.fds.append(fd)
                fcntl.fcntl(fd,fcntl.F_OFD_SETLK,struct.pack('hhqqi4x',fcntl.F_WRLCK,os.SEEK_SET,0,0,0))
        except BaseException:self.close();raise
    def close(self):
        for fd in self.fds:os.close(fd)
        self.fds=[]


def build_pipeline(base,run,config):
    from reproduction.common8.run_pipeline import Pipeline
    class H1008Pipeline(Pipeline):
        def initial_probe_calibration(self):
            if self.run.get('full_state_restore'):
                return super().initial_probe_calibration()
            from reproduction.h1008_qprobe.adopt_reward_zero import adopt
            receipt=adopt(self.base,config['reward_zero_source_base'],config['reward_zero_archive'],config['reward_zero_archive_sha256'])
            self.stage('AUTHENTICATED_OFFICIAL_ZERO_REWARD_CALIBRATION_REUSED',version=receipt['version'],new_actor_or_optimizer_state_inherited=False)
    return H1008Pipeline(base,run)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve()
    config=json.loads((base/'preparation/h1008-launch-config.json').read_text())
    assert config['role'] in ('fresh_h1008_transformer3_question_probe200','restore_h1008_transformer3_question_probe200')
    assert config['base']==str(base)
    assert bool(config.get('restore_receipt')) == (config['role']=='restore_h1008_transformer3_question_probe200')
    assert socket.gethostname().split('.')[0]=='h100-8'
    os.environ.update(SEARCH_G1_BASE=str(base),SEARCH_G1_PROFILE='common8',SEARCH_G1_TRAINING_WORLD_SIZE='8',
        SEARCH_G1_EXECUTION_PROFILE='native8_bf16_varlen_fp32_master',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    from reproduction.common8 import run_pipeline as native
    signal.signal(signal.SIGTERM,native.request_interrupt);signal.signal(signal.SIGINT,native.request_interrupt)
    from reproduction.common8.prepare_run import prepare
    # prepare() binds independent official actor/ref and new optimizer/RNG state.
    run=prepare(base,config['official_model'],config['official_manifest'],config['source_commit'],
        rollout_memory_utilization=.1,local_retrieval_assets=config['retrieval_assets'],execution_site='h1008',actor_micro_batch_per_rank=4,
        restore_receipt=config.get('restore_receipt'))
    from reproduction.common8.native_allocation import admit
    allocation=admit(base)
    # Some admission producers persist the authoritative object rather than
    # returning it; read the actual record consistently.
    allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
    locks=UUIDLocks(allocation['actual_cuda_runtime_uuids'])
    try:
        env=dict(os.environ,PATH=str(base/'env/bin')+':'+os.environ['PATH'],PYTHONPATH=str(base/'repo'),
            TMPDIR=str(base/'tmp'),HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
            TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),OMP_NUM_THREADS='8')
        for n in ('tmp','cache/hf','cache/triton','cache/inductor','artifacts'):(base/n).mkdir(parents=True,exist_ok=True)
        cache=base/'question-features'
        reused=config.get('precomputed_question_features')
        if reused:
            import hashlib
            from search_g1.qprobe_sequence import SequenceCache
            from search_g1.qprobe_features import sha_json, sha_file
            assert cache.resolve()==Path(reused['path']).resolve()
            assert sha_file(cache/'manifest.json')==reused['manifest_sha256']
            features=SequenceCache(cache)
            assert features.manifest['count']==169028
            assert features.spec['train_parquet_sha256']==run['input_sha256']['train.parquet']
            assert features.spec['encoder_revision']==run['model_revision']
            assert features.spec['encoder_files_sha256']==sha_json(run['initial_model_sha256'])
            assert features.spec['official_manifest_sha256']==sha_file(config['official_manifest'])
            once(base/'logs/sequence-feature-reuse.json',dict(source=str(cache.resolve()),
                manifest_sha256=reused['manifest_sha256'],actor_optimizer_state_inherited=False,
                full_sha_checked_by_binding=True,run_id=run['run_id']))
            del features
        else:
            command=[str(base/'env/bin/python'),'-m','search_g1.qprobe_prepare','precompute','--data',str(base/'data'),
                '--model',config['official_model'],'--official-manifest',config['official_manifest'],'--cache-dir',str(cache),
                '--sequence-probe','--device','cuda:0','--batch-size','8']
            once(base/'logs/sequence-precompute-intent.json',dict(command=command,run_id=run['run_id'],actual_actor_updates=0))
            from reproduction.a800.owned_processes import OwnedProcesses
            owned=OwnedProcesses(base/'logs/sequence-precompute-process-ledger.json',run['run_id'])
            try:
                assert native._interrupt_requested is None,'Interrupted before feature preparation'
                process=owned.start('sequence-precompute',command,env,base/'logs/sequence-precompute.log',base/'repo')
                while True:
                    owned.observe('sequence-precompute')
                    assert native._interrupt_requested is None,'Feature preparation interrupted; owned cleanup follows'
                    code=process.poll()
                    if code is not None:break
                    time.sleep(1)
                owned.jobs['sequence-precompute']['exit_code']=code;owned.save()
            finally:owned.stop_all()
            once(base/'logs/sequence-precompute-exit.json',dict(exit_code=code,run_id=run['run_id'],checked_unix=time.time()))
            assert code==0,'Preserve precompute logs/staging; do not silently replay'
        from reproduction.common8.qprobe_bind import bind
        from reproduction.common8.stage_retrieval import ASSETS
        from search_g1.qprobe_features import sha_json
        from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
        if not config.get('restore_receipt'):
            run=bind(base,cache,exploration_questions=64,retriever_version=sha_json(ASSETS),
                teacher_warmup_cache=config.get('frozen_teacher_cache',str(base/'teacher-warmup')),infra_rollout=DEFAULT_ROLLOUT_PHASE,
                sequence_device='allocated_rank7',sequence_train_micro=4,sequence_predict_micro=32)
        run['startup_admission_mode']='actual_first_five'
        from reproduction.a800.owned_processes import publish
        publish(base/'logs/run-manifest.json',run)
        from reproduction.common8.disk_admission import admit as disk_admit
        disk_admit(base)
        from reproduction.common8 import run_pipeline as native
        signal.signal(signal.SIGTERM,native.request_interrupt);signal.signal(signal.SIGINT,native.request_interrupt)
        with (base/'gpu-job.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            build_pipeline(base,run,config).execute(config.get('export_port',19874))
    finally:locks.close()

if __name__=='__main__':main()

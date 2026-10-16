"""Fresh question+20 group-joint probe on eight H100s, using retained trajectories."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

from reproduction.prefix20_online.launch import read, sha, once, publish

PREFIX='/mnt/data/shaoshitong/search-g1-question20-groupjoint100-'
ROLE='fresh_question20_group_joint_probe100'
BATCH_SEMANTICS='average255_260_complete_G5'


def validate_config(c,base):
    if c['base']!=str(base)or not str(base).startswith(PREFIX):
        raise ValueError('A separate remote group-joint namespace is required')
    if (c['epochs']!=100 or c['global_batch']!=256 or c['batch_semantics']!=BATCH_SEMANTICS
            or c['total_optimizer_updates']!=10000 or c['updates_per_epoch']!=100
            or c['lr_start']!=3e-5 or c['lr_end']!=3e-6 or c['group_weight']!=1
            or c['warmup_updates']!=0 or c['lr_schedule']!='cosine'
            or c['world_size']!=8 or c['accumulation']!=1 or c['fresh_head']is not True
            or c['encoding_mode']!='inline_per_training_batch_frozen_layer27'):
        raise ValueError('Authorized fresh joint100, complete-G5 batch, or cosine LR recipe differs')
    if any(key in c for key in ('resume_replay','source_checkpoint','source_adam_steps_per_rank')):
        raise ValueError('This experiment must not import an older head, Adam, RNG, or step')
    schedule=c['data_schedule']
    if len(schedule)!=100 or any(type(value)is not int or not 1<=value<=70 for value in schedule):
        raise ValueError('One retained source-batch identity is required for each of100 epochs')
    if schedule!=list(range(1,71))+list(range(1,31)):
        raise ValueError('The confirmed retained data schedule is batches1..70 followed by1..30')
    if len(c['physical_gpu_uuids'])!=8 or len(set(c['physical_gpu_uuids']))!=8:
        raise ValueError('Eight distinct actual physical GPU UUIDs required')
    if len(c['run_id'])!=32 or any(ch not in'0123456789abcdef'for ch in c['run_id']):
        raise ValueError('A new concrete run UUID is required')
    if c['question_max_tokens']!=273 or c['head_max_tokens']!=293:
        raise ValueError('Retained question-plus20 feature layout must remain Q273/S293')
    return c


def config(base):
    base=Path(base).resolve()
    return validate_config(read(base/'preparation/launch-config.json'),base)


def sbatch_command(base,c):
    return ['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8',
        '--nodes=1','--ntasks=1','--cpus-per-task=96','--mem=512G','--gres=gpu:8','--time=0',
        '--signal=B:TERM@60','--export=NONE','--job-name=g1-q20-joint100-'+c['run_id'][:12],
        '--comment=g1-q20-joint100:'+c['run_id'],'--chdir='+str(base),
        '--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),
        str(base/'preparation/native-launch.sh')]


def query(base):
    from reproduction.common8.native_allocation import fields
    base=Path(base).resolve();intent_path=base/'preparation/submit-intent.json'
    if not intent_path.exists():return dict(status='NOT_SUBMITTED',submit_attempted=False)
    intent=read(intent_path);receipt_path=base/'preparation/submit-receipt.json'
    receipt=read(receipt_path)if receipt_path.exists()else None
    if receipt and receipt['intent_sha256']!=sha(intent_path):raise ValueError('Submission identity changed')
    queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True,timeout=30)
    jobs=[line.split('|')[0]for line in queue.splitlines()if line.split('|')[-1]==intent['job_name']]
    if receipt and receipt.get('job_id')and receipt['job_id']not in jobs:jobs.append(receipt['job_id'])
    matching=[]
    for job in jobs:
        try:raw=subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True,stderr=subprocess.PIPE,timeout=30)
        except subprocess.CalledProcessError:continue
        actual=fields(raw)
        if (actual.get('JobName')==intent['job_name']and actual.get('Comment')==intent['comment']
                and actual.get('Command')==str(base/'preparation/native-launch.sh')and actual.get('WorkDir')==str(base)):
            matching.append(dict(job_id=str(job),state=actual.get('JobState'),exit_code=actual.get('ExitCode'),native=raw))
    if len(matching)>1:raise RuntimeError('Multiple matching native jobs require inspection')
    return dict(status='RECONCILED_EXISTING_JOB'if matching else'UNKNOWN_NO_REPLAY',receipt=receipt,
                matching_jobs=matching,submit_attempted=True,replay_allowed=False)


def submit(base):
    base=Path(base).resolve();prep=base/'preparation'
    with (prep/'submit-control.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (prep/'submit-intent.json').exists():return query(base)
        c=config(base);launch=prep/'native-launch.sh'
        if launch.exists()or(prep/'submit-receipt.json').exists():raise RuntimeError('Partial submission requires reconciliation')
        env_root=Path(c['environment_root'])
        lines=['#!/bin/bash','set -euo pipefail',
            'export PATH='+shlex.quote(str(env_root/'bin')+':/usr/local/bin:/usr/bin:/bin'),
            'export PYTHONPATH='+shlex.quote(str(base/'repo')),
            'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
            'unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy',
            'exec '+shlex.join([str(env_root/'bin/python'),'-m','reproduction.prefix20_group_joint100.launch','run','--base',str(base)])]
        with launch.open('x')as handle:
            handle.write('\n'.join(lines)+'\n');handle.flush();os.fsync(handle.fileno())
        launch.chmod(0o700);command=sbatch_command(base,c)
        intent=dict(role=ROLE,base=str(base),run_id=c['run_id'],source_commit=c['source_commit'],
            config_sha256=sha(prep/'launch-config.json'),command=command,
            job_name='g1-q20-joint100-'+c['run_id'][:12],comment='g1-q20-joint100:'+c['run_id'],created_unix=time.time())
        once(prep/'submit-intent.json',intent)
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                status='REJECTED_NO_REPLAY',intent_sha256=sha(prep/'submit-intent.json'))
            if result.returncode==0:
                job=result.stdout.strip().split(';')[0];receipt['status']='SUBMITTED'if job.isdecimal()else'UNKNOWN_NO_REPLAY'
                if job.isdecimal():receipt['job_id']=job
        except(subprocess.TimeoutExpired,OSError)as error:
            receipt=dict(status='UNKNOWN_NO_REPLAY',error=repr(error),intent_sha256=sha(prep/'submit-intent.json'))
        once(prep/'submit-receipt.json',receipt);return receipt


def native_allocation(base,c):
    from reproduction.common8.native_allocation import fields
    from reproduction.common8.stage_retrieval import memory_bytes
    if socket.gethostname().split('.')[0]!='h100-8'or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Actual native h100-8 allocation required')
    job=os.environ['SLURM_JOB_ID'];raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=30)
    actual=fields(raw)
    if (actual.get('JobId')!=job or actual.get('JobState')!='RUNNING'or actual.get('NodeList')!='h100-8'
            or int(actual['NumCPUs'])<96 or memory_bytes(actual['TRES'])<512*2**30
            or actual.get('TresPerNode')not in('gres:gpu:8','gres:gpu:h100:8')):
        raise ValueError('Actual eight-GPU CPU/RAM allocation differs')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=30).splitlines()
    mapping={int(row.split(',')[0]):row.split(',')[1].strip()for row in rows}
    if [mapping[index]for index in range(8)]!=c['physical_gpu_uuids']:raise ValueError('Physical GPU UUID mapping differs')
    once(base/'logs/native-allocation.json',dict(run_id=c['run_id'],job_id=job,native=raw,
        physical_gpu_uuids=c['physical_gpu_uuids'],checked_unix=time.time()))


def trainer_command(python,c,base,epoch,train_inputs,test_features,test_sha):
    source_epoch=c['data_schedule'][epoch-1]
    return [python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m',
        'reproduction.prefix20_group_joint100.train_epoch','--train-inputs',str(train_inputs),
        '--test-cache',str(test_features),'--test-manifest-sha',test_sha,'--model-dir',c['model_dir'],
        '--output',str(base/'train'),'--epoch',str(epoch),'--source-data-epoch',str(source_epoch),
        '--epochs','100','--global-batch','256','--batch-semantics',BATCH_SEMANTICS,
        '--lr-start',str(c['lr_start']),'--lr-end',str(c['lr_end']),
        '--group-weight',str(c['group_weight']),'--total-optimizer-updates','10000',
        '--save-every','10','--encoder-micro-batch-size',str(c.get('encoder_micro_batch',32)),
        '--seed',str(c['seed'])]


def run(base):
    import shutil
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class,DeferredInterrupt,birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base=Path(base).resolve();c=config(base);prep=base/'preparation'
    if read(prep/'submit-intent.json')['config_sha256']!=sha(prep/'launch-config.json'):raise ValueError('Submitted config changed')
    if (base/'logs/process-ledger.json').exists():raise ValueError('Existing execution requires inspection, never replay')
    native_allocation(base,c);lease=UUIDLocks(c['physical_gpu_uuids'])
    owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id']);interrupt=DeferredInterrupt()
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGINT,signal.SIGTERM)}
    for sig in handlers:signal.signal(sig,interrupt.handler)
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),OMP_NUM_THREADS='8',
        CUDA_VISIBLE_DEVICES=','.join(c['physical_gpu_uuids']),NCCL_NVLS_ENABLE='0',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        TMPDIR=str(base/'tmp'),PYTHONUNBUFFERED='1')
    for key in('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    for name in('tmp','cache/hf','cache/triton','cache/inductor'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python');epoch=0;success=False
    def stage(name,**extra):
        value=dict(run_id=c['run_id'],stage=name,epoch=epoch,checked_unix=time.time(),**extra)
        publish(base/'logs/pipeline-status.json',value);print('QUESTION20_GROUPJOINT='+json.dumps(value),flush=True)
    def phase(name,command):
        interrupt.check();stage(name)
        process=owned.start(name,command,env,str(base/'logs'/(name+'.log')),str(base/'repo'))
        while process.poll()is None:owned.observe(name);interrupt.check();time.sleep(2)
        owned.observe(name);owned.jobs[name]['exit_code']=process.returncode;owned.save()
        if process.returncode:raise RuntimeError('Actual phase failed: '+name)
        owned.stop(name);interrupt.check()
    try:
        test_features=Path(c['fixed_test_features_path']);test_sha=sha(test_features/'manifest.json')
        if test_sha!=c['fixed_test_manifest_sha256']:raise ValueError('Fixed test feature manifest changed')
        inventory_path=prep/'source-inventory.json'
        if sha(inventory_path)!=c['source_inventory_sha256']:raise ValueError('Retained source inventory changed')
        sources={int(row['epoch']):row for row in read(inventory_path)['sources']if row['split']=='train'}
        if set(sources)!=set(range(1,71)):raise ValueError('All seventy retained source batches are required')
        stage('FRESH_PROBE_FIXED_CACHE_BOUND',question_max_tokens=273,head_max_tokens=293,
            old_head_state_imported=False,fixed_test_manifest_sha256=test_sha)
        module='reproduction.prefix20_question_replay.features'
        for epoch in range(1,101):
            source_epoch=c['data_schedule'][epoch-1];source=sources[source_epoch]
            root=base/'epochs'/('epoch-%04d'%epoch);root.mkdir(parents=True)
            args=(['--prepare-inputs',source['inputs'],'--split','train','--epoch',str(source_epoch)]
                if'inputs'in source else['--rollout-complete',source['rollout_complete']])
            phase('e%04d-inputs'%epoch,[python,'-m',module]+args+['--output',str(root/'inputs')])
            phase('e%04d-train-inline-encode'%epoch,trainer_command(python,c,base,epoch,root/'inputs',test_features,test_sha))
            latest=read(base/'train/latest.json');expected=[100*epoch]*8
            if (latest['epoch']!=epoch or latest['actual_adam_steps_per_rank']!=expected
                    or latest['new_adam_steps_per_rank']!=100):raise RuntimeError('Actual complete epoch or Adam budget differs')
            metrics=read(base/'train/epochs'/('epoch-%04d'%epoch)/'metrics.json')
            bundle=dict(epoch=epoch,source_data_epoch=source_epoch,run_id=c['run_id'],source=source,
                question_max_tokens=273,head_max_tokens=293,train_scan_sha256=sha(root/'inputs/scan-complete.json'),
                train_inputs_scan=read(root/'inputs/scan-complete.json'),test_manifest_sha256=test_sha,
                encoding_mode=c['encoding_mode'],training_feature_disk_cache_created=False,
                checkpoint=latest,config_sha256=sha(prep/'launch-config.json'),new_rollouts=0)
            once(root/'data-bundle.json',bundle)
            stage('GROUP_JOINT_EPOCH_COMPLETE',actual_adam_steps_per_rank=expected,
                actual_optimizer_learning_rates=latest['actual_optimizer_learning_rates'],
                train_mixed_group_accuracy=metrics['train_mixed_group_accuracy'],
                test_mixed_group_accuracy=metrics['test_mixed_group_accuracy'])
            removed={}
            for name in('inputs','inputs-policy-source'):
                path=root/name
                if path.exists():
                    if path.is_symlink():raise ValueError('Derived cache path is a symlink')
                    removed[name]=sum(item.stat().st_size for item in path.rglob('*')if item.is_file());shutil.rmtree(path)
            once(root/'derived-cache-pruned.json',dict(removed_bytes=removed,source_trajectories_preserved=True,
                durable_checkpoint_sha256=latest['sha256'],checked_unix=time.time()))
        success=True
    except BaseException as error:
        stage('FAILED',error_type=type(error).__name__,error=str(error));raise
    finally:
        try:
            owned.stop_all();owned.reap();births=[row for job in owned.jobs.values()for row in job['ledger'].values()]
            present=birth_states(births)
            once(base/'logs/cleanup-status.json',dict(run_id=c['run_id'],cleaned=not any(row['state']!='Z'for row in present),
                owned_births_all_gone=not present,remaining_owned_births=[row['pid']for row in present],checked_unix=time.time()))
        finally:
            lease.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if success:once(base/'logs/pipeline-complete.json',dict(run_id=c['run_id'],epochs=100,
        latest=read(base/'train/latest.json'),new_rollouts=0,actor_updates=0,old_head_state_imported=False,
        native_exit0_requires_post_exit_observation=True))


def observe(base):
    from reproduction.prefix20_online.observe import small,tail_json
    from reproduction.common8.native_allocation import fields
    base=Path(base);receipt=small(base/'preparation/submit-receipt.json');stage=small(base/'logs/pipeline-status.json')or{}
    epoch=int(stage.get('epoch',0));root=base/'train/epochs'/('epoch-%04d'%epoch)
    value=dict(base=str(base),observed_unix=time.time(),stage=stage,latest=small(base/'train/latest.json'),
        metrics=tail_json(base/'train/metrics.jsonl'),
        rank_updates={str(rank):tail_json(root/('rank-%d-updates.jsonl'%rank))for rank in range(8)},
        rank_errors={str(rank):small(root/('rank-%d-error.json'%rank))for rank in range(8)},
        inline_progress={str(rank):small(root/('progress-rank-%d.json'%rank))for rank in range(8)},
        cleanup=small(base/'logs/cleanup-status.json'),pipeline_complete=small(base/'logs/pipeline-complete.json'))
    if receipt and receipt.get('job_id'):
        try:value['native']=fields(subprocess.check_output(['scontrol','show','job',str(receipt['job_id']),'-o'],text=True,timeout=30))
        except subprocess.CalledProcessError as error:value['native_unavailable']=str(error)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('run','submit','query','observe'));parser.add_argument('--base',required=True)
    args=parser.parse_args();result={'run':run,'submit':submit,'query':query,'observe':observe}[args.phase](args.base)
    if result is not None:print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()

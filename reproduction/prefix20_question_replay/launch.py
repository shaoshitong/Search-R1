"""Eight-GPU cached question+20 continuation; no rollout or retrieval service."""
import argparse,fcntl,json,os,shlex,signal,socket,subprocess,time
from pathlib import Path
from reproduction.prefix20_online.launch import read,sha,once,publish
ROLE='question_plus20_cached_replay70'

def config(base):
 base=Path(base).resolve();c=read(base/'preparation/launch-config.json')
 if c['base']!=str(base) or not str(base).startswith('/mnt/data/shaoshitong/search-g1-question20-replay70-'):
  raise ValueError('Separate remote replay namespace required')
 if c['iterations']!=70 or c['global_batch']!=256 or c['learning_rate']!=3e-6:
  raise ValueError('Latest authorized replay70/BS256/LR3e-6 contract differs')
 if len(c['physical_gpu_uuids'])!=8 or len(set(c['physical_gpu_uuids']))!=8:
  raise ValueError('Eight actual GPU UUIDs required')
 return c

def sbatch_command(base, c):
    return ['sbatch', '--parsable', '--partition=gpu', '--oversubscribe', '--nodelist=h100-8',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=96', '--mem=512G', '--gres=gpu:8',
            '--time=0', '--signal=B:TERM@60', '--export=NONE',
            '--job-name=g1-q20-replay70-'+c['run_id'][:12],
            '--comment=g1-q20-replay70:'+c['run_id'], '--chdir='+str(base),
            '--output='+str(base/'logs/native-%j.log'), '--error='+str(base/'logs/native-%j.log'),
            str(base/'preparation/native-launch.sh')]


def query(base):
    from reproduction.common8.native_allocation import fields
    base = Path(base).resolve()
    p = base/'preparation/submit-intent.json'
    if not p.exists():
        return dict(status='NOT_SUBMITTED', submit_attempted=False)
    intent = read(p)
    receipt_path = base/'preparation/submit-receipt.json'
    receipt = read(receipt_path) if receipt_path.exists() else None
    if receipt and receipt['intent_sha256'] != sha(p):
        raise ValueError('Submit receipt identity differs')
    queue = subprocess.check_output(['squeue', '-h', '-o', '%i|%j'], text=True, timeout=30)
    jobs = [line.split('|')[0] for line in queue.splitlines()
            if line.split('|')[-1] == intent['job_name']]
    if receipt and receipt.get('job_id') and receipt['job_id'] not in jobs:
        jobs.append(receipt['job_id'])
    matches = []
    for job in jobs:
        try:
            raw = subprocess.check_output(['scontrol', 'show', 'job', str(job), '-o'],
                                          text=True, stderr=subprocess.PIPE, timeout=30)
        except subprocess.CalledProcessError:
            continue
        actual = fields(raw)
        if (actual.get('JobName') == intent['job_name'] and actual.get('Comment') == intent['comment']
                and actual.get('Command') == str(base/'preparation/native-launch.sh')
                and actual.get('WorkDir') == str(base)):
            matches.append(dict(job_id=str(job), state=actual.get('JobState'),
                                exit_code=actual.get('ExitCode'), native=raw))
    if len(matches) > 1:
        raise RuntimeError('Multiple matching native jobs; inspect without replay')
    return dict(status='RECONCILED_EXISTING_JOB' if matches else 'UNKNOWN_NO_REPLAY',
                submit_attempted=True, receipt=receipt, matching_jobs=matches, replay_allowed=False)


def submit(base):
    base = Path(base).resolve()
    prep = base/'preparation'
    with (prep/'submit-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (prep/'submit-intent.json').exists():
            return query(base)
        c = config(base)
        launch = prep/'native-launch.sh'
        if launch.exists() or (prep/'submit-receipt.json').exists():
            raise RuntimeError('Partial submission needs inspection, not replay')
        env_root = Path(c['environment_root'])
        lines = ['#!/bin/bash', 'set -euo pipefail',
                 'export PATH='+shlex.quote(str(env_root/'bin')+':/usr/local/bin:/usr/bin:/bin'),
                 'export PYTHONPATH='+shlex.quote(str(base/'repo')),
                 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
                 'unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy',
                 'exec '+shlex.join([str(env_root/'bin/python'), '-m',
                     'reproduction.prefix20_question_replay.launch', 'run', '--base', str(base)])]
        with launch.open('x') as handle:
            handle.write('\n'.join(lines)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        launch.chmod(0o700)
        command = sbatch_command(base, c)
        intent = dict(role=ROLE, base=str(base), run_id=c['run_id'], source_commit=c['source_commit'],
                      config_sha256=sha(prep/'launch-config.json'), command=command,
                      job_name='g1-q20-replay70-'+c['run_id'][:12], comment='g1-q20-replay70:'+c['run_id'],
                      created_unix=time.time())
        once(prep/'submit-intent.json', intent)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            receipt = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                           status='REJECTED_NO_REPLAY', intent_sha256=sha(prep/'submit-intent.json'))
            if result.returncode == 0:
                job = result.stdout.strip().split(';')[0]
                receipt.update(status='SUBMITTED' if job.isdecimal() else 'UNKNOWN_NO_REPLAY')
                if job.isdecimal():
                    receipt['job_id'] = job
        except (subprocess.TimeoutExpired, OSError) as error:
            receipt = dict(status='UNKNOWN_NO_REPLAY', error=repr(error),
                           intent_sha256=sha(prep/'submit-intent.json'))
        once(prep/'submit-receipt.json', receipt)
        return receipt


def native_allocation(base, c):
    from reproduction.common8.native_allocation import fields
    from reproduction.common8.stage_retrieval import memory_bytes
    if socket.gethostname().split('.')[0] != 'h100-8' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Native h100-8 allocation required')
    job = os.environ['SLURM_JOB_ID']
    raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=30)
    actual = fields(raw)
    if (actual.get('JobId') != job or actual.get('JobState') != 'RUNNING'
            or actual.get('NodeList') != 'h100-8' or int(actual['NumCPUs']) < 96
            or memory_bytes(actual['TRES']) < 512*2**30
            or actual.get('TresPerNode') not in ('gres:gpu:8', 'gres:gpu:h100:8')):
        raise ValueError('Actual native eight-GPU CPU/RAM allocation differs')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
            '--format=csv,noheader,nounits'], text=True, timeout=30).splitlines()
    mapping = {int(row.split(',')[0]):row.split(',')[1].strip() for row in rows}
    if [mapping[i] for i in range(8)] != c['physical_gpu_uuids']:
        raise ValueError('Actual physical GPU UUID mapping differs')
    value = dict(run_id=c['run_id'], job_id=job, native=raw,
                 physical_gpu_uuids=c['physical_gpu_uuids'], checked_unix=time.time())
    once(base/'logs/native-allocation.json', value)
    return value


def run(base):
    import shutil
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base=Path(base).resolve();c=config(base);prep=base/'preparation'
    if read(prep/'submit-intent.json')['config_sha256']!=sha(prep/'launch-config.json'):
        raise ValueError('Submitted configuration changed')
    if (base/'logs/process-ledger.json').exists():
        raise ValueError('Existing execution must be inspected, never blindly replayed')
    native_allocation(base,c)
    lease=UUIDLocks(c['physical_gpu_uuids'])
    owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt()
    handlers={sig:signal.getsignal(sig) for sig in(signal.SIGINT,signal.SIGTERM)}
    for sig in handlers:signal.signal(sig,interrupt.handler)
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),OMP_NUM_THREADS='8',
        CUDA_VISIBLE_DEVICES=','.join(c['physical_gpu_uuids']),NCCL_NVLS_ENABLE='0',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        TMPDIR=str(base/'tmp'),PYTHONUNBUFFERED='1')
    for k in('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(k,None)
    for name in('tmp','cache/hf','cache/triton','cache/inductor'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python');iteration=0;success=False
    def stage(name,**extra):
        value=dict(run_id=c['run_id'],stage=name,replay_iteration=iteration,checked_unix=time.time(),**extra)
        publish(base/'logs/pipeline-status.json',value)
        print('QUESTION20_REPLAY='+json.dumps(value),flush=True)
    def phase(name,command):
        interrupt.check();stage(name)
        process=owned.start(name,command,env,str(base/'logs'/(name+'.log')),str(base/'repo'))
        while process.poll() is None:
            owned.observe(name);interrupt.check();time.sleep(2)
        owned.observe(name);owned.jobs[name]['exit_code']=process.returncode;owned.save()
        if process.returncode:raise RuntimeError('Actual phase failed: '+name)
        owned.stop(name);interrupt.check()
    def torchrun(module,args):
        return [python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m',module]+args
    module='reproduction.prefix20_question_replay.features'
    try:
        phase('FIXED_QUESTION_LENGTH_SCAN',[python,'-m',module,'--question-max-sources',str(prep/'replay-sources.json'),
            '--output',str(prep/'question-max.json')])
        qmax=read(prep/'question-max.json')['question_max_tokens']
        stage('QUESTION_WIDTH_FIXED',question_max_tokens=qmax,head_max_tokens=qmax+20)
        test=base/'fixed-test';test.mkdir()
        phase('FIXED_TEST_INPUTS',[python,'-m',module,'--prepare-inputs',c['cached_inputs'],
            '--split','test','--output',str(test/'inputs')])
        encode=['--model',c['model_dir'],'--question-max-tokens',str(qmax),
            '--micro-batch-size',str(c.get('encoder_micro_batch',32))]
        phase('FIXED_TEST_FEATURES',torchrun(module,['--inputs',str(test/'inputs'),'--output',str(test/'features')]+encode))
        test_sha=sha(test/'features/manifest.json')
        sources=read(prep/'replay-sources.json')['sources']
        train_sources={int(x['epoch']):x for x in sources if x['split']=='train'}
        if set(train_sources)!=set(range(1,71)):raise ValueError('Exactly the saved seventy data batches required')
        for iteration in range(1,71):
            root=base/'iterations'/('iteration-%04d'%iteration);root.mkdir(parents=True)
            source=train_sources[iteration]
            if 'inputs' in source:
                args=['--prepare-inputs',source['inputs'],'--split','train','--epoch',str(iteration)]
            else:args=['--rollout-complete',source['rollout_complete']]
            phase('i%04d-inputs'%iteration,[python,'-m',module]+args+['--output',str(root/'inputs')])
            phase('i%04d-features'%iteration,torchrun(module,['--inputs',str(root/'inputs'),
                '--output',str(root/'features')]+encode))
            phase('i%04d-train'%iteration,torchrun('reproduction.prefix20_question_replay.train_epoch',[
                '--train-cache',str(root/'features'),'--test-cache',str(test/'features'),
                '--test-manifest-sha',test_sha,'--model-dir',c['model_dir'],'--output',str(base/'train'),
                '--replay-iteration',str(iteration),'--source-data-epoch',str(iteration),
                '--source-epoch',str(c['source_epoch']),'--source-checkpoint',c['source_checkpoint'],
                '--source-checkpoint-sha',c['source_checkpoint_sha256'],'--learning-rate',str(c['learning_rate']),
                '--seed',str(c['seed'])]))
            latest=read(base/'train/latest.json')
            expected=[x+100*iteration for x in c['source_adam_steps_per_rank']]
            if latest['replay_iteration']!=iteration or latest['actual_adam_steps_per_rank']!=expected:
                raise RuntimeError('Actual complete replay iteration or eight-rank Adam boundary differs')
            if latest['actual_optimizer_learning_rates']!=[c['learning_rate']]:raise RuntimeError('Actual learning rate differs')
            bundle=dict(replay_iteration=iteration,source_data_epoch=iteration,source_epoch=c['source_epoch'],
                run_id=c['run_id'],source=source,question_max_tokens=qmax,
                train_scan_sha256=sha(root/'inputs/scan-complete.json'),
                train_feature_manifest=read(root/'features/manifest.json'),test_manifest_sha256=test_sha,
                checkpoint=latest,config_sha256=sha(prep/'launch-config.json'),new_rollouts=0)
            once(root/'data-bundle.json',bundle)
            stage('REPLAY_ITERATION_COMPLETE',actual_adam_steps_per_rank=expected,
                mixed_group_accuracy=read(base/'train/iterations'/('iteration-%04d'%iteration)/'metrics.json')['mixed_group_accuracy'])
            removed={}
            for name in('features','inputs','inputs-policy-source'):
                p=root/name
                if p.exists():
                    if p.is_symlink():raise ValueError('Derived data path is a symlink')
                    removed[name]=sum(x.stat().st_size for x in p.rglob('*')if x.is_file())
                    shutil.rmtree(p)
            once(root/'derived-cache-pruned.json',dict(removed_bytes=removed,source_trajectories_preserved=True,
                durable_checkpoint_sha256=latest['sha256'],checked_unix=time.time()))
        success=True
    except BaseException as error:
        stage('FAILED',error_type=type(error).__name__,error=str(error));raise
    finally:
        try:
            owned.stop_all();owned.reap()
            births=[r for j in owned.jobs.values()for r in j['ledger'].values()]
            present=birth_states(births)
            once(base/'logs/cleanup-status.json',dict(run_id=c['run_id'],cleaned=not any(r['state']!='Z' for r in present),
                owned_births_all_gone=not present,remaining_owned_births=[r['pid']for r in present],checked_unix=time.time()))
        finally:
            lease.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if success:once(base/'logs/pipeline-complete.json',dict(run_id=c['run_id'],replay_iterations=70,
        latest=read(base/'train/latest.json'),new_rollouts=0,actor_updates=0,native_exit0_requires_post_exit_observation=True))


def observe(base):
    from reproduction.prefix20_online.observe import small,tail_json
    from reproduction.common8.native_allocation import fields
    base=Path(base);receipt=small(base/'preparation/submit-receipt.json');stage=small(base/'logs/pipeline-status.json')or{}
    i=int(stage.get('replay_iteration',0));root=base/'train/iterations'/('iteration-%04d'%i)
    result=dict(base=str(base),observed_unix=time.time(),stage=stage,latest=small(base/'train/latest.json'),
        metrics=tail_json(base/'train/metrics.jsonl'),
        rank_updates={str(r):tail_json(root/('rank-%d-updates.jsonl'%r))for r in range(8)},
        rank_errors={str(r):small(root/('rank-%d-error.json'%r))for r in range(8)},
        feature_progress={str(r):small(base/'iterations'/('iteration-%04d'%i)/'features'/('progress-rank-%d.json'%r))for r in range(8)},
        question_max=small(base/'preparation/question-max.json'),cleanup=small(base/'logs/cleanup-status.json'),
        pipeline_complete=small(base/'logs/pipeline-complete.json'))
    if receipt and receipt.get('job_id'):
        try:result['native']=fields(subprocess.check_output(['scontrol','show','job',str(receipt['job_id']),'-o'],text=True,timeout=30))
        except subprocess.CalledProcessError as e:result['native_unavailable']=str(e)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','submit','query','observe'));p.add_argument('--base',required=True)
    a=p.parse_args();result={'run':run,'submit':submit,'query':query,'observe':observe}[a.phase](a.base)
    if result is not None:print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()



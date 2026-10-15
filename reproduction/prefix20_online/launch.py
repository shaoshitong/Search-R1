"""One native eight-GPU job: cached epoch1, then fresh rollout/encode/fit epochs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time

ROLE = 'prefix20_online_fresh1000'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def publish(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial-' + str(os.getpid()))
    once(temp, value)
    os.replace(temp, path)


def config(base):
    base = Path(base).resolve()
    c = read(base/'preparation/launch-config.json')
    if (c['base'] != str(base) or not str(base).startswith('/mnt/data/shaoshitong/search-g1-prefix20-online1000-')
            or not re.fullmatch('[0-9a-f]{32}', c['run_id'])
            or not re.fullmatch('[0-9a-f]{40}', c['source_commit'])):
        raise ValueError('Explicit new remote run identity required')
    if len(c['physical_gpu_uuids']) != 8 or len(set(c['physical_gpu_uuids'])) != 8:
        raise ValueError('Eight unique actual GPU UUIDs required')
    for key in ('environment_root', 'model_dir', 'cached_features', 'cached_inputs',
                'canonical_train', 'retrieval_index', 'retrieval_corpus',
                'retrieval_encoder', 'retriever_gpu_lib'):
        if not Path(c[key]).is_absolute():
            raise ValueError('Remote absolute path required: '+key)
    return c


def sbatch_command(base, c):
    return ['sbatch', '--parsable', '--partition=gpu', '--oversubscribe', '--nodelist=h100-8',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=96', '--mem=512G', '--gres=gpu:8',
            '--time=0', '--signal=B:TERM@60', '--export=NONE',
            '--job-name=g1-prefix20-online-'+c['run_id'][:12],
            '--comment=g1-prefix20-online:'+c['run_id'], '--chdir='+str(base),
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
                     'reproduction.prefix20_online.launch', 'run', '--base', str(base)])]
        with launch.open('x') as handle:
            handle.write('\n'.join(lines)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        launch.chmod(0o700)
        command = sbatch_command(base, c)
        intent = dict(role=ROLE, base=str(base), run_id=c['run_id'], source_commit=c['source_commit'],
                      config_sha256=sha(prep/'launch-config.json'), command=command,
                      job_name='g1-prefix20-online-'+c['run_id'][:12], comment='g1-prefix20-online:'+c['run_id'],
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


def prune_rebuildable(base, epoch):
    """Only this completed epoch's derived input/features; keep questions/raw."""
    import shutil
    root = base/'epochs'/('epoch-%04d' % epoch)
    latest = read(base/'train/latest.json')
    complete = read(base/'train/epochs'/('epoch-%04d' % epoch)/'complete.json')
    if latest['epoch'] != epoch or complete['sha256'] != latest['sha256']:
        raise ValueError('Derived cache pruning requires the durable current head boundary')
    if not (root/'complete.json').is_file():
        raise ValueError('Cannot prune before compact rollout publication')
    removed = {}
    for name in ('inputs', 'features'):
        path = root/name
        if path.exists():
            if path.is_symlink() or any(item.is_symlink() for item in path.rglob('*')):
                raise ValueError('Derived owned cache unexpectedly contains symlinks')
            removed[name] = sum(item.stat().st_size for item in path.rglob('*') if item.is_file())
            shutil.rmtree(path)
    once(root/'derived-cache-pruned.json', dict(epoch=epoch, removed_bytes=removed,
         durable_head_sha256=latest['sha256'], questions_and_compact_retained=True,
         full_raw_retained=True, checked_unix=time.time()))
    # Full continuations are optional after their own compact20/EM and head
    # boundary are durable. Keep them while ordinary available space permits.
    if shutil.disk_usage(base).free < 128*2**30:
        for candidate in sorted((base/'epochs').glob('epoch-*')):
            if shutil.disk_usage(base).free >= 192*2**30:
                break
            receipt_path = candidate/'complete.json'
            deleted_path = candidate/'full-raw-pruned.json'
            if not receipt_path.is_file() or deleted_path.exists():
                continue
            record = read(receipt_path)
            prior_epoch = int(record['epoch'])
            boundary = base/'train/epochs'/('epoch-%04d' % prior_epoch)/'complete.json'
            if prior_epoch > epoch or not boundary.is_file():
                continue
            compact = record.get('shards', [])
            if not compact or not all(Path(item['path']).is_file() and
                    Path(item['path']).stat().st_size == item['bytes'] for item in compact):
                continue
            if not (candidate/'questions.json').is_file():
                continue
            deleted = []
            for item in record.get('raw_shards', []):
                path = Path(item['path'])
                if (path.parent.resolve() != (candidate/'rollout').resolve() or path.is_symlink()
                        or not path.is_file() or path.stat().st_size != item['bytes']):
                    continue
                path.unlink()
                deleted.append(dict(path=str(path), bytes=item['bytes'],
                                    producer_sha256=item['sha256']))
            once(deleted_path, dict(epoch=prior_epoch, deleted=deleted,
                 trigger='ordinary_available_below128GiB', compact_and_questions_retained=True,
                 head_boundary_sha256=read(boundary)['sha256'],
                 full_raw_independently_rehashed_before_delete=False, checked_unix=time.time()))


def snapshot_pool(base, epoch, c, latest):
    root = base/'epochs'/('epoch-%04d' % epoch)
    root.mkdir(parents=True, exist_ok=True)
    state_path = base/'question-pool/state.json'
    state = read(state_path)
    if state['next_epoch'] != epoch+1:
        raise RuntimeError('Question pool boundary differs from completed head epoch')
    snapshot = root/'pool-state-at-head-completion.json'
    once(snapshot, state)
    bundle = dict(epoch=epoch, run_id=c['run_id'],
         head_path=str(base/'train/latest.pt'), head_sha256=latest['sha256'],
         actual_adam_steps_per_rank=latest['actual_adam_steps_per_rank'],
         pool_state_path=str(snapshot), pool_state_sha256=sha(snapshot),
         pool_json_sha256=sha(base/'question-pool/pool.json'),
         config_sha256=sha(base/'preparation/launch-config.json'))
    for key, name in (('selection_sha256','selection.json'), ('rollout_complete_sha256','complete.json')):
        if (root/name).is_file():
            bundle[key] = sha(root/name)
    if epoch % int(c.get('save_every',10)) == 0 or epoch == int(c.get('epochs',1000)):
        bundle['head_path'] = str(base/'train'/('head-epoch-%04d.pt' % epoch))
        once(base/'train'/('head-epoch-%04d-bundle.json' % epoch), bundle)
    publish(base/'train/latest-bundle.json', bundle)


def run(base):
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base = Path(base).resolve()
    c = config(base)
    if read(base/'preparation/submit-intent.json')['config_sha256'] != sha(base/'preparation/launch-config.json'):
        raise ValueError('Submitted config changed')
    if (base/'logs/process-ledger.json').exists():
        raise ValueError('Existing run must be inspected and explicitly resumed, never replayed')
    native_allocation(base, c)
    lease = UUIDLocks(c['physical_gpu_uuids'])
    owned = torchrun_owned_class()(base/'logs/process-ledger.json', c['run_id'])
    interrupt = DeferredInterrupt()
    handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupt.handler)
    env = dict(os.environ, PYTHONPATH=str(base/'repo'), OMP_NUM_THREADS='8',
               CUDA_VISIBLE_DEVICES=','.join(c['physical_gpu_uuids']), NCCL_NVLS_ENABLE='0',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HOME=str(base/'cache/hf'), XDG_CACHE_HOME=str(base/'cache'),
               TRITON_CACHE_DIR=str(base/'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
               TMPDIR=str(base/'tmp'), PYTHONUNBUFFERED='1', NO_PROXY='127.0.0.1,localhost')
    for name in ('PYTHONHOME', 'RAY_ADDRESS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(name, None)
    for name in ('tmp', 'cache/hf', 'cache/triton', 'cache/inductor'):
        (base/name).mkdir(parents=True, exist_ok=True)
    python = str(Path(c['environment_root'])/'bin/python')
    retriever = None
    epoch = 0
    success = False

    def stage(value, **extra):
        status = dict(run_id=c['run_id'], stage=value, epoch=epoch, checked_unix=time.time(), **extra)
        publish(base/'logs/pipeline-status.json', status)
        print('PREFIX20_PIPELINE='+json.dumps(status), flush=True)

    def start(name, command, child_env=None):
        interrupt.check()
        process = owned.start(name, command, child_env or env,
                              str(base/'logs'/(name+'.log')), str(base/'repo'))
        owned.observe(name)
        return process

    def wait_stage(name, process):
        while process.poll() is None:
            owned.observe(name)
            interrupt.check()
            if retriever is not None:
                owned.observe('retriever')
                if retriever.poll() is not None:
                    raise RuntimeError('Own retriever exited; preserve logs')
            time.sleep(5)
        owned.observe(name)
        owned.jobs[name]['exit_code'] = process.returncode
        owned.save()
        if process.returncode != 0:
            raise RuntimeError('Actual phase failed: '+name)
        owned.stop(name)
        interrupt.check()

    def phase(name, command, child_env=None):
        stage(name)
        process = start(name, command, child_env)
        wait_stage(name, process)

    def torchrun(module, args):
        return [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                '-m', module] + args

    try:
        for epoch in range(1, int(c.get('epochs', 1000))+1):
            root = base/'epochs'/('epoch-%04d' % epoch)
            phase('e%04d-rollout' % epoch, [python, '-m', 'reproduction.prefix20_online.rollout',
                  '--config', str(base/'preparation/launch-config.json'), '--epoch', str(epoch)])
            if epoch == 1:
                train_cache = c['cached_features']
            else:
                phase('e%04d-pack' % epoch, [python, '-m', 'reproduction.prefix20_online.features',
                      '--rollout-complete', str(root/'complete.json'), '--output', str(root/'inputs')])
                phase('e%04d-encode' % epoch, torchrun('reproduction.prefix20_online.features',
                    ['--inputs', str(root/'inputs'), '--model', c['model_dir'], '--output', str(root/'features'),
                     '--micro-batch-size', str(c.get('encoder_micro_batch', 32))]))
                train_cache = str(root/'features')
            phase('e%04d-train' % epoch, torchrun('reproduction.prefix20_online.train_epoch',
                ['--train-cache', train_cache, '--test-cache', c['cached_features'],
                 '--test-manifest-sha', c['cached_features_manifest_sha256'], '--model-dir', c['model_dir'],
                 '--output', str(base/'train'), '--epoch', str(epoch), '--epochs', str(c.get('epochs', 1000)),
                 '--global-batch', str(c.get('global_batch', 256)), '--save-every', str(c.get('save_every', 10)),
                 '--learning-rate', str(c.get('learning_rate', .0003)), '--seed', str(c.get('seed', 1729))]))
            current = read(base/'train/latest.json')
            expected_adam = epoch * ((25600 + int(c.get('global_batch',256))-1) // int(c.get('global_batch',256)))
            if current['epoch'] != epoch or current['actual_adam_steps_per_rank'] != [expected_adam]*8:
                raise RuntimeError('Actual eight-rank head Adam boundary incomplete')
            snapshot_pool(base, epoch, c, current)
            stage('EPOCH_COMPLETE', actual_adam_steps_per_rank=current['actual_adam_steps_per_rank'])
            if epoch > 1:
                prune_rebuildable(base, epoch)
            if epoch == 1 and int(c.get('epochs', 1000)) > 1:
                stage('STARTING_OWN_RETRIEVER')
                env_root = Path(c['environment_root'])
                libs = list((env_root/'lib/python3.10/site-packages/nvidia').glob('*/lib'))
                libs += [env_root/'lib/python3.10/site-packages/torch/lib']
                retrieve_env = dict(env, PYTHONPATH=c['retriever_gpu_lib']+':'+str(base/'repo'),
                                    LD_LIBRARY_PATH=':'.join(map(str, libs)))
                retriever = start('retriever', [python, '-m', 'reproduction.prefix20_online.retriever',
                    '--index', c['retrieval_index'], '--corpus', c['retrieval_corpus'],
                    '--encoder', c['retrieval_encoder'], '--cache', str(base/'retrieval-cache'),
                    '--run-id', c['run_id'], '--port', str(c.get('retrieval_port', 18010)),
                    '--batch-queries', str(c.get('retrieval_batch_queries', 512))], retrieve_env)
                import urllib.request
                deadline = time.monotonic()+1800
                while True:
                    interrupt.check()
                    owned.observe('retriever')
                    if retriever.poll() is not None:
                        raise RuntimeError('Own retrieval service failed during loading')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:'+str(c.get('retrieval_port', 18010))+'/health', timeout=5) as response:
                            ready = json.load(response)
                        if ready.get('ok') and ready.get('run_id') == c['run_id'] and ready.get('gpu_shards') == 8:
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Own retrieval loading did not become ready')
                    time.sleep(5)
        success = True
    except BaseException as error:
        stage('FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            owned.stop_all()
            owned.reap()
            births = [row for job in owned.jobs.values() for row in job['ledger'].values()]
            present = birth_states(births)
            once(base/'logs/cleanup-status.json', dict(run_id=c['run_id'], cleaned=not any(row['state']!='Z' for row in present),
                 owned_births_all_gone=not present, remaining_owned_births=[row['pid'] for row in present], checked_unix=time.time()))
        finally:
            lease.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if success:
        once(base/'logs/pipeline-complete.json', dict(role=ROLE, run_id=c['run_id'], epochs=epoch,
             latest=read(base/'train/latest.json'), actor_updates=0, native_exit0_requires_post_exit_observation=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('run', 'submit', 'query'))
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    result = {'run':run, 'submit':submit, 'query':query}[args.phase](args.base)
    if result is not None:
        print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

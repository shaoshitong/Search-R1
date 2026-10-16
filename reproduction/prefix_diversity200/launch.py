"""One native allocation for the approved prefix-selection actor200 experiment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

from reproduction.prefix20_online.launch import read, sha, once, publish, query

PREFIX = '/mnt/data/shaoshitong/search-g1-prefix-diversity200-'


def native_allocation(base, c):
    from reproduction.common8.native_allocation import fields
    from reproduction.common8.stage_retrieval import memory_bytes
    node = c['node']
    if socket.gethostname().split('.')[0] != node or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Actual configured native allocation is required')
    job = os.environ['SLURM_JOB_ID']
    raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=30)
    actual = fields(raw)
    if (actual.get('JobId') != job or actual.get('JobState') != 'RUNNING'
            or actual.get('NodeList') != node or int(actual['NumCPUs']) < 96
            or memory_bytes(actual['TRES']) < 512*2**30
            or actual.get('TresPerNode') not in ('gres:gpu:8', 'gres:gpu:h100:8')):
        raise ValueError('Actual eight-GPU CPU/RAM allocation differs')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=30).splitlines()
    mapping = {int(row.split(',')[0]):row.split(',')[1].strip() for row in rows}
    if [mapping[i] for i in range(8)] != c['physical_gpu_uuids']:
        raise ValueError('Actual physical GPU UUID mapping differs')
    value = dict(run_id=c['run_id'], job_id=job, node=node, native=raw,
        physical_gpu_uuids=c['physical_gpu_uuids'], checked_unix=time.time())
    once(base/'logs/native-allocation.json', value)
    return value


def validate_config(c, base):
    if c['base'] != str(base) or not str(base).startswith(PREFIX):
        raise ValueError('Use the isolated remote prefix-selection namespace')
    required = dict(outers=200, candidate_questions=5120, selected_questions=1024, actor_selected_questions=512,
                    responses_per_question=5, prefix_policy_tokens=20, exploration_questions=0,
                    probe_learning_rate=1e-6, probe_epochs_per_outer=1,
                    probe_updates_per_outer=20, probe_global_batch_average=256,
                    actor_global_mini_batch=128, actor_ppo_epochs=1, world_size=8)
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('The approved prefix selection or training budget differs')
    if c['probe_source_sha256'] != '8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56':
        raise ValueError('Initialize from the best retained warmup CP100')
    if c['model_revision'] != 'aa8e72537993ba99e69dfaafa59ed015b17504d1':
        raise ValueError('Official actor/ref/encoder revision differs')
    if len(set(c['physical_gpu_uuids'])) != 8:
        raise ValueError('Eight actual physical GPUs are required')
    if c['node'] not in ('h100-8', 'h100-10'):
        raise ValueError('Only authorized H100 cluster nodes are supported')
    for key in ('environment_root', 'model_dir', 'official_model_dir', 'encoder_model_dir',
                'canonical_train', 'source_question_pool', 'probe_source_checkpoint',
                'fixed_test_features', 'retrieval_index', 'retrieval_corpus', 'grounding_manifest'):
        if not str(c[key]).startswith(('/mnt/', '/dataset/')):
            raise ValueError('All large inputs and outputs must stay remote: ' + key)
    return c


def config(base):
    base = Path(base).resolve()
    return validate_config(read(base / 'preparation/launch-config.json'), base)


def submit(base):
    import fcntl
    base = Path(base).resolve()
    prep = base / 'preparation'
    with (prep / 'submit-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (prep / 'submit-intent.json').exists():
            return query(base)
        c = config(base)
        launcher = prep / 'native-launch.sh'
        if launcher.exists() or (prep / 'submit-receipt.json').exists():
            raise RuntimeError('Partial submission requires reconciliation, never blind replay')
        python = str(Path(c['environment_root']) / 'bin/python')
        lines = ['#!/bin/bash', 'set -euo pipefail',
                 'export PATH=' + shlex.quote(str(Path(python).parent) + ':/usr/local/bin:/usr/bin:/bin'),
                 'export PYTHONPATH=' + shlex.quote(str(base / 'repo') + ':' + c['runtime_extra_packages']),
                 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
                 'unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy',
                 'exec ' + shlex.join([python, '-m', 'reproduction.prefix_diversity200.launch', 'run', '--base', str(base)])]
        with launcher.open('x') as handle:
            handle.write('\n'.join(lines) + '\n'); handle.flush(); os.fsync(handle.fileno())
        launcher.chmod(0o700)
        command = ['sbatch', '--parsable', '--partition=gpu', '--oversubscribe', '--nodelist=' + c['node'],
            '--nodes=1', '--ntasks=1', '--cpus-per-task=96', '--mem=512G', '--gres=gpu:8',
            '--time=0', '--signal=B:TERM@60', '--export=NONE',
            '--job-name=g1-prefix200-' + c['run_id'][:12], '--comment=g1-prefix200:' + c['run_id'],
            '--chdir=' + str(base), '--output=' + str(base / 'logs/native-%j.log'),
            '--error=' + str(base / 'logs/native-%j.log'), str(launcher)]
        once(prep / 'submit-intent.json', dict(run_id=c['run_id'], source_commit=c['source_commit'],
            config_sha256=sha(prep / 'launch-config.json'), command=command,
            job_name='g1-prefix200-' + c['run_id'][:12], comment='g1-prefix200:' + c['run_id'], created_unix=time.time()))
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            receipt = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                status='REJECTED_NO_REPLAY', intent_sha256=sha(prep / 'submit-intent.json'))
            if result.returncode == 0:
                job = result.stdout.strip().split(';')[0]
                receipt.update(status='SUBMITTED' if job.isdecimal() else 'UNKNOWN_NO_REPLAY')
                if job.isdecimal():
                    receipt['job_id'] = job
        except (OSError, subprocess.TimeoutExpired) as error:
            receipt = dict(status='UNKNOWN_NO_REPLAY', error=repr(error), intent_sha256=sha(prep / 'submit-intent.json'))
        once(prep / 'submit-receipt.json', receipt)
        return receipt


def _json_request(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def reused_phase(base, root, phase, c):
    """Only explicitly imported complete source phases may bypass execution."""
    path = root / phase / 'complete.json'
    if not path.exists():
        return False
    imported = read(base / 'preparation/resume-import-receipt.json')
    if (c.get('resume_import', {}).get('source_base') != imported['source_base']
            or imported['run_id'] != c['run_id'] or imported['outer'] != 1
            or imported['phase_hashes'][phase] != sha(path)):
        raise ValueError('Existing phase is not the explicitly inherited complete source')
    value = read(path)
    if value['run_id'] != c['run_id'] or value['outer'] != 1:
        raise ValueError('Inherited phase belongs to another logical run/outer')
    return True


def run(base):
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    from .pool import prepare
    base = Path(base).resolve(); c = config(base)
    if sha(base / 'preparation/launch-config.json') != read(base / 'preparation/submit-intent.json')['config_sha256']:
        raise ValueError('Submitted configuration changed')
    if (base / 'logs/process-ledger.json').exists():
        raise RuntimeError('Existing execution needs explicit recovery; do not replay its launch')
    native_allocation(base, c)
    lease = UUIDLocks(c['physical_gpu_uuids'])
    owned = torchrun_owned_class()(base / 'logs/process-ledger.json', c['run_id'])
    interrupt = DeferredInterrupt(); handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, interrupt.handler)
    runtime_path = str(base / 'repo') + ':' + c['runtime_extra_packages']
    env = dict(os.environ, PYTHONPATH=runtime_path, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        SEARCH_G1_BASE=str(base), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', NCCL_NVLS_ENABLE='0',
        HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'),
        TRITON_CACHE_DIR=str(base / 'cache/triton'), TORCHINDUCTOR_CACHE_DIR=str(base / 'cache/inductor'),
        TMPDIR=str(base / 'tmp'), PYTHONUNBUFFERED='1', NO_PROXY='127.0.0.1,localhost')
    for key in ('PYTHONHOME', 'RAY_ADDRESS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    for name in ('tmp', 'cache/hf', 'cache/triton', 'cache/inductor'):
        (base / name).mkdir(parents=True, exist_ok=True)
    python = str(Path(c['environment_root']) / 'bin/python')
    conf = str(base / 'preparation/launch-config.json')
    outer = 0; services = {}; success = False; timings = {}; actor_micro = c['actor_global_micro_batch_size']

    def stage(name, **extra):
        publish(base / 'logs/pipeline-status.json', dict(run_id=c['run_id'], stage=name,
            outer=outer, checked_unix=time.time(), **extra))

    def start(name, command, child_env=None):
        interrupt.check()
        p = owned.start(name, command, child_env or env, str(base / 'logs' / (name + '.log')), str(base / 'repo'))
        owned.observe(name)
        return p

    def phase(name, command, child_env=None, allow_failed=False):
        began = time.monotonic()
        stage(name); p = start(name, command, child_env)
        while p.poll() is None:
            owned.observe(name); interrupt.check()
            for service_name, service in services.items():
                owned.observe(service_name)
                if service.poll() is not None:
                    raise RuntimeError('Own required service exited: ' + service_name)
            time.sleep(2)
        owned.observe(name); owned.jobs[name]['exit_code'] = p.returncode; owned.save(); owned.stop(name)
        interrupt.check()
        timings[name] = dict(elapsed_seconds=time.monotonic()-began, exit_code=p.returncode)
        publish(base / 'logs/phase-timings.json', timings)
        if p.returncode and not allow_failed:
            raise RuntimeError('Actual phase failed: ' + name + '; preserve its original log and artifacts')
        return p.returncode

    def distributed(module, args):
        return [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '-m', module] + args

    def await_service(name, url, expected):
        deadline = time.monotonic() + 1800
        while True:
            interrupt.check(); owned.observe(name)
            if services[name].poll() is not None:
                raise RuntimeError('Own service failed while loading: ' + name)
            try:
                value = _json_request(url)
                if value.get('ok') and all(value.get(k) == v for k, v in expected.items()):
                    return value
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Own service did not become ready: ' + name)
            time.sleep(3)

    try:
        phase('probe-initialize-model-only', [python, '-m', 'reproduction.prefix_diversity200.probe', 'initialize',
            '--state-dir', str(base / 'probe'), '--source-checkpoint', c['probe_source_checkpoint'],
            '--source-sha256', c['probe_source_sha256'], '--run-id', c['run_id'],
            '--test-cache', c['fixed_test_features'], '--test-manifest-sha', c['fixed_test_manifest_sha256']],
            dict(env, CUDA_VISIBLE_DEVICES=''))
        stage('STARTING_RETRIEVER_AND_EIGHT_GROUNDING_REPLICAS')
        env_root = Path(c['environment_root'])
        libs = list((env_root / 'lib/python3.10/site-packages/nvidia').glob('*/lib'))
        libs += [env_root / 'lib/python3.10/site-packages/torch/lib']
        services['retriever'] = start('retriever', [python, '-m', 'reproduction.prefix20_online.retriever',
            '--index', c['retrieval_index'], '--corpus', c['retrieval_corpus'], '--encoder', c['retrieval_encoder'],
            '--cache', str(base / 'retrieval-cache'), '--run-id', c['run_id'], '--port', str(c['retrieval_port']),
            '--batch-queries', str(c.get('retrieval_batch_queries', 1024))],
            dict(env, PYTHONPATH=c['retriever_gpu_lib'] + ':' + runtime_path, LD_LIBRARY_PATH=':'.join(map(str, libs))))
        for rank in range(8):
            name = 'grounding-rank-%d' % rank
            services[name] = start(name, [python, 'probe_server.py', '--manifest', c['grounding_manifest'],
                '--port', str(c['grounding_probe_base_port'] + rank)],
                dict(env, CUDA_VISIBLE_DEVICES=str(rank), SEARCH_G1_BASE=str(base / 'grounding' / ('rank-%d' % rank)), OMP_NUM_THREADS='2'))
        await_service('retriever', c['search_url'].replace('/retrieve', '/health'), dict(run_id=c['run_id'], gpu_shards=8))
        for rank in range(8):
            await_service('grounding-rank-%d' % rank, 'http://127.0.0.1:%d/health' % (c['grounding_probe_base_port'] + rank),
                dict(version=c['grounding_probe_version']))
        for outer in range(1, 201):
            root = base / 'outers' / ('outer-%06d' % outer)
            stage('CANDIDATE_POOL_RESERVATION')
            prepare(c, outer)
            actor = read(base / 'actor/latest.json') if outer > 1 else None
            actor_binding = dict(run_id=c['run_id'], outer=outer,
                actor_model_dir=actor['hf_model_dir'] if actor else c['official_model_dir'],
                actor_version=actor['actor_version'] if actor else c['run_id'] + '-official0')
            actor_binding_path = root / 'actor-binding.json'
            if actor_binding_path.exists():
                if not c.get('resume_import') or read(actor_binding_path) != actor_binding:
                    raise ValueError('Inherited actual actor binding differs')
            else:
                once(actor_binding_path, actor_binding)
            if not reused_phase(base, root, 'prefix', c):
                phase('outer-%06d-prefix' % outer, [python, '-m', 'reproduction.prefix_diversity200.rollout',
                    '--config', conf, '--outer', str(outer), '--phase', 'prefix'])
            if not reused_phase(base, root, 'probe-score', c):
                phase('outer-%06d-score' % outer, distributed('reproduction.prefix_diversity200.probe',
                    ['score', '--state-dir', str(base / 'probe'), '--prefix-inputs', str(root / 'prefix/scan'),
                     '--output', str(root / 'probe-score'), '--outer', str(outer), '--model-dir', c['official_model_dir'],
                     '--encoder-micro-batch-size', str(c.get('encoder_micro_batch', 256))]))
            phase('outer-%06d-continue' % outer, [python, '-m', 'reproduction.prefix_diversity200.rollout',
                '--config', conf, '--outer', str(outer), '--phase', 'continue', '--selection', str(root / 'probe-score/selection.json')])
            phase('outer-%06d-diversity-selection' % outer, [python, '-m', 'reproduction.prefix_diversity200.diversity',
                '--config', conf, '--outer', str(outer)])
            # Retry only an actual typed CUDA OOM, after all failed owned workers
            # have exited. Restore the previous committed state and SAME rollout;
            # never combine an abandoned partial Adam prefix with the retry.
            from .actor import collect_oom_retry
            attempt_id = uuid.uuid4().hex
            while True:
                attempt_config = dict(c, actor_attempt_id=attempt_id,
                    actor_global_micro_batch_size=actor_micro)
                attempt_path = root / ('actor-attempt-%s-config.json' % attempt_id)
                once(attempt_path, attempt_config)
                code = phase('outer-%06d-actor-%s' % (outer, attempt_id), distributed('reproduction.prefix_diversity200.actor',
                    ['--config', str(attempt_path), '--outer', str(outer), '--rollout', str(root / 'actor-selection/complete.json'),
                     '--output', str(base / 'actor')]), allow_failed=True)
                if code == 0:
                    break
                retry = collect_oom_retry(base / 'actor', attempt_path, outer, root / 'actor-selection/complete.json',
                    attempt_id, phase_exit_code=code)
                once(root / ('actor-attempt-%s-recovery.json' % attempt_id), retry)
                if retry['status'] == 'ACTOR_ALREADY_COMMITTED':
                    break
                if not retry.get('retry'):
                    raise RuntimeError('Actual actor failure requires repair: '+json.dumps(retry))
                actor_micro = retry['actor_global_micro_batch_size']
                attempt_id = retry['new_attempt_id']
            phase('outer-%06d-probe-update' % outer, distributed('reproduction.prefix_diversity200.probe',
                ['update', '--state-dir', str(base / 'probe'), '--scored-features', str(root / 'probe-score'),
                 '--own-em', str(root / 'full/own-em.jsonl'), '--output', str(root / 'probe-update'),
                 '--own-em-sha256', read(root / 'full/complete.json')['own_em']['sha256'],
                 '--actor-selection', str(root / 'actor-selection/selection.json'),
                 '--outer', str(outer), '--model-dir', c['official_model_dir'],
                 '--test-cache', c['fixed_test_features'], '--test-manifest-sha', c['fixed_test_manifest_sha256']]))
            actor, probe = read(base / 'actor/latest.json'), read(base / 'probe/latest.json')
            if actor['actual_adam_steps_per_rank'] != [20 * outer] * 8 or probe['actual_adam_steps_per_rank'] != [20 * outer] * 8:
                raise RuntimeError('Actual actor/probe Adam boundaries differ from the completed outer')
            snapshot = root / 'pool-state-at-completion.json'
            once(snapshot, read(base / 'question-pool/state.json'))
            probe_snapshot = root / 'probe-state-at-completion.pt'
            os.link(base / 'probe/latest.pt', probe_snapshot)
            complete = dict(run_id=c['run_id'], outer=outer, actor=actor, probe=probe,
                probe_snapshot=str(probe_snapshot), pool_snapshot=str(snapshot), pool_sha256=sha(snapshot),
                selection_sha256=sha(root / 'probe-score/selection.json'), rollout=read(root / 'full/complete.json'),
                actor_selection=read(root / 'actor-selection/selection.json'),
                probe_metrics=read(root / 'probe-update/metrics.json'), source_commit=c['source_commit'], observed_unix=time.time())
            complete['phase_timings'] = {key: value for key,value in timings.items() if key.startswith('outer-%06d-' % outer)}
            once(root / 'complete.json', complete)
            publish(base / 'latest-joint.json', complete)
            stage('OUTER_JOINT_COMPLETE', complete_outer=outer, actor_adam_per_rank=20 * outer, probe_adam_per_rank=20 * outer)
            # The evaluation module retains the earlier confirmed legacy QA protocol.
            if outer in (5, 50, 100, 150, 200):
                evaluation_config = root / 'evaluation-config.json'
                once(evaluation_config, dict(c, actor_version=actor['actor_version']))
                phase('outer-%06d-validation' % outer, [python, '-m', 'reproduction.prefix_diversity200.eval',
                    '--config', str(evaluation_config), '--model', actor['hf_model_dir'],
                    '--questions', c['validation_questions'], '--expected-count', '512', '--output', str(root / 'validation')])
            # Only completed run-owned derived features are removable; retain tokens/labels.
            from .storage import prune_completed
            prune_completed(base, outer)
        phase('final51713', [python, '-m', 'reproduction.prefix_diversity200.eval', '--config', str(root / 'evaluation-config.json'),
            '--model', read(base / 'actor/latest.json')['hf_model_dir'], '--questions', c['final_questions'],
            '--expected-count', '51713', '--output', str(base / 'final-evaluation')])
        once(base / 'logs/pipeline-complete.json', dict(run_id=c['run_id'], complete_outers=200,
            latest_joint=read(base / 'latest-joint.json'), final_evaluation=read(base / 'final-evaluation/complete.json')))
        success = True
    finally:
        try:
            owned.stop_all(); owned.reap()
            births = [row for job in owned.jobs.values() for row in job['ledger'].values()]
            present = birth_states(births)
            once(base / 'logs/cleanup-status.json', dict(run_id=c['run_id'], cleaned=not any(row['state'] != 'Z' for row in present),
                owned_births_all_gone=not present, remaining_owned_births=[row['pid'] for row in present], checked_unix=time.time()))
        finally:
            lease.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return dict(status='PIPELINE_COMPLETE' if success else 'INCOMPLETE', base=str(base))


def observe(base):
    from reproduction.prefix20_online.observe import small
    base = Path(base)
    value = dict(base=str(base), observed_unix=time.time(), stage=small(base / 'logs/pipeline-status.json'),
        latest_joint=small(base / 'latest-joint.json'), actor=small(base / 'actor/latest.json'),
        probe=small(base / 'probe/latest.json'), cleanup=small(base / 'logs/cleanup-status.json'),
        pipeline_complete=small(base / 'logs/pipeline-complete.json'))
    receipt = small(base / 'preparation/submit-receipt.json')
    if receipt and receipt.get('job_id'):
        from reproduction.common8.native_allocation import fields
        result = subprocess.run(['scontrol', 'show', 'job', str(receipt['job_id']), '-o'], capture_output=True, text=True, timeout=30)
        value['native'] = fields(result.stdout) if result.returncode == 0 else dict(unavailable=result.stderr)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    value['gpu_actual'] = dict(returncode=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr)
    current = (value['stage'] or {}).get('outer', 0)
    if current:
        root = base / 'outers' / ('outer-%06d' % current)
        value['current_prefix'] = [small(root / 'prefix' / ('rank-%d-progress.json' % rank)) for rank in range(8)]
        value['current_continuation'] = [small(root / 'full' / ('rank-%d-progress.json' % rank)) for rank in range(8)]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('run', 'submit', 'query', 'observe'))
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    result = dict(run=run, submit=submit, query=query, observe=observe)[args.phase](args.base)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

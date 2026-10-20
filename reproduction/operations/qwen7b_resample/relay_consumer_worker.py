"""An additional h100-10 TP4 sampler using the original h100-5 retriever.

The local HTTP relay keeps the original sampling URL and all canonical batch
configs unchanged. The h100-5 GPU unit is never a child and is never signalled.
Only the consumer's own sampler/local relay and four UUID leases are owned.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

import shared_vram_worker as shared


class BorrowedRelay:
    """Observe a verified sibling relay; never own, stop or reap its process."""
    def __init__(self, proof):
        self.proof = proof
        self.pid = int(proof['pid'])

    def poll(self):
        try:
            stat = Path('/proc/%d/stat' % self.pid).read_text().rsplit(')', 1)[1].split()
            if (stat[0] == 'Z' or stat[19] != str(self.proof['birth'])
                    or Path('/proc/%d/cgroup' % self.pid).read_text() != self.proof['cgroup']
                    or os.stat('/proc/%d/ns/net' % self.pid).st_ino != self.proof['netns_inode']):
                return 1
        except (FileNotFoundError, ProcessLookupError):
            return 1
        return None


def borrowed_relay(pool, args):
    path = getattr(args, 'reuse_local_relay_ready', None)
    if not path:
        return None
    shared.require(pool.sha(path) == args.reuse_local_relay_sha256, 'Borrowed relay ready proof changed')
    ready = pool.read(path)
    shared.require(ready['status'] == 'RELAY_LISTENING' and ready['role'] == 'downstream'
                   and ready['listen'] == dict(host='127.0.0.1', port=28831)
                   and ready['listen_url'] == 'http://127.0.0.1:28831'
                   and ready['code_sha256'] == pool.sha(args.relay_script)
                   and ready['upstream_proof']['sha256'] == pool.sha(args.upstream_ready),
                   'Borrowed local relay identity or upstream differs')
    owner = args.reuse_local_relay_unit
    shared.require(owner.startswith('sst-q7b-g5-consumer10-') and owner.endswith('.service')
                   and '/' not in owner and owner in ready['process']['cgroup'], 'Borrowed relay owner differs')
    shared.require(ready['process']['netns_inode'] == os.stat('/proc/1/ns/net').st_ino,
                   'Borrowed relay is not in the shared host network')
    relay = BorrowedRelay(ready['process'])
    shared.require(relay.poll() is None, 'Borrowed relay original PID/birth is no longer live')
    return relay


def admit(pool, args):
    shared.require(socket.gethostname().split('.')[0] == 'h100-10', 'Consumer is restricted to retained h100-10')
    shared.require(os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino,
                   'Consumer requires its own retained-path mount namespace')
    shared.require(os.stat('/proc/self/ns/net').st_ino == os.stat('/proc/1/ns/net').st_ino,
                   'Consumer relay explicitly uses the host private-cluster network')
    evidence = pool.read(args.allocation_evidence)
    props = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    shared.require(evidence['node'] == 'h100-10' and evidence['main_pid'] == os.getpid()
                   and props.get('MainPID') == str(os.getpid()) and props.get('ActiveState') == 'active'
                   and props.get('ControlGroup') == evidence['cgroup']
                   and evidence['cgroup'] in Path('/proc/self/cgroup').read_text(), 'Consumer unit/PID/cgroup differs')
    model = shared.indices(args.model_gpus, 4)
    inventory = {x['index']: x for x in pool.nvml().values()}
    uuids = [inventory[x]['uuid'] for x in model]
    locks = []
    try:
        for uuid in sorted(uuids):
            fd = pool.try_lock(Path(args.base) / 'pool/gpu-locks' / (uuid + '.lock'))
            shared.require(fd is not None, 'Another cooperating sampler holds ' + uuid)
            locks.append(fd)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, model)), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        value = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                    '--cuda-probe', '--expected-uuids', ','.join(uuids)], env=env, text=True, timeout=45)
        shared.require(json.loads(value)['actual_cuda_runtime_uuids'] == uuids, 'Consumer physical UUID order differs')
        borrowed = borrowed_relay(pool, args)
        if borrowed is None:
            with socket.socket() as check:
                check.bind(('127.0.0.1', 28831))
        upstream = pool.read(args.upstream_ready)
        shared.require(upstream['role'] == 'upstream' and upstream['source_pid'] == 1490465,
                       'Retrieval relay is not bound to the currently evidenced original retriever')
        shared.require(upstream['code_sha256'] == pool.sha(args.relay_script), 'Upstream relay source differs from published transport')
        return dict(job_id='relay10-' + str(os.getpid()), node='h100-10', controller_pid=os.getpid(),
            allocation_kind='HUMAN_AUTHORIZED_NATIVE_MODEL_ONLY_FREE_VRAM_CO_SHARE',
            native_unit=evidence['unit'], native_cgroup=evidence['cgroup'],
            model_indices=model, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, model)),
            actual_cuda_runtime_uuids=uuids, tensor_parallel_size=4, model_engine_count=1,
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices'),
            initial_gpu_memory=[inventory[x] for x in model],
            foreign_initial_contexts=shared.foreign_contexts(set(uuids)),
            foreign_contexts_allowed=True, foreign_processes_signalled=False,
            local_relay_reused=borrowed is not None,
            local_relay_process=borrowed.proof if borrowed else None,
            shared_retriever_owned_by_consumer=False, upstream_ready=str(args.upstream_ready),
            upstream_ready_sha256=pool.sha(args.upstream_ready),
            allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], captured_epoch=time.time()), locks
    except BaseException:
        for fd in locks:
            os.close(fd)
        raise


def controller(pool, args):
    base = Path(args.base)
    pool.configuration(base)
    marker = Path(args.handoff_marker or str(base / 'preparation/pool-handoff.json'))
    handoff = pool.handoff(base, marker)
    binding, locks = admit(pool, args)
    jobdir = base / 'pool/jobs' / ('job-' + binding['job_id']) / ('attempt-%d-%d' % (time.time_ns(), os.getpid()))
    jobdir.mkdir(parents=True)
    pool.publish(jobdir / 'allocation.json', binding)
    pool.publish(jobdir / 'handoff.json', handoff)
    pool.publish(jobdir / 'sources.json', {str(p): pool.sha(p) for p in
        [Path(__file__), Path(args.pool_worker), Path(args.relay_script), Path(shared.__file__),
         Path(__file__).with_name('shared_vram_vllm_adapter.py')]})
    upstream = pool.read(args.upstream_ready)
    evidence = pool.read(args.allocation_evidence)
    end = evidence['captured_epoch'] + evidence['resource_limits']['runtime_max_seconds']
    deadline = min(time.time() + args.run_seconds, end - 45)
    children, streams = [], []
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', VLLM_HOST_IP='127.0.0.1',
               VLLM_ATTENTION_BACKEND='XFORMERS', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               Q7B_POOL_INTERNAL=str(jobdir), CUDA_DEVICE_ORDER='PCI_BUS_ID')

    def spawn(name, command, cuda, extra=None):
        stream = (jobdir / (name + '.log')).open('ab')
        streams.append(stream)
        child_env = dict(env, CUDA_VISIBLE_DEVICES=cuda)
        child_env.update(extra or {})
        p = subprocess.Popen([sys.executable, '-u', *command], env=child_env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
        pool.publish(jobdir / 'children.json', dict(own_process_groups=[x.pid for x in children],
            remote_retriever_is_not_owned=True, foreign_processes_signalled=False, epoch=time.time()))
        return p

    result = None
    try:
        relay = borrowed_relay(pool, args)
        if relay is None:
            local_ready = jobdir / 'relay-ready.json'
            relay = spawn('relay', [args.relay_script, 'downstream', '--listen-host', '127.0.0.1', '--listen-port', '28831',
                '--upstream-url', upstream['listen_url'], '--upstream-ready-file', args.upstream_ready, '--token-file', args.token_file,
                '--ready-file', str(local_ready)], '')
        else:
            local_ready = Path(args.reuse_local_relay_ready)
            pool.publish(jobdir / 'borrowed-local-relay.json', dict(ready_file=str(local_ready),
                ready_sha256=pool.sha(local_ready), process=relay.proof, process_owned=False))
        ready = None
        while ready is None:
            shared.require(relay.poll() is None, 'Own local relay failed; see relay.log')
            shared.require(time.time() < deadline - args.batch_reserve_seconds, 'Consumer readiness exhausted runtime')
            try:
                with urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=3) as response:
                    value = json.load(response)
                if value.get('ready'):
                    shared.require(value['pid'] == upstream['source_pid'] and value['run_id'] == pool.RUN_ID
                        and value['gpu_shards'] == 4 and value['logical_flat_shards'] == 64
                        and value['encoder_dtype'] == value['index_dtype'] == 'float16'
                        and value['original_global_ids'] and value['endpoint'] == pool.URL,
                        'Original full FP16 retrieval identity differs across transparent relay')
                    ready = value
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
        pool.publish(jobdir / 'retriever-ready.json', dict(ready, remote_retriever_owned_by_consumer=False,
                     relay_ready_file=str(local_ready), original_upstream_ready=str(args.upstream_ready)))
        history = []
        for ordinal in range(3):
            launch = jobdir / 'sampler-launches' / ('launch-%02d' % ordinal)
            launch.mkdir(parents=True)
            sampler = spawn('sampler', [str(Path(__file__).resolve()), '--sampler-loop', '--base', str(base),
                '--pool-worker', args.pool_worker, '--job-dir', str(jobdir), '--launch-dir', str(launch),
                '--deadline', str(deadline), '--handoff-marker', str(marker),
                '--model-max-seqs', str(args.model_max_seqs), '--model-batch-tokens', str(args.model_batch_tokens),
                '--model-kv-block-cap', str(args.model_kv_block_cap), '--model-reserve-gib', str(args.model_reserve_gib),
                '--batch-reserve-seconds', str(args.batch_reserve_seconds)], binding['model_cuda_visible_devices'],
                dict(Q7B_SHARED_RETRY_ORDINAL=str(ordinal)))
            while sampler.poll() is None:
                shared.require(relay.poll() is None, 'Own local relay exited during sampling')
                shared.require(time.time() < end - 10, 'Consumer native runtime reached; preserve partial data')
                time.sleep(1)
            pool.stop_group(sampler)
            children.remove(sampler)
            if sampler.returncode == 0:
                result = pool.read(jobdir / 'sampler-exit.json')
                break
            oom = launch / 'typed-model-initialization-oom.json'
            progress = pool.read(jobdir / 'progress.json') if (jobdir / 'progress.json').exists() else {}
            shared.require(ordinal < 2 and oom.exists() and progress.get('completed_own_batches', 0) == 0,
                           'Consumer failed without an eligible typed constructor OOM retry; see original sampler.log')
            fact = pool.read(oom)
            shared.require(fact['actual_typed_cuda_oom'] and fact['before_first_new_sampling_request'], 'Invalid OOM receipt')
            history.append(fact)
            pool.publish(jobdir / 'typed-initialization-oom-history.json', history)
        shared.require(result is not None, 'No accepted consumer terminal receipt')
    except BaseException as error:
        pool.publish(jobdir / 'failure.json', dict(status='OWN_RELAY_CONSUMER_PARTIAL_OR_FAILED',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            foreign_processes_signalled=False, upstream_retriever_signalled=False,
            global_goal_complete=False, epoch=time.time()))
        raise
    finally:
        for p in reversed(children):
            pool.stop_group(p)
        for f in streams:
            f.close()
        for fd in locks:
            os.close(fd)
    result.update(controller_status='OWN_RELAY_CONSUMER_FINISHED', sampler_returncode=0,
                  upstream_retriever_signalled=False, foreign_processes_signalled=False,
                  own_children_reaped=True, epoch=time.time())
    pool.publish(jobdir / 'complete.json', result)
    print(json.dumps(dict(result, job_directory=str(jobdir))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base')
    p.add_argument('--pool-worker', default=str(Path(__file__).with_name('pool_worker.py')))
    p.add_argument('--relay-script', default=str(Path(__file__).with_name('retrieval_relay.py')))
    p.add_argument('--upstream-ready')
    p.add_argument('--token-file')
    p.add_argument('--reuse-local-relay-ready')
    p.add_argument('--reuse-local-relay-sha256')
    p.add_argument('--reuse-local-relay-unit')
    p.add_argument('--allocation-evidence', default=os.environ.get('Q7B_RELAY_ALLOCATION_EVIDENCE'))
    p.add_argument('--model-gpus', default='0,1,2,3')
    p.add_argument('--model-max-seqs', type=int, default=8)
    p.add_argument('--model-batch-tokens', type=int, default=4096)
    p.add_argument('--model-kv-block-cap', type=int, default=1024)
    p.add_argument('--model-reserve-gib', type=float, default=.25)
    p.add_argument('--run-seconds', type=int, default=28500)
    p.add_argument('--batch-reserve-seconds', type=int, default=300)
    p.add_argument('--sampler-loop', action='store_true')
    p.add_argument('--cuda-probe', action='store_true')
    p.add_argument('--expected-uuids')
    p.add_argument('--job-dir')
    p.add_argument('--launch-dir')
    p.add_argument('--deadline', type=float)
    p.add_argument('--handoff-marker')
    a = p.parse_args()
    shared.require(bool(a.reuse_local_relay_ready) == bool(a.reuse_local_relay_sha256)
                   == bool(a.reuse_local_relay_unit), 'Borrowed relay arguments must be supplied together')
    a.tp = 4
    if a.cuda_probe:
        print(json.dumps(dict(actual_cuda_runtime_uuids=shared.actual_uuids(a.expected_uuids.split(',')))))
        return
    shared.require(a.base and sys.platform == 'linux', 'Remote Linux only')
    shared.require(a.model_max_seqs > 0 and a.model_batch_tokens >= 4096 and a.model_kv_block_cap >= 256
                   and 0 < a.model_reserve_gib <= 1.5, 'Invalid model capacity settings')
    pool = shared.load_pool(a.pool_worker)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    if a.sampler_loop:
        pool.single_gpu_sampler = lambda binding: shared.multi_sampler(pool, binding, a)
        try:
            pool.sample_loop(a)
        except BaseException as error:
            pool.publish(Path(a.launch_dir) / 'failure.json', dict(error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), global_goal_complete=False, epoch=time.time()))
            raise
    else:
        controller(pool, a)


if __name__ == '__main__':
    main()

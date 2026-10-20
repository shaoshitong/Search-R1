"""Bounded retrieval-only owner; model consumers never control its lifetime."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

try:
    import shared_vram_worker as shared
except ModuleNotFoundError:
    from . import shared_vram_worker as shared

# Optional TP1 slots require an explicit owner manifest; none is inferred from
# current free memory. The first four retain the already measured topology.
SLOTS = {
    'base-tp4': dict(indices=[3, 4, 6, 7], tp=4, seqs=128, tokens=16384, blocks=16384, reserve=1.5),
    'retrieval-tp4': dict(indices=[0, 1, 2, 5], tp=4, seqs=32, tokens=8192, blocks=8192, reserve=.5),
    'gpu2-tp1': dict(indices=[2], tp=1, seqs=128, tokens=16384, blocks=16384, reserve=1.5),
    'gpu1-tp1': dict(indices=[1], tp=1, seqs=64, tokens=8192, blocks=8192, reserve=1.0),
    'gpu3-tp1': dict(indices=[3], tp=1, seqs=8, tokens=4096, blocks=512, reserve=1.0),
    **{'gpu%d-tp1' % i: dict(indices=[i], tp=1, seqs=8, tokens=4096, blocks=1024, reserve=.5)
       for i in (4, 6, 7)},
}
DEFAULT_SLOTS = ('base-tp4', 'retrieval-tp4', 'gpu2-tp1', 'gpu1-tp1')


def process(pid):
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    shared.require(fields[0] not in ('Z', 'X'), 'Resource owner process is terminal')
    return dict(pid=pid, birth=fields[19], cgroup=Path('/proc/%d/cgroup' % pid).read_text(),
                netns_inode=os.stat('/proc/%d/ns/net' % pid).st_ino,
                mntns_inode=os.stat('/proc/%d/ns/mnt' % pid).st_ino)


def slot_directory(base, run_id, owner, slot):
    return Path(base) / 'pool/gpu-slot-leases' / run_id / ('%d-%s' % (owner['pid'], owner['birth'])) / slot


def busy_slots(pool, base, binding):
    busy = []
    for name, slot in binding['slots'].items():
        for uuid in slot['cuda_uuids']:
            fd = pool.try_lock(slot_directory(base, pool.RUN_ID, binding['owner'], name) / (uuid + '.lock'))
            if fd is None:
                busy.append(dict(slot=name, uuid=uuid))
            else:
                os.close(fd)
    return busy


def controller(pool, args):
    base = Path(args.base)
    config = pool.configuration(base)
    pool.handoff(base, Path(args.handoff_marker))
    declared = pool.read(args.slot_manifest)
    shared.require(declared and all(name in SLOTS and row == SLOTS[name] for name, row in declared.items()),
                   'Owner accepts only explicitly declared finite slot specifications')
    binding, locks = shared.admit(pool, args)
    child = stream = None
    jobdir = base / 'pool/jobs' / ('job-retrieval-owner-' + str(os.getpid())) / ('attempt-' + str(time.time_ns()))
    jobdir.mkdir(parents=True)
    try:
        evidence = pool.read(args.allocation_evidence)
        hard_end = evidence.get('unit_hard_end_epoch', evidence['captured_epoch'] + evidence['resource_limits']['runtime_max_seconds'])
        deadline = min(time.time() + args.owner_seconds, hard_end - 3600)
        shared.require(deadline - time.time() > 28500 + 1800, 'Owner lifetime must exceed a model lease plus drain margin')
        owner = dict(process(os.getpid()), unit=evidence['unit'])
        mapping = dict(zip(binding['model_indices'] + binding['retrieval_indices'], binding['actual_cuda_runtime_uuids']))
        slots = {name: dict(row, cuda_uuids=[mapping[index] for index in row['indices']]) for name, row in declared.items()}
        proofs = {}
        for fd in locks:
            info = os.fstat(fd)
            for uuid in mapping.values():
                primary = base / 'pool/gpu-locks' / (uuid + '.lock')
                actual = primary.stat()
                if (info.st_dev, info.st_ino) == (actual.st_dev, actual.st_ino):
                    proofs[uuid] = dict(fd=fd, device=info.st_dev, inode=info.st_ino)
        shared.require(set(proofs) == set(mapping.values()), 'Owner primary lock FD inventory differs')
        binding.update(run_id=pool.RUN_ID, allocation_role='RETRIEVAL_ONLY_PRIMARY_LEASE_OWNER',
            owner=owner, slots=slots, gpu_lock_proofs=proofs, model_processes_owned=False,
            leased_indices=list(mapping), leased_cuda_uuids=list(mapping.values()),
            model_engine_count=0, model_indices=[], model_cuda_uuids=[], model_cuda_visible_devices='',
            deadline_epoch=deadline, consumer_deadline_epoch=deadline - 1800, hard_end_epoch=hard_end)
        allocation = jobdir / 'allocation.json'
        pool.publish(allocation, binding)
        pool.publish(jobdir / 'sources.json', {str(Path(__file__)): pool.sha(Path(__file__))})
        encoder = args.encoder or str(Path(config['base']).parent / 'search-g1-prefix5-h10011-runtime-assets-20261018/common8/assets/e5-base-v2')
        command = [sys.executable, '-u', args.retriever_script, '--index', config['retrieval_index'],
            '--corpus', config['retrieval_corpus'], '--encoder', encoder, '--data-cache', config['retrieval_data_cache'],
            '--cache', str(jobdir / 'retrieval'), '--run-id', pool.RUN_ID, '--allocation-file', str(allocation),
            '--weights', args.retrieval_weights, '--logical-shards', '64', '--port', '28831']
        stream = (jobdir / 'retriever.log').open('ab')
        child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=binding['retrieval_cuda_visible_devices'],
            CUDA_DEVICE_ORDER='PCI_BUS_ID'),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        pool.publish(jobdir / 'children.json', dict(own_process_groups=[child.pid], model_processes_owned=False))
        ready = None
        while ready is None:
            shared.require(child.poll() is None and time.time() < deadline - 1800, 'Original retriever initialization failed or exhausted owner time')
            try:
                with urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=2) as response:
                    value = json.load(response)
                if value.get('ready'):
                    shared.require(value['pid'] == child.pid and value['run_id'] == pool.RUN_ID
                        and value['actual_cuda_uuids'] == binding['retrieval_cuda_uuids']
                        and value['gpu_shards'] == 4 and value['logical_flat_shards'] == 64
                        and value['encoder_dtype'] == value['index_dtype'] == 'float16'
                        and value['original_global_ids'] and value['endpoint'] == pool.URL, 'Full original retrieval identity differs')
                    ready = value
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
        proof = dict(status='RETRIEVAL_ONLY_OWNER_READY', run_id=pool.RUN_ID, owner=owner,
            allocation_path=str(allocation), allocation_sha256=pool.sha(allocation),
            retriever=process(child.pid), slots=slots, deadline_epoch=deadline,
            consumer_deadline_epoch=deadline - 1800, model_processes_owned=False, epoch=time.time())
        pool.publish(jobdir / 'retriever-ready.json', ready)
        pool.publish(jobdir / 'retriever-source-ready.json', dict(role='upstream', source_pid=child.pid,
            listen_url='http://127.0.0.1:28831', source_owned=False))
        pool.publish(jobdir / 'owner-ready.json', proof)
        print(json.dumps(dict(status=proof['status'], ready=str(jobdir / 'owner-ready.json'), sha256=pool.sha(jobdir / 'owner-ready.json'))), flush=True)
        marker = jobdir / 'drain-owner.json'
        while time.time() < deadline and not marker.exists():
            shared.require(child.poll() is None, 'Owned full-index retriever actually exited')
            time.sleep(1)
        if marker.exists():
            request = pool.read(marker)
            shared.require(request['status'] == 'DRAIN_RETRIEVAL_OWNER' and request['run_id'] == pool.RUN_ID
                and request['owner_pid'] == owner['pid'] and request['owner_allocation_sha256'] == proof['allocation_sha256'],
                'Owner drain marker binding differs')
        pool.publish(jobdir / 'drain-request.json', dict(status='DRAIN_RETRIEVAL_OWNER', run_id=pool.RUN_ID,
            owner_pid=owner['pid'], owner_allocation_sha256=proof['allocation_sha256'], epoch=time.time()))
        while True:
            busy = busy_slots(pool, base, binding)
            if not busy:
                break
            shared.require(child.poll() is None and time.time() < hard_end - 120,
                           'Owner hard limit reached with active model leases; real partial work may remain')
            pool.publish(jobdir / 'draining.json', dict(active_slot_leases=busy, epoch=time.time()))
            time.sleep(1)
        pool.stop_group(child)
        pool.publish(jobdir / 'complete.json', dict(status='RETRIEVAL_OWNER_DRAINED', run_id=pool.RUN_ID,
            model_leases_released=True, retriever_returncode=child.returncode, global_goal_complete=False, epoch=time.time()))
        child = None
    except BaseException as error:
        pool.publish(jobdir / 'failure.json', dict(status='RETRIEVAL_OWNER_FAILED_OR_INTERRUPTED',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), global_goal_complete=False))
        raise
    finally:
        if child is not None:
            pool.stop_group(child)
        if stream is not None:
            stream.close()
        for fd in locks:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--allocation-evidence', required=True)
    parser.add_argument('--slot-manifest', required=True)
    parser.add_argument('--owner-seconds', type=int, default=86400)
    parser.add_argument('--pool-worker', default=str(Path(__file__).with_name('pool_worker.py')))
    parser.add_argument('--retriever-script', default=str(Path(__file__).with_name('shared_vram_retriever.py')))
    parser.add_argument('--encoder')
    parser.add_argument('--handoff-marker')
    args = parser.parse_args()
    shared.require(30300 < args.owner_seconds <= 86400, 'Owner requires a bounded 8.5-to-24-hour lifetime')
    args.model_gpus, args.retrieval_gpus, args.retrieval_weights, args.tp = '3,4,6,7', '0,1,2,5', '15,9,20,20', 4
    args.handoff_marker = args.handoff_marker or str(Path(args.base) / 'preparation/pool-handoff.json')
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    controller(shared.load_pool(args.pool_worker), args)


if __name__ == '__main__':
    main()

"""Owned isolated services for pinned compatible checkpoint evaluation; never train.

The manifest is prospective *evaluation* configuration, not a fresh-training
manifest. Original model/data/index files are read in place and never deleted.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

from reproduction.compatible_async_evaluation.runtime import ROLE,validate_binding
from reproduction.compatible_async_evaluation.contract import PROTOCOL_PATH,digest,protocol,verify_source_tree

# Old-source imports are made only after the source-root binding is checked.
OwnedProcesses=publish=same=allocated_indices=cuda_uuids=fields=memory_bytes=None

_interrupt = None


def interrupted(sig, frame):
    global _interrupt
    _interrupt = sig  # Ownership registration must finish before cleanup.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COMPAT_RETRIEVER_SOURCE_SHA='b0e70e06bbd7d4d820ff7c550142fa6b22725789711f927bdc4960dcdc766ca1'


def _asset_record(path,expected):
    path=Path(path).resolve()
    before=path.stat()
    assert path.is_file() and digest(path)==expected, 'Legacy retrieval asset full SHA changed'
    after=path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(path=str(path),full_read_sha256=expected,device=after.st_dev,inode=after.st_ino,
                bytes=after.st_size,mtime_ns=after.st_mtime_ns)


def verify_retrieval_assets(manifest):
    """Remote CPU preflight in the evaluator allocation, never a training gate."""
    spec=protocol()['retrieval']
    source=Path(manifest['source_root'])/'search_r1/search/retrieval_server.py'
    assert source.resolve()==Path(manifest['retrieval_server_source']).resolve() and digest(source)==COMPAT_RETRIEVER_SOURCE_SHA
    reference=Path(manifest['retrieval_reference_admission']).resolve()
    expected_reference=Path(protocol()['protocol_authoritative_source']['base'])/'logs/retrieval-admission.json'
    assert reference==expected_reference.resolve() and digest(reference)==manifest['retrieval_reference_admission_sha256']
    receipt=json.loads(reference.read_text())
    assert receipt['accepted'] is True and receipt['run_id']==protocol()['protocol_authoritative_source']['historical_run']
    assert receipt['index_sha256']==spec['index_sha256'] and receipt['corpus_sha256']==spec['corpus_sha256']
    encoder=receipt['encoder_assets']
    assert encoder['model_id']==spec['encoder'] and encoder['revision']==spec['revision']
    records=[_asset_record(manifest['retrieval_index_path'],spec['index_sha256']),
             _asset_record(manifest['retrieval_corpus_path'],spec['corpus_sha256'])]
    for name,record in sorted(encoder['files'].items()):
        assert Path(name).name==name
        row=_asset_record(Path(manifest['retrieval_encoder'])/name,record['sha256'])
        assert row['bytes']==record['size'];records.append(row)
    return dict(role='compatible_evaluation_actual_asset_full_sha',run_id=manifest['run_id'],
        execution_id=manifest['execution_id'],recipe_sha256=digest(PROTOCOL_PATH),
        reference_admission_sha256=manifest['retrieval_reference_admission_sha256'],
        retriever_source_sha256=COMPAT_RETRIEVER_SOURCE_SHA,encoder_revision=encoder['revision'],
        files=records,checked_unix=time.time(),actual_gpu_execution=False)


def verify_cached_asset_records(base,manifest):
    proof=json.loads((base/'logs/compatible-assets-verified.json').read_text())
    assert proof['role']=='compatible_evaluation_actual_asset_full_sha' and proof['run_id']==manifest['run_id'] and proof['execution_id']==manifest['execution_id'] and proof['recipe_sha256']==digest(PROTOCOL_PATH)
    assert proof['retriever_source_sha256']==COMPAT_RETRIEVER_SOURCE_SHA
    for row in proof['files']:
        stat=Path(row['path']).stat()
        assert (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns']), 'Asset changed since actual evaluator preflight'
    return proof


def read_manifest(base):
    path = base / 'preparation/compatible-evaluation-manifest.json'
    manifest = json.loads(path.read_text())
    if 'resume_intent_sha256' in manifest:
        from reproduction.compatible_async_evaluation.resume import verify_resume_receipt
        verify_resume_receipt(base, manifest)
    assert manifest['role']==ROLE
    assert manifest['checkpoint_step'] in protocol()['checkpoints']['admitted_steps']
    assert Path(manifest['base']).resolve() == base
    assert manifest['world_size'] == 8 and manifest['cpu_request'] == 96
    assert manifest['memory_request'] == '512G'
    for name in ('source_base', 'binding', 'config', 'journal', 'gpu_lock_directory'):
        assert Path(manifest[name]).is_absolute(), name
    binding = json.loads(Path(manifest['binding']).read_text())
    assert binding['role'] == manifest['role'] and binding['checkpoint_step']==manifest['checkpoint_step'] and binding['split']==manifest['split']
    assert binding['run_id'] == manifest['run_id']
    assert sha(manifest['config']) == binding['resolved_config_sha256']
    assert sha(binding['checkpoint_manifest']) == binding['checkpoint_manifest_sha256']
    assert Path(binding['checkpoint']).is_dir()
    assert manifest['recipe_sha256']==digest(PROTOCOL_PATH)==binding['recipe_sha256']
    assert Path(manifest['control_repo']).resolve()==Path(__file__).resolve().parents[2]
    assert manifest['source_root']==binding['source_root']
    validate_binding(binding,manifest['config'],verify_weights=False)
    parsed = urlparse(manifest['retrieval_url'])
    assert parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')
    assert parsed.path == '/retrieve' and parsed.port and 1024 <= parsed.port <= 65535
    return manifest


def rows(query, kind='gpu'):
    raw = subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + query,
                                   '--format=csv,noheader,nounits'], text=True, timeout=30)
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(raw)) if row]


def native_admission(base, manifest):
    job_id = os.environ['SLURM_JOB_ID']
    detail = subprocess.check_output(['scontrol', 'show', 'job', '-d', job_id], text=True)
    job = fields(detail)
    node = socket.gethostname().split('.')[0]
    assert node == 'h100-8' and job['NodeList'] == node
    assert job['JobId'] == job_id and job['JobState'] == 'RUNNING' and job['NumNodes'] == '1'
    mode=manifest.get('allocation_mode')
    assert mode in ('exclusive','shared'), 'Explicit allocation sharing mode is required'
    assert int(job['NumCPUs'])>=96 and memory_bytes(job['TRES'])>=512*2**30
    active=set(subprocess.check_output(['squeue','-h','-w',node,'-t','RUNNING,COMPLETING','-o','%i'],text=True).split())
    forbidden={str(v) for v in manifest.get('forbidden_training_job_ids',[])}
    assert not (active & forbidden), 'An explicitly forbidden active trainer overlaps this evaluation'
    if mode=='exclusive':
        assert job.get('TresPerNode') in ('gres:gpu:8','gres:gpu:h100:8') and job['OverSubscribe']=='NO'
        assert active=={job_id}, 'Exclusive evaluator must preserve and refuse overlapping allocations'
    else:
        permitted_jobs={str(row['job_id']):row for row in manifest.get('preserved_foreign_allocations',[])}
        assert active-{job_id}<=set(permitted_jobs), 'Unbound foreign allocation; never stop it'
        for foreign in active-{job_id}:
            detail_foreign=subprocess.check_output(['scontrol','show','job','-d',foreign],text=True)
            actual=fields(detail_foreign);allowed=permitted_jobs[foreign]
            assert actual['JobId']==foreign and actual['NodeList']==node and actual['JobState'] in ('RUNNING','COMPLETING')
            assert actual['UserId']==allowed['user_id'], 'Foreign allocation owner binding changed'
    physical = {int(i): uuid for i, uuid in rows('index,uuid')}
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    allocated=allocated_indices(detail,node) if mode=='exclusive' else None
    expected = [physical[int(v)] if v.isdecimal() else v for v in visible]
    assert len(expected) == len(set(expected)) == 8
    if allocated is not None:assert set(expected)=={physical[i] for i in allocated}
    else:assert set(expected)==set(physical.values()), 'Shared evaluator still requires all eight actual GPUs'
    assert expected == manifest['physical_gpu_uuids'] and cuda_uuids() == expected
    permitted = manifest.get('preserved_foreign_gpu_processes', [])
    allowed = {(record['gpu_uuid'], str(record['identity']['pid']))
               for record in permitted if same(record['identity'])}
    apps = rows('gpu_uuid,pid', 'compute-apps')
    assert all(tuple(row) in allowed or int(row[1]) == os.getpid() for row in apps), \
        'Unbound GPU application; never stop it'
    proof = dict(role=manifest['role'], run_id=manifest['run_id'],
                 execution_id=manifest['execution_id'],accepted=True,node=node,allocation_mode=mode,
                 actual_over_subscribe=job.get('OverSubscribe'),gres_secondary_reservation_required=mode=='exclusive',
                 slurm_job_id=job_id, actual_cuda_runtime_uuids=expected,
                 checkpoint_step=manifest['checkpoint_step'],split=manifest['split'],recipe_sha256=manifest['recipe_sha256'], cpu_request=96, memory_request='512G',
                 self_cgroup=(Path('/proc/self/cgroup')).read_text(), checked_unix=time.time())
    publish(base / 'logs/compatible-native-allocation.json', proof)
    return proof


class UUIDLocks:
    """New final-only UUID locks; not a claim about previous jobs' locking."""
    def __init__(self, manifest):
        assert sys.platform.startswith('linux'), 'Native OFD lease uses the Linux flock ABI'
        root = Path(manifest['gpu_lock_directory'])
        assert root.is_dir(), 'Explicit GPU lock directory must be prepared'
        self.fds = []
        template = manifest.get('gpu_lock_filename', '{uuid}.lock')
        assert '/' not in template and '{uuid}' in template
        try:
            for uuid in sorted(manifest['physical_gpu_uuids']):
                fd = os.open(root / template.format(uuid=uuid), os.O_RDWR | os.O_CREAT, 0o600)
                self.fds.append(fd)
                fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                            struct.pack('hhqqi4x', fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        except BaseException:
            self.close()
            raise
    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds = []


class FinalPipeline:
    def __init__(self, base, manifest):
        self.base, self.manifest = base, manifest
        self.repo=Path(manifest['control_repo']).resolve()
        self.python = str(base / 'env/bin/python')
        self.visible=os.environ['CUDA_VISIBLE_DEVICES'];self.processes={}
        assert not (base / 'logs/process-ledger.json').exists(), 'Never replay an execution namespace'
        self.owned = OwnedProcesses(base / 'logs/process-ledger.json', manifest['run_id'])
        self.env = dict(os.environ, SEARCH_G1_PROFILE='compatible_async_evaluation', SEARCH_G1_BASE=str(base),
                        PYTHONPATH=str(self.repo)+':'+manifest['source_root'],
                        SEARCH_G1_COMPAT_SOURCE_ROOT=manifest['source_root'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8',
                        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ECO_USE_CF='0',
                        TMPDIR=str(base / 'tmp'), RAY_TMPDIR=manifest['ray_tmp_dir'],
                        HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'),
                        TRITON_CACHE_DIR=str(base / 'cache/triton'),
                        TORCHINDUCTOR_CACHE_DIR=str(base / 'cache/inductor'),
                        NCCL_NVLS_ENABLE='0', VLLM_ATTENTION_BACKEND='XFORMERS',
                        NO_PROXY='127.0.0.1,localhost')
        for name in ('PYTHONHOME', 'RAY_ADDRESS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                     'http_proxy', 'https_proxy', 'all_proxy'):
            self.env.pop(name, None)
        include = base / 'bootstrap-tools/python-dev/usr/include'
        if include.is_dir():
            self.env['C_INCLUDE_PATH'] = str(include / 'python3.10') + ':' + str(include)
        self.env['PATH'] = str(Path(self.python).parent) + ':' + os.environ['PATH']
        libraries = list((base / 'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
        libraries.append(base / 'env/lib/python3.10/site-packages/torch/lib')
        self.env['LD_LIBRARY_PATH'] = ':'.join(str(path) for path in libraries if path.is_dir())

    def stage(self, name, **details):
        publish(self.base / 'logs/compatible-pipeline-status.json', dict(stage=name,
                run_id=self.manifest['run_id'], execution_id=self.manifest['execution_id'],
                checked_unix=time.time(), **details))

    def observe(self):
        for name, job in self.owned.jobs.items():
            if job['state'] != 'STOPPED':
                self.owned.observe(name)
        assert _interrupt is None, 'Final evaluation interrupted; clean only owned services'

    def start(self, name, args, extra=None):
        self.observe()
        env = dict(self.env, CUDA_VISIBLE_DEVICES=self.visible)
        env.update(extra or {})
        process = self.owned.start(name, args, env, self.base / ('logs/' + name + '.log'), self.repo)
        self.processes[name]=process
        self.observe()
        return process

    def wait(self, name, process):
        while True:
            self.observe()
            code = process.poll()
            if code is not None:
                self.owned.jobs[name]['exit_code'] = code
                self.owned.save()
                assert code == 0, name + ' failed; retained owned log'
                remaining=self.owned.observe(name)
                self.owned.jobs[name]['state']='EXIT0_CLEANUP_PENDING' if remaining else 'STOPPED'
                self.owned.jobs[name]['exit0_owned_children_pending']=len(remaining)
                self.owned.save()
                return
            time.sleep(1)

    def execute(self):
        self.stage('STARTING_FINAL_ONLY_RETRIEVER')
        try:
            retriever = self.start('final-retriever', [self.python, '-m',
                                   'reproduction.compatible_async_evaluation.pipeline','--base',
                                   str(self.base), '--component', 'retriever'],
                                   {'PYTHONPATH':str(Path(self.manifest['retriever_gpu_lib']))+':'+str(self.repo)+':'+self.manifest['source_root']})
            url = self.manifest['retrieval_url'].rsplit('/', 1)[0] + '/health'
            deadline = time.monotonic() + self.manifest.get('retriever_start_timeout_seconds', 1800)
            while True:
                self.observe()
                assert retriever.poll() is None, 'Retriever exited during initialization'
                try:
                    with urllib.request.urlopen(url, timeout=3) as response:
                        health = json.load(response)
                    assert health['execution_id'] == self.manifest['execution_id']
                    assert health['run_id'] == self.manifest['run_id'] and health['shard_count'] == 8
                    break
                except (OSError, TimeoutError):
                    assert time.monotonic() < deadline, 'Retriever initialization timed out'
                    time.sleep(2)
            self.stage('COMPAT_CANONICAL_EVALUATION',actual_retriever=health,checkpoint_step=self.manifest['checkpoint_step'],split=self.manifest['split'],recipe_sha256=self.manifest['recipe_sha256'])
            m = self.manifest
            evaluator = self.start('final-evaluator', [self.python, '-m',
                'reproduction.compatible_async_evaluation.runner','--binding', m['binding'],
                '--config', m['config'], '--source-base', m['source_base'], '--journal', m['journal']])
            self.wait('final-evaluator', evaluator)
            self.stage('FINAL_EVALUATOR_EXIT0', training_executed=False)
        except BaseException as error:
            self.stage('FAILED', error=repr(error))
            raise
        finally:
            self.owned.stop_all()
            pids = {str(r['pid']) for job in self.owned.jobs.values() for r in job['ledger'].values()}
            deadline = time.monotonic() + 60
            while True:
                for process in self.processes.values():process.poll()
                remaining=[r for r in rows('gpu_uuid,pid','compute-apps') if r[1] in pids]
                live_births=[r for job in self.owned.jobs.values() for r in job['ledger'].values() if same(r)]
                if not remaining and not live_births:
                    break
                assert time.monotonic() < deadline, 'Owned birth or GPU resource has not released'
                time.sleep(.5)
            publish(self.base / 'logs/cleanup-status.json', dict(cleaned=True,
                    run_id=self.manifest['run_id'], execution_id=self.manifest['execution_id'],
                    checked_unix=time.time(), original_assets_deleted=False,
                    foreign_processes_signalled=False,
                    owned_child_births_released=all(not same(r) for job in self.owned.jobs.values() for r in job['ledger'].values()),
                    owned_gpu_contexts_released=True,
                    native_job_actual_exit0_verified=False,native_cgroup_release_verified=False,
                    native_exit_and_cgroup_release_require_post_exit_observer=True,
                    supervisor_cgroup=Path('/proc/self/cgroup').read_text()))
        result_path=self.base/'journal/compatible-result.json'
        result=json.loads(result_path.read_text())
        assert result.get('old_content_fp32_async') is True and result.get('actual_gpu_execution_verified') is True
        self.stage('COMPAT_EVALUATOR_EXIT0_OWNED_CHILDREN_GPU_RELEASED',
            result_sha256=digest(result_path),canonical_result=True,training_executed=False,
            native_job_actual_exit0_verified=False,native_cgroup_release_verified=False)


def serve_retriever(base, manifest):
    import datasets
    import faiss
    import torch
    import uvicorn
    assert torch.cuda.device_count() == faiss.get_num_gpus() == 8
    assert cuda_uuids() == manifest['physical_gpu_uuids']
    assert Path(faiss.__file__).resolve().is_relative_to(Path(manifest['retriever_gpu_lib']).resolve())
    source = Path(manifest['retrieval_server_source'])
    assert source.resolve().is_relative_to(Path(manifest['source_root']).resolve())
    assert digest(source)==COMPAT_RETRIEVER_SOURCE_SHA
    spec = importlib.util.spec_from_file_location('final_original_retrieval', source)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    corpus_path = Path(manifest['retrieval_corpus_path'])
    assert corpus_path.is_file() and Path(manifest['retrieval_index_path']).is_file()
    def corpus(path):
        assert Path(path).resolve() == corpus_path.resolve()
        return datasets.load_dataset('json', data_files=str(corpus_path), split='train',
                num_proc=4, cache_dir=str(base / 'cache/corpus'))
    upstream.load_corpus = corpus
    upstream.config = upstream.Config(retrieval_method='e5', retrieval_topk=3,
            index_path=manifest['retrieval_index_path'], corpus_path=str(corpus_path), faiss_gpu=True,
            retrieval_model_path=manifest['retrieval_encoder'], retrieval_pooling_method='mean',
            retrieval_query_max_length=256, retrieval_use_fp16=True, retrieval_batch_size=512)
    upstream.retriever = upstream.get_retriever(upstream.config)
    assert upstream.retriever.index.count() == 8 and upstream.retriever.index.d == 768
    assert upstream.retriever.index.ntotal == len(upstream.retriever.corpus)
    original_batch = upstream.retriever.batch_search
    thread = threading.get_ident()
    def serial_batch(*args, **kwargs):
        assert threading.get_ident() == thread, 'FAISS must remain on its initialization thread'
        return original_batch(*args, **kwargs)
    upstream.retriever.batch_search = serial_batch
    @upstream.app.get('/health')
    async def health():
        return dict(ok=True, run_id=manifest['run_id'], execution_id=manifest['execution_id'],
                    shard_count=8, ntotal=upstream.retriever.index.ntotal, original_topk=3)
    uvicorn.run(upstream.app, host='127.0.0.1', port=urlparse(manifest['retrieval_url']).port, workers=1)


def bootstrap_compatible(manifest):
    if Path(manifest['source_root']).resolve()!=Path(__file__).resolve().parents[2]:
        raise ValueError('COMPAT_EVALUATOR_SOURCE_ROOT_CHANGED')
    global OwnedProcesses,publish,same,allocated_indices,cuda_uuids,fields,memory_bytes
    from reproduction.a800.owned_processes import OwnedProcesses,publish,same
    from reproduction.common8.native_allocation import allocated_indices,cuda_uuids,fields
    from reproduction.common8.stage_retrieval import memory_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--component', choices=['pipeline', 'retriever'], default='pipeline')
    args = parser.parse_args()
    base = args.base.resolve()
    manifest=read_manifest(base)
    bootstrap_compatible(manifest)
    if args.component=='retriever':
        verify_cached_asset_records(base,manifest)
        return serve_retriever(base,manifest)
    for name in ('logs', 'tmp', 'ray', 'cache/hf', 'cache/triton', 'cache/inductor', 'cache/corpus'):
        (base / name).mkdir(parents=True, exist_ok=True)
    ray_tmp = Path(manifest['ray_tmp_dir'])
    assert ray_tmp.is_absolute() and str(ray_tmp) == '/tmp/g1ca-' + manifest['execution_id'][:12]
    assert len(str(ray_tmp)) < 32, 'Ray Unix sockets require a short private root'
    ray_tmp.mkdir(mode=0o700, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (base / 'compatible-execution.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = UUIDLocks(manifest)
        try:
            native_admission(base,manifest)
            publish(base/'logs/compatible-assets-verified.json',verify_retrieval_assets(manifest))
            parsed = urlparse(manifest['retrieval_url'])
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', parsed.port))
            FinalPipeline(base, manifest).execute()
        finally:
            lease.close()


if __name__ == '__main__':
    main()

"""Route only the native question Head onto one already admitted actor GPU.

Ray schedules the trainer driver as a CPU task and clears its CUDA visibility.
This hook changes only that task's process environment after Ray's assignment.
It does not alter Ray runtime_env, request a GPU resource, seed an RNG, or change
child actor placement. It runs at the actual training entry, never as a test.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import time


HEAD_KIND = 'qwen_native_lastblock_binary_v1'
SOURCE_PATH = 'search_g1/qprobe_driver_gpu.py'
ENTRY_PATH = 'verl/trainer/main_ppo_eco.py'
RECEIPT_PATH = 'logs/native-qwen-driver-gpu-route.json'


def _require(condition, message):
    if not condition:
        raise ValueError('QWEN_DRIVER_GPU_' + message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _owned(path):
    path = Path(path)
    _require(path.is_file() and path.resolve() == path and not path.is_symlink() and
             path.stat().st_uid == os.getuid(), 'UNOWNED_OR_SUBSTITUTED_BOUND_FILE')
    return path


def _json(path):
    path = _owned(path)
    _require(path.stat().st_size <= 128 * 1024**2, 'BOUND_METADATA_OVERSIZE')
    return json.loads(path.read_bytes())


def _config_value(config, key, default=None):
    return config.get(key, default) if hasattr(config, 'get') else getattr(config, key, default)


def validate_driver_route(run, allocation, placement, pipeline_claim, training_claim, *,
                          head_device, hostname, incoming_visible):
    """Metadata-only check of the exact existing four-rank GPU admission."""
    from reproduction.common8.native_allocation import local_placement, LOCAL_BINDING_RECEIPT
    _require(run.get('question_probe', {}).get('head_kind') == HEAD_KIND and head_device == 'cuda:0' and
             run['question_probe'].get('head_device') == head_device and
             run.get('execution_backend') == 'ssh_local' and run.get('profile') == 'common8' and
             run.get('training_world_size') == 4 and run.get('actor_micro_batch_per_rank') == 4 and
             run.get('execution_host') == hostname, 'UNSUPPORTED_NATIVE_DRIVER_OR_HEAD_DEVICE')
    training, auxiliary, _, _ = local_placement(run)
    uuids = run['training_gpu_uuids']
    _require(all(isinstance(value, str) and value.startswith('GPU-') for value in uuids) and
             run.get('gpu_placement_binding_receipt') == LOCAL_BINDING_RECEIPT, 'INVALID_BOUND_TRAINING_UUIDS')
    for key in ('run_id', 'execution_id', 'source_commit', 'training_physical_indices',
                'auxiliary_physical_indices', 'excluded_physical_indices', 'training_gpu_uuids',
                'assigned_gpu_uuids', 'physical_gpu_inventory', 'gpu_placement_bound_unix'):
        _require(placement.get(key) == run.get(key), 'PLACEMENT_MANIFEST_BINDING_CHANGED_' + key.upper())
    _require(placement.get('host') == hostname and
             allocation.get('accepted') is True and allocation.get('execution_backend') == 'ssh_local' and
             allocation.get('slurm_job_id') is None and allocation.get('node') == hostname and
             allocation.get('training_world_size') == 4 and
             allocation.get('training_physical_indices') == training and
             allocation.get('auxiliary_physical_indices') == auxiliary and
             allocation.get('excluded_physical_indices') == run['excluded_physical_indices'] and
             allocation.get('training_visible_devices') == list(map(str, training)) and
             allocation.get('actual_cuda_runtime_uuids') == uuids and
             allocation.get('assigned_gpu_uuids') == run['assigned_gpu_uuids'] and
             allocation.get('gpu_placement_binding_sha256') == run['gpu_placement_binding_sha256'],
             'ACTUAL_NATIVE_ALLOCATION_CHANGED')
    for value in (allocation, pipeline_claim, training_claim):
        _require(value.get('run_id') == run['run_id'] and value.get('execution_id') == run['execution_id'] and
                 value.get('execution_backend') == 'ssh_local', 'LAUNCH_IDENTITY_CHANGED')
    _require(training_claim.get('hostname') == hostname and
             pipeline_claim.get('node') == hostname and
             pipeline_claim.get('training_physical_indices') == training and
             pipeline_claim.get('auxiliary_physical_indices') == auxiliary and
             pipeline_claim.get('excluded_physical_indices') == run['excluded_physical_indices'] and
             pipeline_claim.get('assigned_gpu_uuids') == run['assigned_gpu_uuids'] and
             pipeline_claim.get('gpu_placement_binding_sha256') == run['gpu_placement_binding_sha256'] and
             pipeline_claim.get('pipeline_identity') == allocation.get('pipeline_identity') == run.get('pipeline_identity'),
             'ACTUAL_TRAINING_CLAIM_CHANGED')
    allowed = (None, '', ','.join(map(str, training)), ','.join(uuids), uuids[0])
    _require(incoming_visible in allowed, 'UNBOUND_INCOMING_CUDA_VISIBILITY')
    return dict(head_device='cuda:0', physical_index=training[0], gpu_uuid=uuids[0],
                training_physical_indices=list(training), training_gpu_uuids=list(uuids),
                auxiliary_physical_indices=list(auxiliary), ray_gpu_resources_reserved=0,
                shares_admitted_actor_training_gpu=True)


def install_driver_visibility(route, *, environ, cuda_initialized, cuda_device_count, cuda_uuids):
    """Injectable identity checks; performs no tensor allocation or RNG operation."""
    _require(environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES') in (None, ''),
             'RAY_GLOBAL_CUDA_OVERRIDE_FORBIDDEN')
    _require(cuda_initialized() is False, 'CUDA_INITIALIZED_BEFORE_BOUND_DRIVER_ROUTE')
    environ['CUDA_VISIBLE_DEVICES'] = route['gpu_uuid']
    _require(cuda_device_count() == 1 and cuda_uuids(1) == [route['gpu_uuid']],
             'ACTUAL_CUDA_UUID_DIFFERS_FROM_FIRST_ADMITTED_TRAINING_GPU')
    return dict(actual_cuda_runtime_uuids=[route['gpu_uuid']], cuda_visible_devices=route['gpu_uuid'],
                cuda_initialized_before=False, tensor_allocation_performed=False, rng_seed_performed=False)


def _birth(pid):
    root = Path('/proc') / str(pid)
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    _require(root.stat().st_uid == os.getuid(), 'PROCESS_NOT_OWNED')
    return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), uid=root.stat().st_uid)


def bind_native_driver_gpu(config):
    """Native-only actual Ray-task entry before any question Head CUDA work."""
    settings = _config_value(config, 'question_probe', {}) or {}
    if _config_value(settings, 'head_kind') != HEAD_KIND:
        return None
    import ray
    import torch
    from reproduction.common8.native_allocation import cuda_uuids, _write_local_once, LOCAL_BINDING_RECEIPT
    from reproduction.common8.topology import validate_base
    _require(ray.get_gpu_ids() == [], 'DRIVER_MUST_REMAIN_A_CPU_RAY_TASK')
    base = Path(os.environ['SEARCH_G1_BASE'])
    _require(base.resolve() == base and str(base).startswith(('/data/', '/mnt/', '/backup/')),
             'BASE_NOT_CANONICAL_REMOTE_STORAGE')
    run_path = base/'logs/run-manifest.json'
    allocation_path = base/'logs/native-allocation-admission.json'
    placement_path = base/LOCAL_BINDING_RECEIPT
    pipeline_path = base/'logs/pipeline-launch-claim.json'
    claim_path = base/'logs/training-launch-claim.json'
    run, allocation, placement, pipeline, claim = map(_json,
        (run_path, allocation_path, placement_path, pipeline_path, claim_path))
    validate_base(base, run)
    _require(Path(__file__).resolve() == base/'repo'/SOURCE_PATH and
             _sha(_owned(base/'repo'/SOURCE_PATH)) == run['source_sha256'][SOURCE_PATH] and
             _sha(_owned(base/'repo'/ENTRY_PATH)) == run['source_sha256'][ENTRY_PATH],
             'DRIVER_ROUTE_SOURCE_NOT_BOUND_TO_RUN')
    _require(_sha(placement_path) == run['gpu_placement_binding_sha256'] and
             _sha(allocation_path) == pipeline['allocation_sha256'], 'ADMISSION_OR_PLACEMENT_SHA_CHANGED')
    actual_shell = _birth(claim['shell_pid'])
    _require(actual_shell['start_ticks'] == claim['shell_start_ticks'], 'TRAINING_SHELL_BIRTH_CHANGED')
    trainer = _config_value(config, 'trainer', {})
    _require(_config_value(trainer, 'n_gpus_per_node') == 4 and _config_value(trainer, 'nnodes') == 1,
             'RAY_ACTOR_POOL_MUST_STAY_EXACTLY_FOUR_GPUS')
    incoming = os.environ.get('CUDA_VISIBLE_DEVICES')
    route = validate_driver_route(run, allocation, placement, pipeline, claim,
        head_device=_config_value(settings, 'head_device'), hostname=socket.gethostname(), incoming_visible=incoming)
    actual = install_driver_visibility(route, environ=os.environ, cuda_initialized=torch.cuda.is_initialized,
        cuda_device_count=torch.cuda.device_count, cuda_uuids=cuda_uuids)
    event = dict(stage='NATIVE_QWEN_DRIVER_SHARED_TRAINING_GPU_BOUND', run_id=run['run_id'],
        execution_id=run['execution_id'], source_commit=run['source_commit'], producer_identity=_birth(os.getpid()),
        hostname=socket.gethostname(), incoming_cuda_visible_devices=incoming, **route, **actual,
        allocation_sha256=_sha(allocation_path), placement_sha256=_sha(placement_path),
        training_claim_sha256=_sha(claim_path), source_sha256=run['source_sha256'][SOURCE_PATH],
        ray_runtime_env_modified=False, child_actor_cuda_assignment_unchanged=True,
        scope='actual native question-Head driver routing within the admitted actor GPUs; no extra GPU reservation or GPU test',
        observed_unix=time.time())
    _write_local_once(base/RECEIPT_PATH, event)
    return event

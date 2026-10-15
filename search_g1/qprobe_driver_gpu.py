"""Run the native Head driver in a process born with an admitted GPU UUID.

The CPU Ray task only validates metadata and waits for this process. A fresh
Python interpreter connects to the same Ray cluster and runs the unchanged
trainer body. Actor GPU placement and the original two runtime env vars remain
unchanged; the Head shares the first admitted actor GPU without a reservation.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time


HEAD_KIND = 'qwen_native_lastblock_binary_v1'
SOURCE_PATH = 'search_g1/qprobe_driver_gpu.py'
ENTRY_PATH = 'verl/trainer/main_ppo_eco.py'
RECEIPT_PATH = 'logs/native-qwen-driver-gpu-route.json'
PROCESS_CLAIM = 'logs/native-qwen-driver-process-once.json'
PROCESS_EXIT = 'logs/native-qwen-driver-process-exit.json'
RAY_RUNTIME_ENV = {'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}
MAX_PAYLOAD_BYTES = 4 * 1024**2


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


def child_environment(route, environ):
    """Bind a copy for exec; never change the already running Ray worker."""
    _require(environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES') in (None, ''),
             'RAY_GLOBAL_CUDA_OVERRIDE_FORBIDDEN')
    result = dict(environ)
    result['CUDA_VISIBLE_DEVICES'] = route['gpu_uuid']
    result.pop('RAY_ADDRESS', None)
    return result


def verify_driver_visibility(route, *, environ, birth_visible, cuda_initialized, cuda_device_count, cuda_uuids):
    """Actual child-entry identity check, with no late visibility mutation."""
    _require(environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES') in (None, ''),
             'RAY_GLOBAL_CUDA_OVERRIDE_FORBIDDEN')
    _require(birth_visible == environ.get('CUDA_VISIBLE_DEVICES') == route['gpu_uuid'],
             'DRIVER_NOT_BORN_WITH_BOUND_GPU_UUID')
    _require(cuda_initialized() is False, 'CUDA_INITIALIZED_BEFORE_BOUND_DRIVER_ROUTE')
    _require(cuda_device_count() == 1 and cuda_uuids(1) == [route['gpu_uuid']],
             'ACTUAL_CUDA_UUID_DIFFERS_FROM_FIRST_ADMITTED_TRAINING_GPU')
    return dict(actual_cuda_runtime_uuids=[route['gpu_uuid']], cuda_visible_devices=route['gpu_uuid'],
                birth_cuda_visible_devices=birth_visible, visibility_changed_after_process_birth=False,
                cuda_initialized_before=False, tensor_allocation_performed=False, rng_seed_performed=False)


def _birth(pid):
    root = Path('/proc') / str(pid)
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    _require(root.stat().st_uid == os.getuid(), 'PROCESS_NOT_OWNED')
    return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), uid=root.stat().st_uid)


def _bound_route(config):
    """The same source, allocation, placement and live-shell checks in both processes."""
    settings = _config_value(config, 'question_probe', {}) or {}
    from reproduction.common8.native_allocation import LOCAL_BINDING_RECEIPT
    from reproduction.common8.topology import validate_base
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
    binding = dict(run_id=run['run_id'], execution_id=run['execution_id'], source_commit=run['source_commit'],
        allocation_sha256=_sha(allocation_path), placement_sha256=_sha(placement_path),
        training_claim_sha256=_sha(claim_path), source_sha256=run['source_sha256'][SOURCE_PATH],
        entry_source_sha256=run['source_sha256'][ENTRY_PATH])
    return base, route, binding


def capture_cpu_rng():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state())


def cpu_rng_sha256(state):
    """Stable identity of only Python, NumPy and Torch CPU RNG bytes."""
    digest = hashlib.sha256(repr(state['python']).encode())
    name, keys, position, gaussian, cached = state['numpy']
    digest.update(repr((name, str(keys.dtype), keys.shape, position, gaussian, cached)).encode())
    digest.update(keys.tobytes())
    digest.update(state['torch_cpu'].numpy().tobytes())
    return digest.hexdigest()


def restore_cpu_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    actual = cpu_rng_sha256(capture_cpu_rng())
    _require(actual == cpu_rng_sha256(state), 'CPU_RNG_TRANSFER_READBACK_CHANGED')
    return actual


def _cluster_identity(address, namespace):
    _require(isinstance(address, str) and address not in ('auto', 'local') and
             ':' in address and address.rsplit(':', 1)[1].isdigit() and
             isinstance(namespace, str) and bool(namespace), 'EXACT_PRIVATE_RAY_CLUSTER_REQUIRED')
    return dict(ray_gcs_address=address, ray_namespace=namespace)


def _spawn_driver_process(base, payload, environment):
    raw = pickle.dumps(payload, protocol=4)
    _require(len(raw) <= MAX_PAYLOAD_BYTES, 'DRIVER_CONFIG_RNG_PAYLOAD_OVERSIZE')
    sha = hashlib.sha256(raw).hexdigest()
    from reproduction.common8.native_allocation import _write_local_once
    claim = dict(stage='NATIVE_QWEN_DRIVER_FRESH_PROCESS_ONCE', **payload['binding'],
        **payload['cluster'], parent_identity=payload['parent_identity'], route=payload['route'],
        payload_sha256=sha, payload_bytes=len(raw), cpu_rng_sha256=payload['cpu_rng_sha256'],
        child_cuda_visible_devices=environment['CUDA_VISIBLE_DEVICES'], ray_gpu_resources_reserved=0,
        claimed_unix=time.time())
    _write_local_once(base/PROCESS_CLAIM, claim)
    child = subprocess.Popen([sys.executable, '-m', 'search_g1.qprobe_driver_gpu',
        '--native-driver-child', '--payload-sha256', sha], cwd=base/'repo', env=environment,
        stdin=subprocess.PIPE, start_new_session=False)
    child.communicate(raw)
    _write_local_once(base/PROCESS_EXIT, dict(stage='NATIVE_QWEN_DRIVER_FRESH_PROCESS_EXIT',
        **payload['binding'], parent_identity=payload['parent_identity'], child_pid=child.pid,
        exit_code=child.returncode, payload_sha256=sha, observed_unix=time.time()))
    _require(child.returncode == 0, 'FRESH_DRIVER_CHILD_EXIT_' + str(child.returncode))


def run_native_driver(config):
    """Native-only CPU Ray wrapper: no CUDA query, no retry and no GPU request."""
    settings = _config_value(config, 'question_probe', {}) or {}
    if _config_value(settings, 'head_kind') != HEAD_KIND:
        return False
    state = capture_cpu_rng()
    import ray
    _require(ray.get_gpu_ids() == [], 'WRAPPER_MUST_REMAIN_A_CPU_RAY_TASK')
    base, route, binding = _bound_route(config)
    context = ray.get_runtime_context()
    cluster = _cluster_identity(context.gcs_address, context.namespace)
    payload = dict(schema_version=1, config=config, cpu_rng=state, cpu_rng_sha256=cpu_rng_sha256(state),
        parent_identity=_birth(os.getpid()), route=route, binding=binding, cluster=cluster)
    _spawn_driver_process(base, payload, child_environment(route, os.environ))
    return True


def _child(raw, expected_sha):
    """Only stdin from the live owned wrapper can enter the original trainer body."""
    _require(len(raw) <= MAX_PAYLOAD_BYTES and hashlib.sha256(raw).hexdigest() == expected_sha,
             'DRIVER_STDIN_PAYLOAD_SHA_CHANGED')
    # Do not leave a driver/actor Ray job behind if its synchronous wrapper dies.
    # This precedes unpickling CPU tensors, which itself imports Torch.
    parent_identity = _birth(os.getppid())
    import ctypes
    import signal
    _require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) == 0,
             'PARENT_DEATH_SIGNAL_BINDING_FAILED')
    _require(_birth(os.getppid()) == parent_identity, 'WRAPPER_DIED_DURING_CHILD_BOOTSTRAP')
    payload = pickle.loads(raw)
    _require(payload.get('schema_version') == 1 and parent_identity == payload['parent_identity'],
             'WRAPPER_BIRTH_CHANGED')
    base, route, binding = _bound_route(payload['config'])
    cluster = _cluster_identity(**dict(address=payload['cluster']['ray_gcs_address'],
                                      namespace=payload['cluster']['ray_namespace']))
    claim = _json(base/PROCESS_CLAIM)
    _require(binding == payload['binding'] and route == payload['route'] and
             claim['payload_sha256'] == expected_sha and claim['parent_identity'] == payload['parent_identity'] and
             claim['route'] == route and all(claim[key] == value for key, value in {**binding, **cluster}.items()) and
             claim['cpu_rng_sha256'] == payload['cpu_rng_sha256'] == cpu_rng_sha256(payload['cpu_rng']),
             'FRESH_DRIVER_CLAIM_CHANGED')
    initial_environment = dict(item.split(b'=', 1) for item in Path('/proc/self/environ').read_bytes().split(b'\0')
                               if b'=' in item)
    born_visible = initial_environment.get(b'CUDA_VISIBLE_DEVICES', b'').decode()
    _require(born_visible == os.environ.get('CUDA_VISIBLE_DEVICES') == route['gpu_uuid'],
             'DRIVER_NOT_BORN_WITH_BOUND_GPU_UUID')
    import ray
    _require(not ray.is_initialized(), 'CHILD_MUST_CONNECT_WITHOUT_STARTING_A_RAY_CLUSTER')
    ray.init(address=cluster['ray_gcs_address'], namespace=cluster['ray_namespace'], runtime_env=RAY_RUNTIME_ENV)
    try:
        context = ray.get_runtime_context()
        _require(_cluster_identity(context.gcs_address, context.namespace) == cluster and ray.get_gpu_ids() == [],
                 'CHILD_NOT_CONNECTED_TO_ORIGINAL_CPU_DRIVER_CLUSTER')
        import torch
        from reproduction.common8.native_allocation import cuda_uuids, _write_local_once
        actual = verify_driver_visibility(route, environ=os.environ, birth_visible=born_visible,
            cuda_initialized=torch.cuda.is_initialized, cuda_device_count=torch.cuda.device_count, cuda_uuids=cuda_uuids)
        from verl.trainer.main_ppo_eco import run_main_task
        restored_sha = restore_cpu_rng(payload['cpu_rng'])
        _require(restored_sha == payload['cpu_rng_sha256'], 'ORIGINAL_DRIVER_CPU_RNG_NOT_RESTORED')
        _write_local_once(base/RECEIPT_PATH, dict(stage='NATIVE_QWEN_DRIVER_SHARED_TRAINING_GPU_BOUND',
            **binding, **cluster, **route, **actual, producer_identity=_birth(os.getpid()),
            parent_identity=payload['parent_identity'], hostname=socket.gethostname(),
            payload_sha256=expected_sha, cpu_rng_sha256=restored_sha, original_driver_cpu_rng_restored=True,
            driver_process_kind='fresh_python_exec_with_birth_bound_cuda_visibility',
            ray_runtime_env_modified=False, child_actor_cuda_assignment_unchanged=True,
            scope='actual native driver shares an admitted actor GPU; no extra GPU reservation or GPU test',
            observed_unix=time.time()))
        run_main_task(payload['config'])
    finally:
        ray.shutdown()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-driver-child', action='store_true', required=True)
    parser.add_argument('--payload-sha256', required=True)
    args = parser.parse_args()
    _child(sys.stdin.buffer.read(MAX_PAYLOAD_BYTES+1), args.payload_sha256)

"""Verify real native allocation or the explicitly assigned local A800 resources.

Slurm's requested GRES alone is insufficient: also check RUNNING/node binding,
visible CUDA devices, no other active allocation on this node, and no foreign
compute applications. The local branch binds only the authorized GPUs and actual
PID birth/host; foreign jobs outside that assignment do not affect admission.
The explicit shared8 native branch separately proves CPU/memory allocation and
actual shared CUDA UUIDs, without claiming Slurm reserved the shared GPUs twice.
This is resource evidence, not model/GRPO acceptance.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


def fields(text):
    result=dict(re.findall(r'(\w+)=(\S+)', text))
    # Newer Slurm distinguishes requested from actually allocated resources.
    # Never use ReqTRES to manufacture allocation evidence.
    if 'TRES' not in result and result.get('AllocTRES') not in (None,'(null)'):
        result['TRES']=result['AllocTRES']
    return result


def allocated_indices(detail, node, count=8):
    line=next((line for line in detail.splitlines() if 'Nodes='+node+' ' in line), '')
    m=re.search(r'GRES=gpu:(?:h100:)?(\d+)\(IDX:([^)]*)\)',line)
    assert count in (4,8) and m and int(m.group(1))==count, 'Actual Slurm allocation GPU count differs'
    indices=[]
    for item in m.group(2).split(','):
        if '-' in item:
            lo,hi=map(int,item.split('-'));indices.extend(range(lo,hi+1))
        else:indices.append(int(item))
    assert len(indices)==count and len(set(indices))==count, 'Invalid actual Slurm GPU indices'
    return indices


def cuda_uuids(count=8):
    if type(count) is not int or count < 1:
        raise ValueError('CUDA UUID count must be a positive integer')
    import ctypes
    import importlib.util
    import sysconfig
    import uuid
    path=Path(sysconfig.get_paths()['purelib'])/'nvidia/cuda_runtime/lib/libcudart.so.12'
    if not path.is_file():
        spec=importlib.util.find_spec('nvidia.cuda_runtime')
        assert spec is not None and spec.submodule_search_locations, 'Installed CUDA runtime package is unavailable'
        path=next((Path(root)/'lib/libcudart.so.12' for root in spec.submodule_search_locations
                   if (Path(root)/'lib/libcudart.so.12').is_file()),None)
        assert path is not None, 'Installed CUDA12 runtime shared library is unavailable'
    lib=ctypes.CDLL(str(path))
    # CUDA 12 exports properties_v2, not cudaDeviceGetUuid. Verify the
    # installed header's ABI prefix before reading its UUID field.
    header=(path.parents[1]/'include/driver_types.h').read_text()
    header=re.sub(r'/\*.*?\*/', '', header, flags=re.S)
    assert re.search(r'struct\s+(?:__device_builtin__\s+)?cudaDeviceProp\s*\{\s*char\s+name\[256\];\s*cudaUUID_t\s+uuid\s*;',header), 'Unknown CUDA property ABI prefix'
    call=lib.cudaGetDeviceProperties_v2;call.argtypes=[ctypes.c_void_p,ctypes.c_int];call.restype=ctypes.c_int
    result=[]
    for ordinal in range(count):
        # The complete CUDA-12 structure is much smaller; reserve ample space
        # because only the verified name/UUID prefix is needed here.
        value=ctypes.create_string_buffer(65536)
        assert call(value,ordinal)==0, 'Cannot obtain actual CUDA device properties'
        result.append('GPU-'+str(uuid.UUID(bytes=value.raw[256:272])))
    return result


def nvml_minor_number(gpu_uuid):
    """NVML supports device minors even when nvidia-smi's query parser does not."""
    import ctypes
    library=ctypes.CDLL('libnvidia-ml.so.1')
    if library.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
    try:
        handle=ctypes.c_void_p();minor=ctypes.c_uint()
        get_handle=library.nvmlDeviceGetHandleByUUID
        get_handle.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)]
        get_minor=library.nvmlDeviceGetMinorNumber
        get_minor.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)]
        if get_handle(gpu_uuid.encode(),ctypes.byref(handle))!=0 or get_minor(handle,ctypes.byref(minor))!=0:
            raise RuntimeError('Cannot authenticate allocated GPU minor from UUID')
        return minor.value
    finally:
        library.nvmlShutdown()


def validate(job, node, visible, active_jobs, gpu_rows, applications, allocated, actual_cuda_uuids, allowed_applications=None):
    assert job['JobState']=='RUNNING', 'No active native allocation'
    assert job['NodeList']==node and job['NumNodes']=='1', 'Allocation is not this single node'
    assert job['TresPerNode'] in ('gres:gpu:h100:8','gres:gpu:8','gres/gpu:8'), 'Eight GPUs were not requested'
    assert len(visible)==8 and len(set(visible))==8, 'Expected eight unique allocated CUDA devices'
    assert set(active_jobs)=={job['JobId']}, 'Another native allocation shares the node'
    assert len(gpu_rows)==8 and len({r[1] for r in gpu_rows})==8, 'Expected eight physical GPUs'
    physical={int(r[0]):r[1] for r in gpu_rows}
    assert len(physical)==8
    expected=[physical[int(i)] if i.isdecimal() else i for i in visible]
    assert set(expected)=={physical[i] for i in allocated}, 'Visible GPUs differ from actual Slurm allocation'
    assert actual_cuda_uuids==expected, 'CUDA runtime UUID mapping differs from allocated visible devices'
    if allowed_applications is None:
        assert not applications, 'Foreign compute applications occupy this node'
    else:
        allowed = {(str(row[0]).strip(), str(row[1]).strip()) for row in allowed_applications}
        assert all((str(row[0]).strip(), str(row[1]).strip()) in allowed for row in applications), 'Unbound compute application appeared before shared launch'
    return True


def validate_native_four(job, node, visible, active_jobs, gpu_rows, applications, allocated, actual_cuda_uuids):
    """Four real reserved GPUs; foreign allocations may use other UUIDs only."""
    assert job['JobState']=='RUNNING' and job['NodeList']==node and job['NumNodes']=='1'
    assert job['TresPerNode'] in ('gres:gpu:h100:4','gres:gpu:4','gres/gpu:4')
    assert job['JobId'] in active_jobs and len(visible)==len(set(visible))==4
    physical={int(row[0]):row[1].strip() for row in gpu_rows}
    assert len(physical)==len(gpu_rows) and len(set(physical.values()))==len(physical)
    # Under ConstrainDevices, NVML indices are relative to the permitted
    # devices, while Slurm's global GRES IDs retain their device-file numbering.
    # Keep both namespaces and authenticate their common UUIDs. Never infer the
    # assigned physical GPUs from the cgroup-relative numeric indices alone.
    minors={int(row[3]):row[1].strip() for row in gpu_rows} if all(len(row)>3 for row in gpu_rows) else physical
    assert len(minors)==len(physical) and len(allocated)==len(set(allocated))==4 and set(allocated)<=set(minors), 'Assigned Slurm devices absent from actual minor-number inventory'
    expected=[physical[int(token)] if token.isdecimal() else token for token in visible]
    assigned={minors[index] for index in allocated}
    assert len(set(expected))==4 and set(expected)==assigned and actual_cuda_uuids==expected
    assert all(row[0].strip() not in assigned for row in applications), 'Assigned GPUs have foreign compute applications'
    return True


def admit(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run.get('execution_backend','slurm')=='ssh_local':
        return admit_local(base,run)
    import torch
    assert 'SLURM_JOB_ID' in os.environ, 'Must run under native Slurm'
    job_id=os.environ['SLURM_JOB_ID']
    node=socket.gethostname().split('.')[0]
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8' and run['completed_updates']==0
    detail=subprocess.check_output(['scontrol','show','job','-d',job_id],text=True)
    job=fields(detail)
    from reproduction.common8.shared_slurm import POLICY as SHARED_POLICY
    sharing=run.get('slurm_gpu_sharing_policy')==SHARED_POLICY
    from reproduction.common8.topology import world_size
    size=world_size(run)
    allocated=[] if sharing else allocated_indices(detail,node,size)
    assert job['JobId']==job_id
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    active=subprocess.check_output(['squeue','-h','-w',node,'-t','RUNNING,COMPLETING','-o','%i'],text=True).split()
    query='index,uuid,memory.total'
    raw=subprocess.check_output(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader,nounits'],text=True)
    rows=[[x.strip() for x in r] for r in csv.reader(io.StringIO(raw))]
    if size==4 and not sharing:
        rows=[row+[str(nvml_minor_number(row[1]))] for row in rows]
    raw_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    applications=[r for r in csv.reader(io.StringIO(raw_apps)) if r]
    assert torch.cuda.device_count()==size
    actual=cuda_uuids(size)
    observation=dict(run_id=run['run_id'],slurm_job_id=job_id,node=node,
        training_world_size=size,actual_allocation_detail=detail,visible_devices=visible,
        slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),actual_allocated_indices=allocated,
        physical_gpu_inventory=rows,actual_cuda_runtime_uuids=actual,
        active_node_jobs=active,foreign_compute_applications=applications,checked_unix=time.time(),
        accepted=False,scope='raw actual resource observation before admission; not acceptance')
    with (base/'logs/native-allocation-observation.json').open('x') as f:
        json.dump(observation,f,indent=2);f.flush();os.fsync(f.fileno())
    cohosts = run.get('shared_gpu_cohosts')
    allowed = None
    if cohosts is not None:
        assert run.get('shared_gpu_policy') == 'preserve_bound_existing_processes'
        from reproduction.a800.owned_processes import same
        allowed = [record['application'] for record in cohosts if same(record['process_identity'])]
    if sharing:
        from reproduction.common8.shared_slurm import validate as validate_shared
        assert allowed is not None
        for cohost,birth in run['shared_native_cohost_births'].items():
            if cohost in active:
                current=fields(subprocess.check_output(['scontrol','show','job','-o',cohost],text=True))
                assert all(current[k]==birth[k] for k in ('JobId','StartTime','NodeList','Command'))
                assert current['JobState'] in ('RUNNING','COMPLETING')
        validate_shared(job,node,visible,active,rows,applications,actual,allowed,run)
    elif size==4:
        assert run.get('execution_site')=='hpc3' and cohosts is None
        validate_native_four(job,node,visible,active,rows,applications,allocated,actual)
    else:
        validate(job,node,visible,active,rows,applications,allocated,actual,allowed)
    memory_raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    memory_rows = [[x.strip() for x in r] for r in csv.reader(io.StringIO(memory_raw))]
    proof=dict(accepted=True,run_id=run['run_id'],slurm_job_id=job_id,node=node,
        actual_job_state=job['JobState'],requested_gres=job.get('TresPerNode'),
        visible_devices=visible,actual_allocated_indices=allocated,actual_cuda_runtime_uuids=actual,
        actual_allocation_detail=detail,physical_gpu_inventory=rows,active_node_jobs=active,
        foreign_compute_applications=applications,shared_gpu_policy=run.get('shared_gpu_policy'),
        bound_cohosts=cohosts,actual_memory_inventory=memory_rows,checked_unix=time.time(),
        training_world_size=size, scope='native allocation and actual UUID mapping before project GPU work; not model/GRPO acceptance')
    if size==4 and not sharing:
        by_uuid={row[1]:int(row[0]) for row in rows}
        proof.update(gpu_index_mapping_policy='slurm_global_minor_to_process_nvml_uuid_v1',
            actual_nvml_indices=[by_uuid[value] for value in actual],
            actual_gpu_minor_inventory=rows,slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'))
    if sharing:
        proof.update(slurm_gpu_sharing_policy=SHARED_POLICY,slurm_reserved_gpu_count=0,
            actual_shared_physical_indices=list(range(8)),
            slurm_shared_authorization_sha256=run['slurm_shared_authorization_sha256'],
            scope='actual native CPU/memory allocation plus authorized shared physical8 CUDA UUIDs; no exclusive or second Slurm GPU reservation claimed')
    target=base/'logs/native-allocation-admission.json'
    with target.open('x') as f:
        json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
    return proof


LOCAL_PLACEMENT_POLICY='launch_time_four_idle_excluding_auxiliary_v1'
LOCAL_IDLE_MAX_USED_MIB=256
LOCAL_BINDING_MAX_AGE_SECONDS=120
LOCAL_BINDING_RECEIPT='logs/local-gpu-placement-binding.json'


def _local_inventory(rows):
    """Preserve actual NVML IDs; a physical index is never a device count."""
    physical={}
    for row in rows:
        assert len(row)==5, 'Incomplete actual local GPU inventory'
        index=int(row[0]);uuid=row[1].strip();total,used,util=map(int,row[2:])
        assert index>=0 and index not in physical and uuid.startswith('GPU-')
        assert total>0 and 0<=used<=total and 0<=util<=100
        physical[index]=dict(uuid=uuid,total_mib=total,used_mib=used,util_percent=util)
    assert physical and len({row['uuid'] for row in physical.values()})==len(physical), 'Invalid actual GPU UUID inventory'
    return physical


def _local_application_indices(physical,applications):
    by_uuid={row['uuid']:index for index,row in physical.items()}
    occupied=set()
    for row in applications:
        assert len(row)==2 and row[0].strip() in by_uuid and row[1].strip().isdecimal(), 'Unknown actual local compute application'
        occupied.add(by_uuid[row[0].strip()])
    return occupied


def select_idle_local_placement(rows,applications):
    """CPU-only launch selection: four idle cards and two idle service cards."""
    physical=_local_inventory(rows);auxiliary=[5,6]
    assert set(auxiliary)<=set(physical), 'Auxiliary GPUs5,6 absent from actual local inventory'
    occupied=_local_application_indices(physical,applications)
    idle={index for index,row in physical.items() if index not in occupied
          and row['used_mib']<=LOCAL_IDLE_MAX_USED_MIB and row['util_percent']==0}
    assert set(auxiliary)<=idle, 'Authorized local GPUs are occupied: '+repr(sorted(set(auxiliary)-idle))
    eligible=sorted(idle-set(auxiliary))
    assert len(eligible)>=4, 'Insufficient idle local training GPUs: '+repr(dict(idle_training_indices=eligible,occupied_indices=sorted(occupied)))
    training=eligible[:4]
    return training,sorted(set(physical)-set(training)-set(auxiliary))


def _query_local_gpu_state():
    def query(fields,kind):
        raw=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,
            '--format=csv,noheader,nounits'],text=True,timeout=30)
        return [[value.strip() for value in row] for row in csv.reader(io.StringIO(raw)) if row]
    rows=query('index,uuid,memory.total,memory.used,utilization.gpu','gpu')
    applications=query('gpu_uuid,pid','compute-apps')
    return rows,applications


def local_placement(run):
    """Validate the actual bound four-card assignment and its full complement."""
    from reproduction.common8.topology import world_size, execution_job_id
    assert execution_job_id(run) is None and world_size(run)==4
    assert run.get('gpu_placement_policy')==LOCAL_PLACEMENT_POLICY and run.get('gpu_placement_bound') is True, 'Local GPU placement is still an unbound preparation placeholder'
    training=run['training_physical_indices'];auxiliary=run['auxiliary_physical_indices']
    retriever=run['retriever_physical_indices'];probe=run['probe_physical_index']
    assert isinstance(training,list) and len(training)==4 and auxiliary==[5,6] and retriever==[5,6] and probe==6
    assigned=training+auxiliary
    assert all(type(i) is int for i in assigned) and len(set(assigned))==len(assigned)
    physical=_local_inventory(run['physical_gpu_inventory'])
    assert set(assigned)<=set(physical), 'Assigned cards absent from bound actual inventory'
    assert run['excluded_physical_indices']==sorted(set(physical)-set(assigned)), 'Excluded GPU indices differ from the actual assignment complement'
    assert run['assigned_gpu_uuids']=={str(i):physical[i]['uuid'] for i in assigned}, 'Bound physical GPU UUID mapping differs'
    assert run['training_gpu_uuids']==[physical[i]['uuid'] for i in training], 'Bound training UUID order differs'
    return training,auxiliary,retriever,probe


def _assert_local_idle(run,rows,applications,indices=None):
    training,auxiliary,_,_=local_placement(run)
    physical=_local_inventory(rows)
    bound=_local_inventory(run['physical_gpu_inventory'])
    assert {index:row['uuid'] for index,row in physical.items()}=={index:row['uuid'] for index,row in bound.items()}, 'Actual physical inventory changed since launch-time binding'
    occupied=_local_application_indices(physical,applications)
    assigned=training+auxiliary if indices is None else indices
    assert set(assigned)<=set(training+auxiliary)
    busy=[index for index in assigned if index in occupied or physical[index]['used_mib']>LOCAL_IDLE_MAX_USED_MIB or physical[index]['util_percent']!=0]
    assert not busy, 'Authorized local GPUs are occupied: '+repr(busy)
    return physical


def assert_local_training_idle(run):
    """Recheck the fixed selection before trainer launch; do not reselect."""
    rows,applications=_query_local_gpu_state()
    _assert_local_idle(run,rows,applications,indices=run['training_physical_indices'])


def assert_local_component_gpus_available(run,indices,owned_records):
    """Permit only idle cards or CUDA contexts of actual recorded PID births."""
    from reproduction.a800.owned_processes import same
    training,auxiliary,_,_=local_placement(run)
    assert all(type(index) is int for index in indices) and set(indices)<=set(training+auxiliary)
    rows,applications=_query_local_gpu_state();physical=_local_inventory(rows)
    bound=_local_inventory(run['physical_gpu_inventory'])
    assert {index:row['uuid'] for index,row in physical.items()}=={index:row['uuid'] for index,row in bound.items()}, 'Actual physical inventory changed before component launch'
    _local_application_indices(physical,applications)
    by_pid={record['pid']:record for record in owned_records}
    assert all(by_pid[record['pid']]==record for record in owned_records), 'Ambiguous recorded project PID births'
    for index in indices:
        contexts=[row for row in applications if row[0]==physical[index]['uuid']]
        if contexts:
            for _,pid in contexts:
                record=by_pid.get(int(pid))
                assert record is not None and same(record), 'Authorized local GPUs are occupied by an unowned or changed PID birth: '+repr(dict(index=index,pid=pid))
        else:
            assert physical[index]['used_mib']<=LOCAL_IDLE_MAX_USED_MIB and physical[index]['util_percent']==0, 'Authorized local GPUs are occupied without an authenticated project context: '+str(index)


def _write_local_once(path,data):
    raw=(json.dumps(data,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def bind_idle_local_placement(base):
    """Persist a launch-time selection before any CUDA work or launch claim.

    Repeated prelaunch calls retain the exact selected assignment, validate the
    immutable receipt, and query current occupancy. They never reselect a bound
    run. An interrupted binding or an aged/busy binding requires inspection.
    """
    from reproduction.a800.owned_processes import identity,publish
    from reproduction.common8.topology import validate_base
    base=Path(base).resolve();target=base/'logs/run-manifest.json'
    with (base/'gpu-placement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=json.loads(target.read_text());validate_base(base,run)
        assert run['execution_host']==socket.gethostname() and run['profile']=='common8' and run['completed_updates']==0
        assert run.get('gpu_placement_policy')==LOCAL_PLACEMENT_POLICY, 'Fresh launch-time placement policy is required'
        for name in ('native-allocation-admission.json','pipeline-launch-claim.json','training-launch-claim.json','process-ledger.json'):
            assert not (base/'logs'/name).exists(), 'GPU execution already admitted or claimed; never bind a running attempt'
        receipt_path=base/LOCAL_BINDING_RECEIPT
        rows,applications=_query_local_gpu_state()
        if run.get('gpu_placement_bound') is True:
            local_placement(run)
            assert run['gpu_placement_binding_receipt']==LOCAL_BINDING_RECEIPT
            assert receipt_path.is_file() and not receipt_path.is_symlink() and receipt_path.stat().st_uid==os.getuid()
            raw=receipt_path.read_bytes();receipt=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==run['gpu_placement_binding_sha256'], 'Immutable placement receipt changed'
            for key in ('run_id','execution_id','source_commit','training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','training_gpu_uuids','assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_bound_unix'):
                assert receipt[key]==run[key], 'Manifest differs from immutable placement receipt: '+key
            assert receipt['host']==socket.gethostname()
            assert 0<=time.time()-run['gpu_placement_bound_unix']<=LOCAL_BINDING_MAX_AGE_SECONDS, 'Launch-time GPU placement is stale; never reuse a staging-time selection'
            _assert_local_idle(run,rows,applications)
            return run
        assert run.get('gpu_placement_bound') is False and not receipt_path.exists(), 'Partial or unknown placement binding requires inspection'
        for name in ('prepared-launch-claim.json','qprobe-launch-claim.json','preclaim-retry-claim.json'):
            assert not (base/'logs'/name).exists(), 'Select idle GPUs before any launch claim'
        training,excluded=select_idle_local_placement(rows,applications)
        auxiliary=run['auxiliary_physical_indices']
        assert auxiliary==run['retriever_physical_indices']==[5,6] and run['probe_physical_index']==6
        physical=_local_inventory(rows);now=time.time()
        run.update(training_physical_indices=training,excluded_physical_indices=excluded,
            physical_gpu_inventory=rows,training_gpu_uuids=[physical[i]['uuid'] for i in training],
            assigned_gpu_uuids={str(i):physical[i]['uuid'] for i in training+auxiliary},
            gpu_placement_bound=True,gpu_placement_bound_unix=now,
            gpu_placement_binding_receipt=LOCAL_BINDING_RECEIPT,
            gpu_placement_semantics='actual launch-time idle physical GPU selection; no exclusive system GPU reservation or foreign process ownership claimed')
        local_placement(run)
        receipt=dict(role='actual_launch_time_idle_local_gpu_selection',gpu_placement_policy=LOCAL_PLACEMENT_POLICY,
            host=socket.gethostname(),selector_identity=identity(os.getpid()),
            observed_compute_applications=applications,idle_max_used_mib=LOCAL_IDLE_MAX_USED_MIB,
            idle_required_util_percent=0,cuda_initialized=False,admission=False,launch_claim=False,
            scope='read-only actual NVML selection before CUDA/admission/launch claims; foreign GPUs excluded and never signalled',
            **{key:run[key] for key in ('run_id','execution_id','source_commit','training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','training_gpu_uuids','assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_bound_unix')})
        run['gpu_placement_binding_sha256']=_write_local_once(receipt_path,receipt)
        publish(target,run)
        assert json.loads(target.read_text())==run, 'Actual run manifest did not persist the selection'
        return run


def bind_local_placement(base,run=None):
    """Compatibility entry for callers holding a prepared manifest."""
    bound=bind_idle_local_placement(base)
    if run is not None:
        assert run['run_id']==bound['run_id'] and run['execution_id']==bound['execution_id']
    return bound


def _local_cuda_probe(indices,expected_uuids):
    """Create short-lived real contexts on each assigned card, then exit."""
    assert len(indices)==len(expected_uuids)==len(set(expected_uuids)) and all(uuid.startswith('GPU-') for uuid in expected_uuids)
    code='''import json,os,sys,torch
from reproduction.common8.native_allocation import cuda_uuids
from reproduction.a800.owned_processes import identity
expected=json.loads(sys.argv[1])
indices=json.loads(sys.argv[2])
assert os.environ['CUDA_VISIBLE_DEVICES'].split(',')==list(map(str,indices))
count=len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
assert count==len(expected)
assert torch.cuda.device_count()==count
actual=cuda_uuids(count)
assert actual==expected, 'Actual assigned CUDA UUID mapping differs before context creation'
contexts=[torch.empty(1,dtype=torch.uint8,device='cuda:'+str(i)) for i in range(count)]
for i in range(count):torch.cuda.synchronize(i)
print(json.dumps(dict(actual_cuda_runtime_uuids=actual,context_logical_ordinals=list(range(count)),producer_identity=identity(os.getpid()))))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,indices)))
    return json.loads(subprocess.check_output([sys.executable,'-c',code,json.dumps(expected_uuids),json.dumps(indices)],text=True,env=env,timeout=60))


def admit_local(base,run):
    """Bind actual local process birth and assigned UUIDs without a Slurm fiction.

    GPUs outside the assigned set are inventoried for mapping only. Their jobs
    are neither an admission failure nor eligible for project cleanup.
    """
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.topology import validate_base
    validate_base(base,run)
    assert run['profile']=='common8' and run['completed_updates']==0
    training,auxiliary,retriever,probe=local_placement(run)
    persisted=json.loads((base/'logs/run-manifest.json').read_text())
    for key in ('run_id','execution_id','source_commit','training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','training_gpu_uuids','assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_binding_sha256'):
        assert persisted[key]==run[key], 'Admission differs from the persisted launch-time manifest: '+key
    receipt=base/run['gpu_placement_binding_receipt']
    assert run['gpu_placement_binding_receipt']==LOCAL_BINDING_RECEIPT and receipt.is_file() and not receipt.is_symlink()
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==run['gpu_placement_binding_sha256'], 'Placement receipt changed before native admission'
    assigned=training+auxiliary
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    assert visible==list(map(str,training)), 'Local pipeline must expose only its four training GPUs'
    rows,applications=_query_local_gpu_state()
    physical=_assert_local_idle(run,rows,applications)
    assigned_uuids={physical[i]['uuid'] for i in assigned}
    # The persisted manifest and fresh idle check precede this first CUDA work.
    cuda_probe=_local_cuda_probe(assigned,[physical[i]['uuid'] for i in assigned])
    assigned_actual=cuda_probe['actual_cuda_runtime_uuids']
    assert assigned_actual==[physical[i]['uuid'] for i in assigned], 'Actual assigned CUDA UUID mapping differs from assignment'
    assert cuda_probe['context_logical_ordinals']==list(range(len(assigned)))
    actual=assigned_actual[:len(training)]
    # The probe has exited; refuse a foreign arrival before admission and never
    # stop it. Bound UUIDs and selected indices remain unchanged after claims.
    deadline=time.monotonic()+30
    while True:
        rows,applications=_query_local_gpu_state()
        assert not any(row[0] in assigned_uuids and int(row[1])!=cuda_probe['producer_identity']['pid'] for row in applications), 'Authorized local GPUs are occupied by a foreign arrival after CUDA mapping proof'
        try:
            _assert_local_idle(run,rows,applications)
            break
        except AssertionError:
            assert time.monotonic()<deadline, 'Assigned CUDA proof contexts or telemetry did not become idle after probe exit'
            time.sleep(.5)
    producer=identity(os.getpid())
    proof=dict(accepted=True,profile='common8',run_id=run['run_id'],slurm_job_id=None,
        execution_backend='ssh_local',execution_id=run['execution_id'],node=socket.gethostname(),
        pipeline_identity=producer,training_world_size=len(training),
        visible_devices=list(map(str,assigned)),training_visible_devices=visible,
        actual_allocated_indices=assigned,training_physical_indices=training,
        auxiliary_physical_indices=auxiliary,retriever_physical_indices=retriever,
        probe_physical_index=probe,assigned_gpu_uuids={str(i):physical[i]['uuid'] for i in assigned},
        excluded_physical_indices=run['excluded_physical_indices'],
        gpu_placement_binding_sha256=run['gpu_placement_binding_sha256'],
        assigned_cuda_runtime_uuids=assigned_actual,actual_cuda_context_probe=cuda_probe,
        actual_cuda_runtime_uuids=actual,physical_gpu_inventory=rows,
        assigned_compute_applications=[r for r in applications if r[0] in assigned_uuids],unassigned_compute_applications=[r for r in applications if r[0] not in assigned_uuids],
        checked_unix=time.time(),scope='actual local PID birth/host and idle assigned GPU UUIDs before project launch; not model/GRPO acceptance')
    target=base/'logs/native-allocation-admission.json'
    with target.open('x') as f:
        json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(target.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    print(json.dumps(admit(p.parse_args().base)))

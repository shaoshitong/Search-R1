"""Bound Ray 2.40 RPC pools separately from scheduler CPU resources.

Ray's lazy GetServerCallExecutor and both GCS RPC pools otherwise use the host
hardware_concurrency()/4, even when ray.init(num_cpus=16) is explicit.
Source: ray-project/ray tag ray-2.40.0, src/ray/common/ray_config_def.h,
src/ray/common/ray_config.cc and src/ray/rpc/server_call.cc.
"""
import os

RPC_THREADS = dict(num_server_call_thread=2,
                   gcs_server_rpc_server_thread_num=2,
                   gcs_server_rpc_client_thread_num=2)


def environment():
    return {'RAY_'+key:str(value) for key,value in RPC_THREADS.items()}


def install_environment():
    # Set before Ray initialization; run.sh also exports these before Python
    # starts, covering C++ extension imports and all Ray child process births.
    os.environ.update(environment())
    return dict(RPC_THREADS)


def verify_environment():
    expected=environment()
    if any(os.environ.get(key)!=value for key,value in expected.items()):
        raise ValueError('PERIODIC_RAY_RPC_THREAD_BOUND_NOT_INHERITED')
    return dict(RPC_THREADS)


def init_options():
    """Use both cluster config and worker env, without changing CPU/GPU budgets."""
    return dict(_system_config=install_environment(),
                runtime_env=dict(env_vars=dict(TOKENIZERS_PARALLELISM='false',
                    RAYON_NUM_THREADS='1',NCCL_DEBUG='WARN',**environment())))

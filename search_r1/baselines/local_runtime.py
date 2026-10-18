"""Opt-in fail-closed single-host runtime boundary for this experiment."""
import ipaddress
import os
import socket
from urllib.parse import urlparse


def loopback(value):
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value == 'localhost'


def check_local_runtime(config, ray_nodes=None):
    if not config.baselines.get('local_runtime_only', False):
        return
    from .execution_identity import shared_h100
    shared = shared_h100()
    expected = 'h100-5' if shared else config.baselines.local_runtime_host
    if not expected or socket.gethostname().split('.')[0] != expected:
        raise ValueError('Runtime is not on the expected local host')
    direct=os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='hkust_direct'
    if direct:
        if expected!='xielab1' or os.getuid()!=1005 or not os.environ.get('SEARCH_R1_BASELINE_RUN_ID'):
            raise ValueError('Direct HKUST runtime identity changed')
        from .preflight import validate_inherited_gpu_leases
        validate_inherited_gpu_leases(config.trainer.n_gpus_per_node)
    elif shared:
        from .preflight import validate_inherited_gpu_leases
        validate_inherited_gpu_leases(config.trainer.n_gpus_per_node)
    elif os.environ.get('SLURM_JOB_NODELIST') != expected or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Local runtime requires the matching Slurm allocation')
    parsed = urlparse(config.retriever.url)
    host_addresses = {record[4][0] for record in socket.getaddrinfo(socket.gethostname(), None)}
    retriever_local = loopback(parsed.hostname or '') or (direct and parsed.hostname in host_addresses)
    if (parsed.scheme != 'http' or not retriever_local or parsed.username
            or parsed.password or parsed.path != '/retrieve'):
        raise ValueError('Retriever must be a local loopback HTTP service')
    address = os.environ.get('RAY_ADDRESS', '')
    if not (direct and address=='local') and not loopback(urlparse('http://' + address).hostname or ''):
        raise ValueError('Ray must use an explicit local loopback address')
    proxies = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    if any(os.environ.get(name) for name in proxies):
        raise ValueError('Online local runtime must not use an HTTP/SOCKS proxy')
    if ray_nodes is not None:
        local = {'127.0.0.1', '::1'}
        local.update(record[4][0] for record in socket.getaddrinfo(socket.gethostname(), None))
        alive = [node for node in ray_nodes if node.get('Alive')]
        if len(alive) != 1 or alive[0].get('NodeManagerAddress') not in local:
            raise ValueError('Ray contains a remote or unexpected alive node')

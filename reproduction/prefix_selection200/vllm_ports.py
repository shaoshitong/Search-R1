"""Bounded, process-held TCP namespaces for independent TP1 vLLM constructors.

The installed vLLM get_open_port releases its probe socket and, with VLLM_PORT,
can scan upward without a bound. Four constructors can therefore select the
same ephemeral port. Cooperating workers lease disjoint ranges with flock; only
constructor-time get_open_port aliases are replaced, without changing vLLM
files, inference settings, model loading concurrency, or sampling RNG.
"""
import atexit
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
from types import ModuleType

from .contract import WORLD

POOL_START = 20000
PORTS_PER_SLOT = 32
SLOTS = 128
_HELD = []
_CONSTRUCTOR = threading.RLock()


class PortNamespaceExhausted(RuntimeError):
    pass


def _available(start, end):
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(('', port))
            return port
        except OSError as error:
            if error.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
    raise PortNamespaceExhausted('No free TCP port inside owned bounded range %d..%d' % (start, end))


class PortNamespace:
    def __init__(self, fd, slot, start, end, identity_sha256, lock_path):
        self.fd, self.slot, self.start, self.end = fd, slot, start, end
        self.identity_sha256, self.lock_path = identity_sha256, str(lock_path)
        self.cursor = _available(start, end)
        self.initial_port = self.cursor
        self.ports, self.constructor_succeeded = [], False
        self._lock = threading.Lock()

    def get_open_port(self):
        with self._lock:
            port = _available(self.cursor, self.end)
            self.cursor = port + 1
            self.ports.append(port)
            return port

    def receipt(self):
        return dict(kind='bounded_process_held_vllm_tp1_tcp_namespace_v1',
            slot=self.slot, port_start=self.start, port_end=self.end,
            initial_vllm_port=self.initial_port, actual_get_open_port_calls=list(self.ports),
            identity_sha256=self.identity_sha256, lock_path=self.lock_path,
            lease_held_until_process_exit=self.fd is not None,
            constructor_succeeded=self.constructor_succeeded,
            model_initializations_serialized_across_processes=False)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def _release_all():
    for namespace in _HELD:
        namespace.close()


atexit.register(_release_all)


def _claim(config, rank, outer, phase, *, pool_start, span, slots, lock_root):
    if (type(rank) is not int or not 0 <= rank < WORLD or type(outer) is not int
            or not 1 <= outer <= 200 or phase not in ('prefix', 'continue')
            or type(pool_start) is not int or type(span) is not int or span < 1
            or type(slots) is not int or slots < WORLD or slots % WORLD
            or not 1024 <= pool_start <= pool_start + span * slots - 1 <= 65535):
        raise ValueError('Invalid bounded four-rank vLLM TCP namespace')
    identity = [config['run_id'], config.get('execution_id'), config.get('base'), outer, phase, rank]
    digest = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
    root = Path(lock_root or ('/tmp/search-g1-vllm-port-namespaces-v1-uid%d' % os.getuid()))
    root.mkdir(parents=True, exist_ok=True)
    lane_slots = slots // WORLD
    offset = int(digest[:16], 16) % lane_slots
    for index in range(lane_slots):
        slot = ((offset + index) % lane_slots) * WORLD + rank
        start = pool_start + slot * span; end = start + span - 1
        path = root / ('tcp-%d-%d.lock' % (start, end))
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        try:
            namespace = PortNamespace(fd, slot, start, end, digest, path)
        except PortNamespaceExhausted:
            os.close(fd)
            continue
        except BaseException:
            os.close(fd)
            raise
        _HELD.append(namespace)
        return namespace
    raise PortNamespaceExhausted('No available owned vLLM TCP range in rank%d lane' % rank)


def _aliases(value):
    for name, module in tuple(sys.modules.items()):
        if not (name == 'vllm' or name.startswith('vllm.')) or not isinstance(module, ModuleType):
            continue
        for key, item in tuple(vars(module).items()):
            if item is value:
                yield module, key


@contextmanager
def bounded_vllm_port_namespace(config, rank, *, outer, phase, utils_module=None,
        pool_start=POOL_START, span=PORTS_PER_SLOT, slots=SLOTS, lock_root=None):
    """Bound every constructor port lookup; retain the range lease until exit.

    Optional module/pool arguments exist for tiny CPU fixtures. Production uses
    the fixed host-wide pool and the actual installed vLLM module.
    """
    with _CONSTRUCTOR:
        utils = utils_module or importlib.import_module('vllm.utils')
        original = utils.get_open_port
        namespace = _claim(config, rank, outer, phase, pool_start=pool_start,
                           span=span, slots=slots, lock_root=lock_root)
        previous_port = os.environ.get('VLLM_PORT')
        bounded = namespace.get_open_port
        try:
            os.environ['VLLM_PORT'] = str(namespace.initial_port)
            # Replace imported-by-value aliases already loaded before this
            # context. Later imports receive this same bounded function.
            aliases = list(_aliases(original))
            if not any(module is utils and key == 'get_open_port' for module, key in aliases):
                aliases.append((utils, 'get_open_port'))
            for module, key in aliases:
                setattr(module, key, bounded)
            yield namespace
            namespace.constructor_succeeded = True
        finally:
            for module, key in list(_aliases(bounded)):
                setattr(module, key, original)
            if utils.get_open_port is bounded:
                utils.get_open_port = original
            if previous_port is None:
                os.environ.pop('VLLM_PORT', None)
            else:
                os.environ['VLLM_PORT'] = previous_port

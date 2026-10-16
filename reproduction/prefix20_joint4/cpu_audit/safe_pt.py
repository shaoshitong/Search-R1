"""Read Torch ZIP metadata without importing Torch or executing pickle globals.

Only this module's inert constructors are permitted. DataProto/TensorDict/Numpy
objects are inert metadata containers; no original __setstate__ is invoked.
Tensor data are never allocated as model tensors. Numeric scalar reads are
bounded to one serialized storage scalar, for Adam's actual step counters.
"""
import collections
import hashlib
import io
import math
import pickle
import struct
import zipfile
from dataclasses import dataclass


class UnsafeArchive(ValueError):
    pass


@dataclass(frozen=True)
class Storage:
    dtype: str
    key: str
    location: str
    elements: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


class Inert:
    """Records state; never imports or invokes an experiment class."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.state = None

    def __setstate__(self, state):
        self.state = state


class Dtype:
    def __init__(self, name):
        self.name = name


DTYPES = {
    'ByteStorage': ('B', 1), 'CharStorage': ('b', 1),
    'ShortStorage': ('h', 2), 'IntStorage': ('i', 4),
    'LongStorage': ('q', 8), 'HalfStorage': ('e', 2),
    'FloatStorage': ('f', 4), 'DoubleStorage': ('d', 8),
    'BoolStorage': ('?', 1), 'BFloat16Storage': (None, 2),
    'UntypedStorage': (None, 1),
}


def rebuild(storage, offset, size, stride, *ignored):
    if not isinstance(storage, Storage):
        raise UnsafeArchive('Tensor storage is not a validated persistent reference')
    if type(offset) is not int or offset < 0:
        raise UnsafeArchive('Tensor offset invalid')
    if not all(type(x) is int and x >= 0 for x in size):
        raise UnsafeArchive('Tensor shape invalid')
    if len(size) != len(stride) or not all(type(x) is int and x >= 0 for x in stride):
        raise UnsafeArchive('Tensor stride invalid')
    return Tensor(storage, offset, tuple(size), tuple(stride))


def parameter(data, *ignored):
    if not isinstance(data, Tensor):
        raise UnsafeArchive('Parameter is not inert tensor metadata')
    return data


def latin1_bytes(value, encoding):
    if type(value) is not str or encoding not in ('latin1','latin-1'):
        raise UnsafeArchive('Unsupported pickle byte reconstruction')
    return value.encode('latin1')


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '_codecs' and name == 'encode':
            return latin1_bytes
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module == 'torch' and name in DTYPES:
            return Dtype(name)
        if module == 'torch' and name == 'Size':
            return tuple
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_tensor_v3'):
            return rebuild
        if module == 'torch._utils' and name in ('_rebuild_parameter', '_rebuild_parameter_with_state'):
            return parameter
        if (module, name) in {
            ('verl.protocol', 'DataProto'), ('tensordict._td', 'TensorDict'),
            ('numpy', 'dtype'), ('numpy', 'ndarray'),
            ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
            ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
        }:
            return Inert
        raise UnsafeArchive('Disallowed pickle global: ' + module + '.' + name)

    def persistent_load(self, value):
        if (not isinstance(value, tuple) or len(value) != 5 or value[0] != 'storage' or
                not isinstance(value[1], Dtype) or not isinstance(value[2], str) or
                not value[2].isdigit() or not isinstance(value[3], str) or
                type(value[4]) is not int or value[4] < 0):
            raise UnsafeArchive('Disallowed persistent reference')
        return Storage(value[1].name, value[2], value[3], value[4])


class Archive:
    def __init__(self, path_or_bytes):
        self.zip = zipfile.ZipFile(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
        names = self.zip.namelist()
        if len(names) != len(set(names)) or any(n.startswith('/') or '..' in n.split('/') for n in names):
            raise UnsafeArchive('Ambiguous ZIP members')
        candidates = [n for n in names if n.endswith('/data.pkl')]
        if len(candidates) != 1:
            raise UnsafeArchive('Expected one modern Torch ZIP data.pkl')
        self.prefix = candidates[0][:-len('data.pkl')]
        if self.zip.getinfo(candidates[0]).file_size > 2 * 1024**3:
            raise UnsafeArchive('Pickle metadata exceeds CPU reader bound')
        byteorder = self.prefix + 'byteorder'
        if byteorder in names and self.zip.read(byteorder).strip() != b'little':
            raise UnsafeArchive('Only explicitly little-endian storage is supported')
        self.value = Restricted(io.BytesIO(self.zip.read(candidates[0]))).load()
        self.cache = {}

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *ignored):
        self.close()

    def member(self, storage):
        name = self.prefix + 'data/' + storage.key
        item = self.zip.getinfo(name)
        if item.file_size != storage.elements * DTYPES[storage.dtype][1]:
            raise UnsafeArchive('Storage byte count differs from persistent metadata')
        return name

    def scalar(self, value):
        if type(value) in (int, float):
            if not math.isfinite(value):
                raise UnsafeArchive('Nonfinite numeric scalar')
            return value
        if not isinstance(value, Tensor) or math.prod(value.shape) != 1:
            raise UnsafeArchive('Expected actual scalar tensor')
        code, width = DTYPES[value.storage.dtype]
        if code is None:
            raise UnsafeArchive('Unsupported scalar storage dtype')
        with self.zip.open(self.member(value.storage)) as stream:
            stream.read(value.offset * width)
            raw = stream.read(width)
        return struct.unpack('<' + code, raw)[0]

    def signature(self, value):
        """Exact metadata/storage-byte signature; no tensor/model reload."""
        if value is Inert:
            return ('inert_global_type',)
        if isinstance(value, Tensor):
            key = value.storage.key
            if key not in self.cache:
                h = hashlib.sha256()
                with self.zip.open(self.member(value.storage)) as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b''):
                        h.update(block)
                self.cache[key] = h.hexdigest()
            return ('tensor', value.storage.dtype, value.offset, value.shape, value.stride, self.cache[key])
        if isinstance(value, Inert):
            return ('inert', self.signature(getattr(value,'args',())), self.signature(getattr(value,'kwargs',{})), self.signature(getattr(value,'state',None)))
        if isinstance(value, dict):
            return tuple((self.signature(k), self.signature(v)) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return tuple(self.signature(v) for v in value)
        if isinstance(value, bytes):
            return ('bytes', len(value), hashlib.sha256(value).hexdigest())
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise UnsafeArchive('Unsupported inert value in structural signature')

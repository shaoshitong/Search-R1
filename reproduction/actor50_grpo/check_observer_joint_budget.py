"""Exercise the actual observer joint reader in memory; no SSH or disk payloads."""
import io
import json
import textwrap
from types import SimpleNamespace

from .control import REMOTE


class _Reader(io.BytesIO):
    def fileno(self):
        return 0


class _Path:
    def __init__(self, raw, declared_size=None):
        self.raw = raw
        self.size = len(raw) if declared_size is None else declared_size

    def __truediv__(self, name):
        assert name == 'latest-joint.json'
        return self

    def is_file(self):
        return True

    def open(self, mode):
        assert mode == 'rb'
        return _Reader(self.raw)


def _read(path, used=0):
    namespace = dict(
        b=path, p={'action': 'observe'}, json=json,
        pathlib=SimpleNamespace(Path=lambda value: value),
        os=SimpleNamespace(fstat=lambda fd: SimpleNamespace(st_size=path.size)),
        metadata_read_used=used, metadata_read_limit=63 * 1024**2,
    )
    helper = REMOTE[REMOTE.index('def file_bytes('):REMOTE.index('def read(')]
    joint = textwrap.dedent(REMOTE[REMOTE.index(' joint_metadata_limit='):REMOTE.index('\n if joint:')])
    exec(helper, namespace)
    exec(joint, namespace)
    return namespace


def main():
    compile(REMOTE, '<actual_remote_observer>', 'exec')
    # This is the failing boundary: a valid joint JSON just above the old8MiB cap.
    size = 8 * 1024**2 + 1
    raw = b'{"outer":151}'
    path = _Path(raw + b' ' * (size - len(raw)))
    value = _read(path)
    assert value['joint'] == {'outer': 151}
    assert value['metadata_read_used'] == size
    for candidate, used, expected in (
        (_Path(b'', 16 * 1024**2 + 1), 0, 'per-file budget'),
        (path, 63 * 1024**2 - size + 1, 'byte budget exhausted'),
    ):
        try:
            _read(candidate, used)
        except RuntimeError as error:
            assert expected in str(error)
        else:
            raise AssertionError('Observer budget guard did not reject oversized read')
    print(json.dumps(dict(
        status='PASS', old_8MiB_boundary_bytes=size,
        joint_per_file_limit_bytes=16 * 1024**2,
        aggregate_limit_bytes=63 * 1024**2,
        per_file_and_aggregate_rejection=True,
        remote_queries=0, disk_payload_bytes=0,
    )))


if __name__ == '__main__':
    main()

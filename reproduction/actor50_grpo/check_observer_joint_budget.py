"""Exercise the actual observer joint reader in memory; no SSH or disk payloads."""
import io
import hashlib
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

    def stat(self):
        return SimpleNamespace(st_size=self.size)

    def resolve(self):
        return self

    def is_symlink(self):
        return False

    def open(self, mode):
        assert mode == 'rb'
        return _Reader(self.raw)


def _read(path, used=0):
    namespace = dict(
        b=path, p={'action': 'observe'}, json=json, hashlib=hashlib,
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
        (path, 63 * 1024**2 - size + 1, 'byte budget exhausted'),
    ):
        try:
            _read(candidate, used)
        except RuntimeError as error:
            assert expected in str(error)
        else:
            raise AssertionError('Observer budget guard did not reject oversized read')
    value['os'] = SimpleNamespace(fstat=lambda fd: SimpleNamespace(st_size=16 * 1024**2 + 1))
    try:
        value['file_bytes'](_Path(b'', 16 * 1024**2 + 1), 16 * 1024**2)
    except RuntimeError as error:
        assert 'per-file budget' in str(error)
    else:
        raise AssertionError('Single-file guard did not reject oversized direct read')
    # Exercise the actual seek reader and JSON anchors without reading a large body.
    helpers = REMOTE[REMOTE.index('def file_range('):REMOTE.index('def once(')]
    exec(helpers, value)
    small = _Path(b'{"actor": {"completed_outer":151, "grounding_state":null}, "observed_unix":1, "outer":151}')
    value['os'] = SimpleNamespace(fstat=lambda fd: SimpleNamespace(st_size=small.size))
    head = value['file_range'](small, 0, 55)
    assert value['json_prefix'](head, 'grounding_state', b'}}')['actor']['completed_outer'] == 151
    tail = value['file_range'](small, -40, 40)
    assert value['json_suffix'](tail, 'observed_unix')['outer'] == 151
    pretty = json.dumps({'rank_receipts': [], 'sampling_actor_version': 'actor0150',
                         'scheduler_completed_outer': 151, 'status': 'complete'},
                        sort_keys=True, indent=2).encode()
    assert value['json_suffix'](pretty, 'sampling_actor_version')['scheduler_completed_outer'] == 151
    partial = _Path(b'{"event":"ACTUAL_ADAM_WRITE"}\n{"event":')
    value['os'] = SimpleNamespace(fstat=lambda fd: SimpleNamespace(st_size=partial.size))
    rows, prefix = value['ledger_rows'](partial)
    assert rows == [{'event': 'ACTUAL_ADAM_WRITE'}] and prefix is None
    # Execute the actual phase-selection block: periodic diagnostics contain
    # '-actor-' but must not be mistaken for an actor attempt UUID.
    selection = textwrap.dedent(REMOTE[REMOTE.index('  name=(value'):
        REMOTE.index('   advantage=small(directory')])
    for name, expected in (
        ('outer-000190-current-actor-diagnostic', None),
        ('outer-000190-probe-update', None),
        ('outer-000190-actor-' + 'a' * 32, 'a' * 32),
    ):
        phase = dict(value={'stage': {'stage': name}}, outer=190, b=__import__('pathlib').Path('/remote'), re=__import__('re'))
        exec(selection, phase)
        assert phase.get('attempt') == expected
        assert phase['value']['actor_inflight'] == []
    print(json.dumps(dict(
        status='PASS', old_8MiB_boundary_bytes=size,
        joint_per_file_limit_bytes=16 * 1024**2,
        aggregate_limit_bytes=63 * 1024**2,
        per_file_and_aggregate_rejection=True,
        bounded_seek_and_JSON_prefix_suffix=True,
        remote_queries=0, disk_payload_bytes=0,
    )))


if __name__ == '__main__':
    main()

"""Read one coherent snapshot of an atomically published owned process ledger.

This is historical evidence, never a current process-liveness admission. It
does not alter a ledger, signal a process, or relax the caller's birth checks.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def _read_once(path, expected_uid):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, 'O_CLOEXEC', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode), 'Owned ledger is not a regular file'
        assert before.st_uid == expected_uid, 'Owned ledger file owner differs'
        stream = os.fdopen(fd, 'rb')
        fd = None  # The context owns this FD even when a later read fails.
        with stream:
            chunks = []
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                chunks.append(block)
                digest.update(block)
            after = os.fstat(stream.fileno())
        # Atomic path replacement is allowed: the opened file is our snapshot.
        # In-place content/owner/mode changes violate the publication contract.
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_uid', 'st_mode')
        assert all(getattr(before, key) == getattr(after, key) for key in fields), \
            'Owned ledger changed in place during snapshot read'
        raw = b''.join(chunks)
        assert len(raw) == before.st_size, 'Owned ledger snapshot is incomplete'
        return raw, dict(path=str(path), sha256=digest.hexdigest(), bytes=len(raw),
                         device=before.st_dev, inode=before.st_ino,
                         uid=before.st_uid, mtime_ns=before.st_mtime_ns)
    finally:
        if fd is not None:
            os.close(fd)


def read_owned_ledger_snapshot(path, *, expected_uid=None, attempts=8, retry_delay=.1):
    """Retry only transient missing/stale reads; parse/schema failures stay fatal.

    An atomic publisher may replace the path while this FD is open. A complete
    old FD remains a valid point-in-time snapshot; its hash must bind the same
    bytes used for every subsequent rank identity check.
    """
    path = Path(path)
    if expected_uid is None:
        expected_uid = os.getuid()
    assert type(expected_uid) is int and expected_uid >= 0
    assert type(attempts) is int and 1 <= attempts <= 32
    assert type(retry_delay) in (int, float) and 0 <= retry_delay <= 1
    for attempt in range(1, attempts + 1):
        try:
            raw, proof = _read_once(path, expected_uid)
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ESTALE) or attempt == attempts:
                raise
            time.sleep(retry_delay)
            continue
        # Keep malformed JSON and all caller schema/run/birth/scheduler failures
        # outside the retry handler; a successful I/O cannot disguise corruption.
        document = json.loads(raw)
        assert isinstance(document, dict), 'Owned ledger is not an object'
        proof.update(read_attempts=attempt,
                     scope='one complete owned file snapshot; not current liveness')
        return document, proof

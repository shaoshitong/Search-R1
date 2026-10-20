"""New owner entry: retain its retriever while the finite CPU relay drains."""
import hashlib
import os
from pathlib import Path
try:
    import retrieval_only_owner as owner
except ModuleNotFoundError:
    from . import retrieval_only_owner as owner

CONTRACT = 'UPSTREAM_DEPENDENCY_LEASE_V1'
_busy_slots = owner.busy_slots
_controller = owner.controller


def dependency_path(base, run_id, proof):
    return Path(base) / 'pool/retrieval-relay-leases' / run_id / ('%d-%s' % (proof['pid'], proof['birth'])) / 'upstream.lock'


def busy_slots(pool, base, binding):
    busy = _busy_slots(pool, base, binding)
    fd = pool.try_lock(dependency_path(base, pool.RUN_ID, binding['owner']))
    if fd is None:
        busy.append(dict(relay_dependency='upstream', owner_pid=binding['owner']['pid']))
    else:
        os.close(fd)
    return busy


def controller(pool, args):
    publish = pool.publish
    def guarded_publish(path, value):
        if Path(path).name in ('allocation.json', 'owner-ready.json') and 'owner' in value:
            value['remote_relay_guard'] = dict(contract=CONTRACT,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_path=str(dependency_path(args.base, pool.RUN_ID, value['owner'])))
        publish(path, value)
    pool.publish = guarded_publish
    try:
        return _controller(pool, args)
    finally:
        pool.publish = publish


def main():
    owner.busy_slots = busy_slots
    owner.controller = controller
    owner.main()


if __name__ == '__main__':
    main()

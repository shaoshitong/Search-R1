"""Future guarded relay generation using bounded transport and health60."""
from contextlib import contextmanager
from pathlib import Path
from . import retrieval_relay_generation_control as original
from . import queued_relay_slots as slots
from . import rollover_paths

_bundle = original.bundle


@contextmanager
def relay_registry():
    previous, defaults = original.SLOTS, original.DEFAULT_SLOTS
    original.SLOTS, original.DEFAULT_SLOTS = slots.specifications(previous), slots.DEFAULT_SLOTS
    try:yield
    finally:original.SLOTS, original.DEFAULT_SLOTS = previous, defaults


def bundle(role='upstream', generation='owner-v1', slot=None, instance='r1', enabled_slots=slots.DEFAULT_SLOTS):
    with relay_registry():
        identity, files, manifest = _bundle(role, 'queued-' + generation, slot, instance, enabled_slots)
    for name in ('queued_generation_control.py', 'queued_health.py', 'queued_slot_consumer.py',
                 'queued_relay_owner.py', 'queued_relay_slots.py', 'retrieval_relay_backpressure.py',
                 'rollover_terminal.py', 'rollover_paths.py', 'predecessor_generation_terminal.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.old.digest(files[name]))
    source = files['relay_generation_bootstrap.py']
    assert source.count(b"str(stage/'retrieval_relay.py')") == 2
    source = source.replace(b"str(stage/'retrieval_relay.py')", b"str(stage/'retrieval_relay_backpressure.py')")
    assert source.count(b"str(stage/'retrieval_relay_owner.py')") == 1
    source = source.replace(b"str(stage/'retrieval_relay_owner.py')", b"str(stage/'queued_relay_owner.py')")
    assert source.count(b"str(stage/'relay_slot_consumer_worker.py')") == 1
    source = source.replace(b"str(stage/'relay_slot_consumer_worker.py')", b"str(stage/'queued_slot_consumer.py'),'--slot-mode','node10'")
    files['relay_generation_bootstrap.py'] = source
    compile(source, 'relay_generation_bootstrap.py', 'exec')
    identity = original.old.digest(original.old.encoded(dict(parent=identity, source_files=manifest['source_files'])))
    manifest.update(source_bundle_sha256=identity, readiness_timeout_seconds=60, transport_request_budget_seconds=59,
        files={name: dict(bytes=len(raw), sha256=original.old.digest(raw)) for name, raw in files.items()})
    return identity, files, manifest


def remote_source():
    source = original.REMOTE
    assert source.count('import retrieval_relay_owner as runtime') == 1
    return rollover_paths.remote_reads(source.replace('import retrieval_relay_owner as runtime', 'import queued_relay_owner as runtime'))


def main():
    remote = remote_source()
    previous_bundle, previous_remote = original.bundle, original.REMOTE
    try:
        original.bundle = bundle
        original.REMOTE = remote
        with relay_registry():original.main()
    finally:original.bundle, original.REMOTE = previous_bundle, previous_remote


if __name__ == '__main__':main()

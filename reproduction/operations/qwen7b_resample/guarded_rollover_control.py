"""Fresh guarded source owner consuming actual dynamic-cohort terminal proof."""
from contextlib import contextmanager
from pathlib import Path
from . import retrieval_only_control as original
from . import retrieval_remote_guard_control as guarded
from . import rollover_source_slots as slots
from . import rollover_paths

_slot_bundle = original.bundle


@contextmanager
def source_registry():
    previous, defaults = original.SLOTS, original.DEFAULT_SLOTS
    original.SLOTS, original.DEFAULT_SLOTS = slots.specifications(previous), slots.DEFAULT_SLOTS
    try:yield
    finally:original.SLOTS, original.DEFAULT_SLOTS = previous, defaults


def bundle(role='owner', generation='owner-v2', enabled_slots=slots.DEFAULT_SLOTS, slot=None, instance='r1'):
    assert role in ('owner', 'slot')
    with source_registry():
        if role == 'owner':identity, files, manifest = guarded.bundle(role, 'rollover-' + generation, enabled_slots, slot, instance)
        else:identity, files, manifest = _slot_bundle(role, 'guard-rollover-' + generation, enabled_slots, slot, instance)
    for name in ('guarded_rollover_control.py', 'guarded_rollover_slot_control.py', 'rollover_terminal.py',
                 'predecessor_generation_terminal.py', 'queued_slot_consumer.py', 'queued_health.py',
                 'rollover_source_slots.py', 'rollover_retrieval_owner.py', 'rollover_paths.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.old.digest(files[name]))
    if role == 'owner':
        source = files['generation_bootstrap.py']
        old = b"str(stage/'retrieval_remote_guard_owner.py')"
        assert source.count(old) == 1
        files['generation_bootstrap.py'] = source.replace(old, b"str(stage/'rollover_retrieval_owner.py')")
    else:
        source = files['generation_bootstrap.py']
        assert source.count(b"str(stage/'source_slot_consumer_worker.py')") == 1
        files['generation_bootstrap.py'] = source.replace(b"str(stage/'source_slot_consumer_worker.py')",
            b"str(stage/'queued_slot_consumer.py'),'--slot-mode','source5'")
    compile(files['generation_bootstrap.py'], 'generation_bootstrap.py', 'exec')
    identity = original.old.digest(original.old.encoded(dict(parent=identity, source_files=manifest['source_files'])))
    manifest.update(source_bundle_sha256=identity, predecessor_contract='OWN_BOUND_COHORT_TERMINAL_RELEASED_V1',
        files={name: dict(bytes=len(raw), sha256=original.old.digest(raw)) for name, raw in files.items()})
    return identity, files, manifest


def remote_source():
    source = original.REMOTE
    start = source.index("    assert PREDECESSOR and re.fullmatch")
    end = source.index("    context=dict(predecessor_proof=", start)
    assert "proof['status']=='OWN_PREDECESSOR_GENERATION_TERMINAL'" in source[start:end]
    replacement = """    assert PREDECESSOR and re.fullmatch(r'[0-9a-f]{64}',PREDECESSOR_SHA or'')and sha(PREDECESSOR)==PREDECESSOR_SHA
    import sys
    sys.path.insert(0,str(stage));import rollover_terminal as terminal
    proof=terminal.verify_source_proof(PREDECESSOR,PREDECESSOR_SHA,BASE,RUN_ID)
"""
    return rollover_paths.remote_reads(source[:start] + replacement + source[end:])


def main(role='owner'):
    remote = remote_source()
    previous_bundle, previous_remote = original.bundle, original.REMOTE
    try:
        original.bundle = bundle
        original.REMOTE = remote
        with source_registry():original.main(role)
    finally:original.bundle, original.REMOTE = previous_bundle, previous_remote


if __name__ == '__main__':main()

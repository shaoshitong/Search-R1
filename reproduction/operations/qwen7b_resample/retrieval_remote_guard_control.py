"""New guarded source-owner bundle; existing published controls are unchanged."""
from pathlib import Path
from . import retrieval_only_control as original

_bundle = original.bundle


def bundle(role='owner', generation='owner-v1', enabled_slots=original.DEFAULT_SLOTS, slot=None, instance='r1'):
    identity, files, manifest = _bundle(role, 'guard-' + generation, enabled_slots, slot, instance)
    for name in ('retrieval_remote_guard_owner.py', 'retrieval_remote_guard_control.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.old.digest(files[name]))
    if role == 'owner':
        old = b"str(stage/'retrieval_only_owner.py')"
        new = b"str(stage/'retrieval_remote_guard_owner.py')"
        assert files['generation_bootstrap.py'].count(old) == 1
        files['generation_bootstrap.py'] = files['generation_bootstrap.py'].replace(old, new)
    identity = original.old.digest(original.old.encoded(dict(parent=identity, guard_sources=manifest['source_files'])))
    manifest.update(source_bundle_sha256=identity, remote_relay_guard='UPSTREAM_DEPENDENCY_LEASE_V1',
        files={name: dict(bytes=len(raw), sha256=original.old.digest(raw)) for name, raw in files.items()})
    return identity, files, manifest


def main():
    original.bundle = bundle
    original.main()


if __name__ == '__main__':
    main()

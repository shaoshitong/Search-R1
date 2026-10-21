"""Isolated future TP4[1,3,4,6] under the failed GPU4 r1 anchor.

Only these new immutable stages select this declaration. The existing source
owner, supplementary bundles and runtime sources are not modified.
"""
from pathlib import Path
import sys
from . import source_supplementary_control as original

SLOT = 'gpu4-tp1'
SPECIFICATIONS = {SLOT: dict(indices=[1, 3, 4, 6], tp=4, seqs=32, tokens=8192, blocks=4096, reserve=.5)}
ANCHOR_UNIT = 'sst-q7b-g5-guard-rollover-owner-v2-gpu4-tp1-r1-70188b8e50fd.service'
_bundle, _remote_source = original.bundle, original.remote_source


def bundle(role='slot', generation='supp-residual-v1', enabled_slots=(), slot=SLOT, instance='r1'):
    assert role == 'slot' and slot == SLOT and generation == 'supp-residual-v1'
    previous = original.SPECIFICATIONS
    try:
        original.SPECIFICATIONS = SPECIFICATIONS
        # The legacy helper caps its prefixed generation at 30 characters.
        # Build its unchanged template, then bind the new full identity below.
        identity, files, manifest = _bundle(role, 'supp-v1', enabled_slots, slot, instance)
    finally:
        original.SPECIFICATIONS = previous
    for name in ('source_residual_supplementary_control.py', 'source_residual_supplementary_worker.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.original.old.digest(files[name]))
    old_unit = manifest['unit']
    for key in ('generation', 'directory', 'unit'):
        manifest[key] = manifest[key].replace('guard-rollover-supp-v1', 'guard-rollover-' + generation)
    source = files['generation_bootstrap.py']
    unit_literal = ('UNIT=' + repr(old_unit)).encode()
    assert source.count(unit_literal) == 1
    source = source.replace(unit_literal, ('UNIT=' + repr(manifest['unit'])).encode())
    before = b"str(stage/'source_supplementary_worker.py')"
    assert source.count(before) == 1
    source = source.replace(before, b"str(stage/'source_residual_supplementary_worker.py')")
    compile(source, 'generation_bootstrap.py', 'exec')
    files['generation_bootstrap.py'] = source
    artifacts = {name: dict(bytes=len(raw), sha256=original.original.old.digest(raw)) for name, raw in files.items()}
    identity = original.original.old.digest(original.original.old.encoded(dict(parent=identity,
        generation=manifest['generation'], source_files=manifest['source_files'], files=artifacts, anchor_unit=ANCHOR_UNIT)))
    manifest.update(source_bundle_sha256=identity, files=artifacts, anchor_unit=ANCHOR_UNIT,
                    physical_model_indices=[1, 3, 4, 6], residual_capacity_not_runtime_fit_proof=True)
    return identity, files, manifest


def remote_source():
    source = _remote_source()
    before = "    assert prior['role']=='slot'and prior['slot']==MANIFEST['slot']and prior['instance']=='r1'"
    assert source.count(before) == 1
    return source.replace(before, before + "\n    assert oldunit==MANIFEST['anchor_unit'],'Only the original failed GPU4 r1 lifecycle anchor is allowed'")


def main():
    saved = original.bundle, original.remote_source, sys.argv
    try:
        original.bundle, original.remote_source = bundle, remote_source
        args = sys.argv[1:]
        if '--generation' not in args: args += ['--generation', 'supp-residual-v1']
        if '--slot' not in args: args += ['--slot', SLOT]
        if '--enabled-slots' not in args: args += ['--enabled-slots', SLOT]
        sys.argv = [sys.argv[0], *args]
        original.main()
    finally:
        original.bundle, original.remote_source, sys.argv = saved


if __name__ == '__main__': main()

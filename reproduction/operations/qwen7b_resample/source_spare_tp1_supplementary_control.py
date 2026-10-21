"""Two fresh TP1 models using the terminal GPU6/7 source lifecycle anchors."""
from pathlib import Path
import sys
from . import source_supplementary_control as original

GENERATION = 'supp-tp1-v1'
SPECIFICATIONS = {
    'gpu6-tp1': dict(indices=[2], tp=1, seqs=64, tokens=8192, blocks=2048, reserve=1.5),
    'gpu7-tp1': dict(indices=[5], tp=1, seqs=128, tokens=16384, blocks=4096, reserve=1.5),
}
ANCHOR_UNITS = {slot: 'sst-q7b-g5-guard-rollover-owner-v2-' + slot + '-r1-70188b8e50fd.service'
                for slot in SPECIFICATIONS}
_bundle, _remote_source = original.bundle, original.remote_source


def bundle(role='slot', generation=GENERATION, enabled_slots=(), slot='gpu6-tp1', instance='r1'):
    assert role == 'slot' and slot in SPECIFICATIONS and generation == GENERATION
    previous = original.SPECIFICATIONS
    try:
        original.SPECIFICATIONS = SPECIFICATIONS
        identity, files, manifest = _bundle(role, generation, enabled_slots, slot, instance)
    finally:
        original.SPECIFICATIONS = previous
    for name in ('source_spare_tp1_supplementary_control.py', 'source_spare_tp1_supplementary_worker.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.original.old.digest(files[name]))
    source = files['generation_bootstrap.py']
    before = b"str(stage/'source_supplementary_worker.py')"
    assert source.count(before) == 1
    source = source.replace(before, b"str(stage/'source_spare_tp1_supplementary_worker.py')")
    compile(source, 'generation_bootstrap.py', 'exec')
    files['generation_bootstrap.py'] = source
    artifacts = {name: dict(bytes=len(raw), sha256=original.original.old.digest(raw)) for name, raw in files.items()}
    identity = original.original.old.digest(original.original.old.encoded(dict(parent=identity,
        source_files=manifest['source_files'], files=artifacts, anchor_unit=ANCHOR_UNITS[slot])))
    manifest.update(source_bundle_sha256=identity, files=artifacts, anchor_unit=ANCHOR_UNITS[slot],
                    physical_model_indices=SPECIFICATIONS[slot]['indices'], lifecycle_anchor_is_model_gpu=False)
    return identity, files, manifest


def remote_source():
    source = _remote_source()
    before = "    assert prior['role']=='slot'and prior['slot']==MANIFEST['slot']and prior['instance']=='r1'"
    assert source.count(before) == 1
    return source.replace(before, before + "\n    assert oldunit==MANIFEST['anchor_unit'],'Only the original exact failed source r1 lifecycle anchor is allowed'")


def main():
    saved = original.bundle, original.remote_source, sys.argv
    try:
        original.bundle, original.remote_source = bundle, remote_source
        args = sys.argv[1:]
        if '--generation' not in args: args += ['--generation', GENERATION]
        if '--slot' not in args: args += ['--slot', 'gpu6-tp1']
        if '--enabled-slots' not in args: args += ['--enabled-slots', ','.join(SPECIFICATIONS)]
        sys.argv = [sys.argv[0], *args]
        original.main()
    finally:
        original.bundle, original.remote_source, sys.argv = saved


if __name__ == '__main__': main()

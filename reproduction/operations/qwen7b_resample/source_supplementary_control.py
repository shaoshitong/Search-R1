"""Fresh once-only supplementary TP4s borrowing an unchanged source owner."""
import argparse
from pathlib import Path
import sys
from . import guarded_rollover_control as guarded
from . import retrieval_only_control as original

SPECIFICATIONS = {
    'gpu1-tp1': dict(indices=[0, 1, 2, 5], tp=4, seqs=32, tokens=8192, blocks=4096, reserve=.5),
    'gpu3-tp1': dict(indices=[3, 4, 6, 7], tp=4, seqs=32, tokens=8192, blocks=4096, reserve=.5),
}


def bundle(role='slot', generation='supp-v1', enabled_slots=(), slot=None, instance='r1'):
    assert role == 'slot' and slot in SPECIFICATIONS and generation.startswith('supp-')
    identity, files, manifest = guarded.bundle(role, generation, (slot,), slot, instance)
    for name in ('source_supplementary_worker.py', 'source_supplementary_control.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.old.digest(files[name]))
    source = files['generation_bootstrap.py']
    changes = {
        b"slot=ready['slots'][SLOT]": b"slot=manifest['supplementary_slots'][SLOT]",
        b"str(stage/'queued_slot_consumer.py'),'--slot-mode','source5'": b"str(stage/'source_supplementary_worker.py')",
        b"Q7B_SLOT_OWNER_READY=str(path),Q7B_SLOT_OWNER_READY_SHA256=context['owner_ready_sha256'],":
            b"Q7B_SLOT_OWNER_READY=str(path),Q7B_SLOT_OWNER_READY_SHA256=context['owner_ready_sha256'],\n"
            b"  Q7B_SUPPLEMENTARY_MANIFEST=str(stage/'manifest.json'),Q7B_SUPPLEMENTARY_MANIFEST_SHA256=hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest(),\n"
            b"  Q7B_SUPPLEMENTARY_CONTEXT=str(root/'submit-context.json'),Q7B_SUPPLEMENTARY_CONTEXT_SHA256=hashlib.sha256((root/'submit-context.json').read_bytes()).hexdigest(),",
    }
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    files['generation_bootstrap.py'] = source
    compile(source, 'generation_bootstrap.py', 'exec')
    manifest['supplementary_slots'] = {slot: SPECIFICATIONS[slot]}
    manifest['anchor_terminal_required'] = True
    manifest['owner_registry_unchanged'] = True
    artifacts = {name: dict(bytes=len(raw), sha256=original.old.digest(raw)) for name, raw in files.items()}
    identity = original.old.digest(original.old.encoded(dict(parent=identity, source_files=manifest['source_files'],
        files=artifacts, supplementary_slots=manifest['supplementary_slots'])))
    manifest.update(source_bundle_sha256=identity, files=artifacts)
    return identity, files, manifest


def remote_source():
    source = guarded.remote_source()
    before = "    context=dict(owner_ready=OWNER_READY,owner_ready_sha256=OWNER_READY_SHA,slot=MANIFEST['slot'])"
    assert source.count(before) == 1
    after = """    assert ANCHOR_INTENT and sha(ANCHOR_INTENT)==ANCHOR_INTENT_SHA
    anchor=read(ANCHOR_INTENT);prior=anchor['manifest'];oldunit=anchor['unit']
    assert prior['role']=='slot'and prior['slot']==MANIFEST['slot']and prior['instance']=='r1'
    assert prior['unit']==oldunit and oldunit!=UNIT and oldunit.endswith('-'+MANIFEST['slot']+'-r1-'+RUN_ID[:12]+'.service')
    assert anchor['context']['owner_ready_sha256']==OWNER_READY_SHA
    assert sha(anchor['context']['owner_ready'])==OWNER_READY_SHA
    terminal=query(oldunit);p=terminal['properties']
    assert terminal['returncode']==0 and p.get('Id')==oldunit and p.get('LoadState')=='loaded'
    assert p.get('MainPID')=='0'and p.get('ActiveState')in('inactive','failed'),'Original exact r1 anchor is not terminal'
    originalgroup='/system.slice/'+oldunit
    for proc in Path('/proc').glob('[0-9]*'):
     try:groups=[line.split(':',2)[-1]for line in(proc/'cgroup').read_text().splitlines()]
     except(FileNotFoundError,ProcessLookupError,PermissionError):continue
     assert not any(group==originalgroup or group.startswith(originalgroup+'/')for group in groups),'Original anchor cgroup still has a process'
    anchor_terminal=dict(unit=oldunit,intent_path=str(readable(ANCHOR_INTENT,BASE,PHYSICAL)),original_intent_path=ANCHOR_INTENT,intent_sha256=ANCHOR_INTENT_SHA,observed=p,epoch=time.time())
    context=dict(owner_ready=str(readable(OWNER_READY,BASE,PHYSICAL)),owner_ready_sha256=OWNER_READY_SHA,slot=MANIFEST['slot'],anchor_terminal=anchor_terminal)"""
    return source.replace(before, after)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--anchor-intent')
    parser.add_argument('--anchor-intent-sha256')
    own, remainder = parser.parse_known_args()
    if '--generation' not in remainder:remainder += ['--generation', 'supp-v1']
    prefix = 'ANCHOR_INTENT=' + repr(own.anchor_intent) + '\nANCHOR_INTENT_SHA=' + repr(own.anchor_intent_sha256) + '\n'
    previous_bundle, previous_remote, argv = original.bundle, original.REMOTE, sys.argv
    try:
        original.bundle = bundle
        original.REMOTE = prefix + remote_source()
        sys.argv = [sys.argv[0], *remainder]
        with guarded.source_registry():original.main('slot')
    finally:
        original.bundle, original.REMOTE, sys.argv = previous_bundle, previous_remote, argv


if __name__ == '__main__':main()

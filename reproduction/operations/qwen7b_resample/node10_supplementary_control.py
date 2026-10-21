"""Once-only GPU4 TP1; borrow a proven terminal GPU1 lifecycle anchor.

The live downstream owner/registry and all existing model slots stay intact.
Use action stage/submit/status; model fit is established only by its real
constructor/profile. No capacity probe or synthetic GPU work is launched here.
"""
import argparse
from pathlib import Path
import sys
from . import queued_generation_control as queued

original = queued.original
SLOT, ANCHOR = 'gpu4-tp1', 'gpu1-tp1'
SPECIFICATION = dict(indices=[4], tp=1, seqs=64, tokens=8192, blocks=4096, reserve=1.0)
ANCHOR_UNIT = 'sst-q7b-g5-relaygen-queued-owner-v2-slot-gpu1-tp1-r1-70188b8e50fd.service'


def bundle(role='slot', generation='supp-v1', slot=SLOT, instance='r1', enabled_slots=queued.slots.DEFAULT_SLOTS):
    assert role == 'slot' and slot == SLOT and generation.startswith('supp-')
    identity, files, manifest = queued.bundle(role, generation, ANCHOR, instance, enabled_slots)
    old_unit = manifest['unit']
    unit = old_unit.replace('-slot-' + ANCHOR + '-', '-slot-' + SLOT + '-')
    manifest.update(unit=unit, slot=SLOT,
        directory=manifest['directory'].replace('-slot-' + ANCHOR + '-', '-slot-' + SLOT + '-'))
    for name in ('node10_supplementary_worker.py', 'node10_supplementary_control.py'):
        files[name] = Path(__file__).with_name(name).read_bytes()
        manifest['source_files'][name] = dict(bytes=len(files[name]), sha256=original.old.digest(files[name]))
    source = files['relay_generation_bootstrap.py']
    changes = {
        ('UNIT=' + repr(old_unit)).encode(): ('UNIT=' + repr(unit)).encode(),
        ('SLOT=' + repr(ANCHOR)).encode(): ('SLOT=' + repr(SLOT)).encode(),
        b"slot=ready['slots'][SLOT]": b"slot=manifest['supplementary_slots'][SLOT]",
        b"str(stage/'queued_slot_consumer.py'),'--slot-mode','node10'": b"str(stage/'node10_supplementary_worker.py')",
        b"Q7B_RELAY_SLOT_NAME=SLOT,Q7B_SLOT_OWNER_DRAIN=":
            b"Q7B_NODE10_SUPPLEMENT_MANIFEST=str(stage/'manifest.json'),Q7B_NODE10_SUPPLEMENT_MANIFEST_SHA256=hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest(),\n"
            b"  Q7B_NODE10_SUPPLEMENT_CONTEXT=str(root/'submit-context.json'),Q7B_NODE10_SUPPLEMENT_CONTEXT_SHA256=hashlib.sha256((root/'submit-context.json').read_bytes()).hexdigest(),\n"
            b"  Q7B_RELAY_SLOT_NAME=SLOT,Q7B_SLOT_OWNER_DRAIN=",
    }
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    compile(source, 'relay_generation_bootstrap.py', 'exec')
    files['relay_generation_bootstrap.py'] = source
    manifest.update(supplementary_slots={SLOT: SPECIFICATION}, anchor_slot=ANCHOR, anchor_unit=ANCHOR_UNIT,
        anchor_terminal_required=True, owner_registry_unchanged=True, slot_hard_seconds=28800)
    artifacts = {name: dict(bytes=len(raw), sha256=original.old.digest(raw)) for name, raw in files.items()}
    identity = original.old.digest(original.old.encoded(dict(parent=identity, files=artifacts,
        supplementary_slots=manifest['supplementary_slots'], anchor_unit=ANCHOR_UNIT)))
    manifest.update(source_bundle_sha256=identity, files=artifacts)
    return identity, files, manifest


def remote_source():
    source = queued.remote_source()
    before = "assert MANIFEST['slot']in ready['slots']"
    assert source.count(before) == 1
    source = source.replace(before, "assert MANIFEST['anchor_slot']in ready['slots']")
    before = "    context=dict(downstream_owner_ready=DOWN_READY,downstream_owner_ready_sha256=DOWN_SHA)"
    assert source.count(before) == 1
    after = """    import node10_supplementary_worker as supplement
    anchor_terminal=supplement.anchor_binding(pool,BASE,ready,DOWN_SHA,ANCHOR_INTENT,ANCHOR_INTENT_SHA,ANCHOR_PROOF,ANCHOR_PROOF_SHA)
    context=dict(downstream_owner_ready=str(readable(DOWN_READY,BASE,PHYSICAL)),downstream_owner_ready_sha256=DOWN_SHA,anchor_terminal=anchor_terminal)"""
    source = source.replace(before, after)
    return source


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ('anchor-intent', 'anchor-intent-sha256', 'anchor-terminal-proof', 'anchor-terminal-sha256'):
        parser.add_argument('--' + name)
    own, remainder = parser.parse_known_args()
    if '--generation' not in remainder: remainder += ['--generation', 'supp-v1']
    if '--slot' not in remainder: remainder += ['--slot', SLOT]
    prefix = '\n'.join(key + '=' + repr(value) for key, value in dict(ANCHOR_INTENT=own.anchor_intent,
        ANCHOR_INTENT_SHA=own.anchor_intent_sha256, ANCHOR_PROOF=own.anchor_terminal_proof,
        ANCHOR_PROOF_SHA=own.anchor_terminal_sha256).items()) + '\n'
    saved = original.bundle, original.REMOTE, original.SLOTS, original.DEFAULT_SLOTS, sys.argv
    try:
        original.bundle, original.REMOTE = bundle, prefix + remote_source()
        # This parser-only declaration never enters the unchanged owner's slots.
        original.SLOTS = dict(queued.slots.specifications(original.SLOTS), **{SLOT: SPECIFICATION})
        original.DEFAULT_SLOTS = queued.slots.DEFAULT_SLOTS
        sys.argv = [sys.argv[0], 'slot', *remainder]
        original.main()
    finally:
        original.bundle, original.REMOTE, original.SLOTS, original.DEFAULT_SLOTS, sys.argv = saved


if __name__ == '__main__': main()

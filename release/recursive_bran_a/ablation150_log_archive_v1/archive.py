"""Supplement an accepted canonical archive; never open tensor/model payloads.

Only this package's separate supplement namespace is writable. Existing
canonical archive receipts, training groups and archive daemons are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from release.recursive_bran_a.ablation150_archive_v1 import archive as canonical

meta, closed = canonical.meta, canonical.closed
PROTOCOL = 'ablation150_closed_log_tree_supplement_v1'
TREE_PROTOCOL = 'recursive_ablation150_tree_summary_v1'
NAME = 'logs-trees-evaluation-v1'
MAX_TREE_PROVENANCE = 8 * 1024**2
MAX_INVENTORY = 128 * 1024**2
MAX_COMPLETE = 256 * 1024**2


def stable_json(path, root, cap=2 * 1024**2):
    row = canonical.metadata(path, root)
    meta.require(row['bytes'] <= cap, 'supplement metadata unexpectedly large')
    raw = Path(path).read_bytes()
    closed.same_file(row, canonical.metadata(path, root))
    meta.require(len(raw) == row['bytes'], 'metadata read was short')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def bound_canonical(path, expected_sha, source_commit, root, archive_root, ready):
    """Reuses exact exit0/wait0/release and prior full-SHA archive validators.

    Metadata-only verification here does not rehash or deserialize archived
    weights, and still works after authorized canonical native relocation.
    """
    canonical.verify_complete(path, expected_sha, root, archive_root, ready, full_sha=False)
    complete, digest = stable_json(path, archive_root)
    plan, checksum = stable_json(complete['inventory'], archive_root)
    meta.require(digest == expected_sha and checksum == complete['inventory_sha256'] and
                 plan['source_commit'] == source_commit and
                 Path(path) == archive_root / plan['group'] / plan['run_id'] / 'complete-sha-verified.json',
                 'not the exact accepted canonical source/group/run archive')
    canonical.small_closure(plan, root)
    return plan


def tree_files(root, plan, summary_root):
    """Use the existing accepted tree collector's exact 512 source pointers.

    Original event/root/child/snapshot JSON bytes are copied, not reconstructed.
    No new questions, predictions, reward calls or tensor inputs are generated.
    """
    meta.safe_path(summary_root, root / 'analysis', 'directory')
    base = root / plan['group']
    receipt, _ = meta.binding(root, plan['group'], plan['source_commit'])
    values, coverage = {}, {}
    for step in meta.KEYS:
        folder = summary_root / plan['group'] / f'outer-{step:03d}'
        accepted, _ = stable_json(folder / 'complete.json', root)
        meta.require(accepted.get('protocol') == TREE_PROTOCOL and
                     all(accepted.get(k) == plan[k] for k in ('group', 'run_id', 'config_sha256')) and
                     accepted.get('outer') == step and
                     set(accepted.get('files', {})) == {'summary.json', 'provenance-private.json'},
                     'sparse tree acceptance identity/files differ')
        loaded = {}
        for name, proof in accepted['files'].items():
            value, digest = stable_json(folder / name, root, MAX_TREE_PROVENANCE)
            meta.require(digest == proof['sha256'] and (folder/name).stat().st_size == proof['bytes'],
                         'accepted sparse tree source/provenance changed')
            loaded[name] = value
        summary, provenance = loaded['summary.json'], loaded['provenance-private.json']
        boundary = root / plan['group'] / f'checkpoints/training_state/global_step_{step}/post-evaluation/manifest.json'
        boundary_digest = canonical.sha(boundary)
        meta.require(all(summary.get(k) == plan[k] for k in
                         ('group', 'run_id', 'config_sha256', 'source_commit')) and
                     summary.get('protocol') == TREE_PROTOCOL and summary.get('outer') == step and
                     accepted['boundary_sha256'] == boundary_digest and
                     summary['native_boundary']['sha256'] == boundary_digest and
                     summary.get('tree', {}).get('status') == 'COMPLETE_CLOSED_TREE512' and
                     summary['tree']['questions'] == 512 and
                     provenance['configuration_receipt_sha256'] == canonical.sha(base/'receipts/configuration.json'),
                     'tree summary not bound to this exact formal512 milestone')
        sources = provenance.get('sparse_tree_sources', [])
        meta.require(len(sources) == 512, 'sparse source coverage is not512')
        directories = set()
        for source in sources:
            event = canonical.inside(source['events_path'], base/'output')
            relative = event.relative_to(base/'output')
            meta.require(len(relative.parts) == 5 and relative.parts[0].startswith('attempt-') and
                         relative.parts[1] == 'recursive-events' and relative.parts[3].isdecimal() and
                         event.name == 'events.jsonl' and event.parent not in directories,
                         'tree source not an exact unique own question directory')
            directories.add(event.parent)
            root_path = canonical.inside(source['root_path'], event.parent)
            meta.require(root_path.parent == event.parent and root_path.suffix == '.json' and
                         source.get('root') is not None, 'accepted tree root evidence unavailable')
            for path, proof in ((event, source['events']), (root_path, source['root'])):
                row = canonical.metadata(path, root)
                meta.require(row['bytes'] == proof['bytes'] and
                             canonical.stable_sha(row, root) == proof['sha256'],
                             'original accepted event/root bytes changed')
            for path in sorted(event.parent.iterdir()):
                meta.safe_path(path, root)
                meta.require(path.name == 'events.jsonl' or
                             re.fullmatch(r'[A-Za-z0-9_-]+\.json', path.name),
                             'unexpected file in sparse question directory; no binary copy')
                meta.require(str(path) not in values, 'duplicate source question across milestones')
                values[str(path)] = ('SPARSE_TREE_ORIGINAL', step)
        for name in ('summary.json', 'provenance-private.json', 'complete.json'):
            values[str(folder/name)] = ('ACCEPTED_TREE_SUMMARY_AND_PROVENANCE', step)
        coverage[str(step)] = dict(closed_questions=len(directories), boundary_sha256=boundary_digest)
    return values, coverage


def inventory(root, archive_root, plan, canonical_path, canonical_sha, summary_root, implementation):
    meta.require(meta.COMMIT.fullmatch(implementation), 'published supplement implementation commit required')
    base = root / plan['group']
    paths, coverage = tree_files(root, plan, summary_root)
    # Full native log retains original per-action/stage clocks and dev aggregate
    # prints. The trainer does not persist separate per-question dev predictions.
    log = base / 'logs/training.log'
    meta.safe_path(log, root)
    for path in sorted((base/'logs').rglob('*.log')):
        paths[str(path)] = ('NATIVE_LOG', None)
    for row in plan['closure_refs']:
        paths[row['path']] = ('CLOSURE_ORIGINAL', None)
    trajectories = {step: 0 for step in meta.KEYS}
    exclusions = []
    for path in sorted((base/'output').glob('attempt-*/trajectories/outer-*')):
        meta.safe_path(path, root, 'directory')
        match = re.fullmatch(r'outer-(\d{6})(?:-attempt-\d{6})?', path.name)
        meta.require(match is not None and int(match[1]) in meta.KEYS,
                     'trajectory output outside sparse milestone schedule')
        step = int(match[1])
        manifest, _ = stable_json(path/'manifest.json', root)
        identity = manifest.get('identity', {})
        meta.require(manifest.get('status') == 'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED' and
                     identity.get('outer_step') == step and identity.get('source') == plan['source_commit'] and
                     identity.get('config') == plan['config_sha256'],
                     'saved text trajectories not bound to exact frozen sparse milestone')
        text = path/'trajectories.jsonl.gz'
        row = canonical.metadata(text, root)
        meta.require(canonical.stable_sha(row, root) == manifest['payloads'].get(text.name),
                     'saved all-leaf text SHA differs from original manifest')
        paths[str(text)], paths[str(path/'manifest.json')] = ('SPARSE_LEAF_TEXT', step), ('SPARSE_LEAF_MANIFEST', step)
        trajectories[step] += 1
        exclusions.append(dict(path=str(path/'actor-inputs.pt'), reason='TENSOR_PAYLOAD_EXCLUDED_FROM_LOG_SUPPLEMENT'))
    meta.require(all(count > 0 for count in trajectories.values()), 'missing saved leaf text at a formal milestone')
    target = archive_root/'supplements'/plan['group']/plan['run_id']/NAME
    members = []
    for path, (kind, step) in sorted(paths.items()):
        row = canonical.metadata(path, root)
        members.append(dict(row, sha256=canonical.stable_sha(row, root), kind=kind, step=step,
            archive_path=str(target/'members'/Path(path).relative_to(root))))
    return dict(protocol=PROTOCOL, study_root=str(root), archive_root=str(archive_root),
        archive_directory=str(target), group=plan['group'], run_id=plan['run_id'],
        source_commit=plan['source_commit'], config_sha256=plan['config_sha256'],
        implementation_commit=implementation, helper_sha256=canonical.sha(__file__),
        canonical_complete_receipt=str(canonical_path), canonical_complete_receipt_sha256=canonical_sha,
        canonical_identity_verification='PRIOR_FULL_SHA_PLUS_FRESH_UNCHANGED_ARCHIVE_METADATA',
        closure_refs=plan['closure_refs'], accepted_study_row=plan['accepted_study_row'],
        tree_summary_root=str(summary_root), tree_coverage=coverage,
        leaf_text_packages={str(k): v for k, v in trajectories.items()}, exclusions=exclusions,
        members=members, total_member_bytes=sum(row['bytes'] for row in members),
        eval_scope='Original aggregate dev512 in native training.log/completed-outers and accepted summaries; '
                   'no separate per-question dev predictions are invented',
        tensor_payloads_copied=False, original_payloads_reclaimed=False)


def copy(root, archive_root, canonical_path, canonical_sha, source_commit, summary_root, implementation, ready):
    plan = bound_canonical(canonical_path, canonical_sha, source_commit, root, archive_root, ready)
    output = inventory(root, archive_root, plan, canonical_path, canonical_sha, summary_root, implementation)
    target = canonical.directory(output['archive_directory'], archive_root/'supplements')
    with canonical.operation_lock(target, archive_root/'supplements', '.supplement-copy.lock'):
        saved = target/'inventory.json'
        if saved.exists():
            previous, _ = stable_json(saved, archive_root, MAX_INVENTORY)
            canonical.plan_equal(previous, output)
        else:
            closed.durable_json(saved, output, exclusive=True)
        complete_path = target/'complete-sha-verified.json'
        if complete_path.exists():
            return verify(root, archive_root, complete_path, canonical.sha(complete_path), ready)
        missing = sum(row['bytes'] for row in output['members'] if not Path(row['archive_path']).exists())
        meta.require(closed.free_bytes(archive_root) > missing + meta.HEADROOM,
                     'supplement storage lacks measured remaining bytes plus16GiB; hold copy only')
        results = []
        last_progress = time.monotonic()
        for index, row in enumerate(output['members']):
            proof = canonical.copy_member(row, root, archive_root,
                lambda v: canonical.progress_record(target, dict(v, group=plan['group'], index=index)))
            record = dict(index=index, source=row['path'], destination=row['archive_path'],
                          bytes=row['bytes'], sha256=row['sha256'], verification=proof)
            results.append(record)
            if time.monotonic() - last_progress >= 10 or index + 1 == len(output['members']):
                canonical.progress_record(target, dict(phase='MEMBER_ACCEPTED', index=index,
                    copied_members=index+1, total_members=len(output['members']), time=time.time()))
                last_progress = time.monotonic()
        canonical.small_closure(plan, root)
        # Repeat discovery to detect new/removed files as well as changed bytes.
        canonical.plan_equal(output, inventory(root, archive_root, plan, canonical_path, canonical_sha,
                                               summary_root, implementation))
        for row in output['members']:
            actual = canonical.metadata(row['archive_path'], archive_root)
            meta.require(canonical.stable_sha(actual, archive_root) == row['sha256'], 'destination changed before complete')
        complete = dict(protocol=PROTOCOL, status='COMPLETE_FULL_SHA_ACCEPTED_COPY_ONLY', time=time.time(),
            inventory=str(saved), inventory_sha256=canonical.sha(saved), group=plan['group'], run_id=plan['run_id'],
            source_commit=source_commit, config_sha256=plan['config_sha256'], implementation_commit=implementation,
            total_members=len(results), total_member_bytes=output['total_member_bytes'], members=results,
            storage_ready=ready, source_before_after_verified=True, destination_full_sha_verified=True,
            tensor_payloads_copied=False, original_payloads_reclaimed=False, training_action_executed=False)
        closed.durable_json(complete_path, complete, exclusive=True)
        return dict(status=complete['status'], complete_receipt=str(complete_path),
            complete_receipt_sha256=canonical.sha(complete_path), total_members=len(results),
            total_member_bytes=complete['total_member_bytes'], training_action_executed=False)


def verify(root, archive_root, path, expected_sha, ready):
    path = canonical.inside(path, archive_root/'supplements')
    complete, digest = stable_json(path, archive_root, MAX_COMPLETE)
    meta.require(digest == expected_sha and complete.get('protocol') == PROTOCOL and
                 complete.get('status') == 'COMPLETE_FULL_SHA_ACCEPTED_COPY_ONLY' and
                 complete.get('source_before_after_verified') is True and
                 complete.get('destination_full_sha_verified') is True, 'supplement completion not accepted')
    plan, checksum = stable_json(complete['inventory'], archive_root, MAX_INVENTORY)
    expected = archive_root/'supplements'/plan['group']/plan['run_id']/NAME
    meta.require(checksum == complete['inventory_sha256'] and plan.get('protocol') == PROTOCOL and
                 plan.get('study_root') == str(root) and plan.get('archive_root') == str(archive_root) and
                 plan['group'] in meta.GROUPS and path.parent == expected == Path(plan['archive_directory']) and
                 complete['inventory'] == str(expected/'inventory.json') and
                 all(complete.get(k) == plan[k] for k in ('group', 'run_id', 'source_commit', 'config_sha256', 'implementation_commit')),
                 'supplement plan/identity/path differs')
    bound_canonical(Path(plan['canonical_complete_receipt']), plan['canonical_complete_receipt_sha256'],
                    plan['source_commit'], root, archive_root, ready)
    meta.require(complete['total_members'] == len(plan['members']) == len(complete['members']) and
                 complete['total_member_bytes'] == sum(row['bytes'] for row in plan['members']), 'supplement coverage differs')
    seen = set()
    for index, row in enumerate(plan['members']):
        source = canonical.inside(row['path'], root)
        dest = canonical.inside(row['archive_path'], expected/'members')
        meta.require(dest == expected/'members'/source.relative_to(root) and str(source) not in seen,
                     'duplicate/escaping supplement member map')
        seen.add(str(source))
        record = complete['members'][index]
        meta.require(all(record.get(k) == v for k, v in dict(index=index, source=str(source),
            destination=str(dest), bytes=row['bytes'], sha256=row['sha256']).items()), 'accepted member mapping differs')
        actual = canonical.metadata(dest, archive_root)
        meta.require(actual['bytes'] == row['bytes'] and canonical.stable_sha(actual, archive_root) == row['sha256'],
                     'supplement destination fresh SHA differs')
    return dict(status='SUPPLEMENT_FRESH_FULL_SHA_VERIFIED', complete_receipt=str(path),
        complete_receipt_sha256=digest, total_members=complete['total_members'],
        total_member_bytes=complete['total_member_bytes'], source_sha_mode='HISTORICAL_BEFORE_AFTER_ACCEPTANCE',
        destination_full_sha_verified_this_call=True, original_payloads_reclaimed=False, training_action_executed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('copy', 'verify'))
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--implementation-commit', required=True)
    parser.add_argument('--storage-ready', required=True)
    parser.add_argument('--storage-ready-sha256', required=True)
    parser.add_argument('--canonical-complete')
    parser.add_argument('--canonical-complete-sha256')
    parser.add_argument('--tree-summary-root', type=Path)
    parser.add_argument('--complete-receipt', type=Path)
    parser.add_argument('--complete-receipt-sha256')
    args = parser.parse_args()
    meta.require(meta.COMMIT.fullmatch(args.source_commit) and meta.COMMIT.fullmatch(args.implementation_commit),
                 'published science/implementation commits required')
    try:
        ready = canonical.storage_ready(args.storage_ready, args.storage_ready_sha256)
        if args.action == 'copy':
            meta.require(args.canonical_complete and meta.SHA.fullmatch(args.canonical_complete_sha256 or '') and
                         args.tree_summary_root is not None, 'copy requires explicit canonical receipt/SHA/tree summary root')
            value = copy(canonical.ROOT, canonical.ARCHIVE_ROOT, Path(args.canonical_complete),
                args.canonical_complete_sha256, args.source_commit, args.tree_summary_root, args.implementation_commit, ready)
        else:
            meta.require(args.complete_receipt and meta.SHA.fullmatch(args.complete_receipt_sha256 or ''),
                         'verify requires explicit supplement receipt/SHA')
            saved, _ = stable_json(args.complete_receipt, canonical.ARCHIVE_ROOT, MAX_COMPLETE)
            meta.require(saved['source_commit'] == args.source_commit and saved['implementation_commit'] == args.implementation_commit,
                         'CLI science/implementation differs from accepted supplement')
            value = verify(canonical.ROOT, canonical.ARCHIVE_ROOT, args.complete_receipt, args.complete_receipt_sha256, ready)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps(dict(protocol=PROTOCOL, status='SUPPLEMENT_HOLD_INCOMPLETE', error=str(error),
            training_action_executed=False, original_payloads_reclaimed=False), sort_keys=True), flush=True)
        return 2
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

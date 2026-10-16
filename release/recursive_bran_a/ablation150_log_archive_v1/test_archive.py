"""Small local CPU checks; no remote, model imports or production files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.recursive_bran_a.ablation150_archive_v1 import test_archive as fixtures
from . import archive as supplement

canonical = supplement.canonical
SOURCE, IMPLEMENTATION, save = fixtures.SOURCE, fixtures.IMPLEMENTATION, fixtures.save


class SupplementChecks(unittest.TestCase):
    def test_exact_sparse_scope_copy_resume_and_final_sha(self):
        # Existing tiny canonical fixtures exercise the unchanged closure and
        # actual-exit/wait/release binding, without running their entire suite.
        fixture = fixtures.ArchiveChecks()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        result, canonical_path, canonical_complete, canonical_plan = fixture.accepted()
        root, base, output, destination = fixture.root, fixture.base, fixture.output, fixture.destination
        summary_root = root/'analysis/tree-summary-fixture'
        (base/'logs').mkdir(exist_ok=True)
        (base/'logs/training.log').write_text('original clocks and dev512 aggregate\n')
        identity = {key: canonical_plan[key] for key in ('group', 'run_id', 'config_sha256', 'source_commit')}
        for step in canonical.meta.KEYS:
            sources = []
            for question in range(512):
                folder = output/f'recursive-events/batch-{step}/{question}'
                folder.mkdir(parents=True)
                event, saved_root = folder/'events.jsonl', folder/'root.json'
                event.write_text('{"event":"question_end"}\n')
                saved_root.write_text('{"state":"fixture"}\n')
                sources.append(dict(events_path=str(event), events=dict(bytes=event.stat().st_size,
                    sha256=canonical.sha(event)), root_path=str(saved_root),
                    root=dict(bytes=saved_root.stat().st_size, sha256=canonical.sha(saved_root))))
            folder = summary_root/fixture.group/f'outer-{step:03d}'
            boundary = base/f'checkpoints/training_state/global_step_{step}/post-evaluation/manifest.json'
            boundary_sha = canonical.sha(boundary)
            summary = dict(protocol=supplement.TREE_PROTOCOL, outer=step, **identity,
                native_boundary=dict(sha256=boundary_sha), tree=dict(status='COMPLETE_CLOSED_TREE512', questions=512))
            provenance = dict(sparse_tree_sources=sources,
                configuration_receipt_sha256=canonical.sha(base/'receipts/configuration.json'))
            save(folder/'summary.json', summary)
            save(folder/'provenance-private.json', provenance)
            files = {name: dict(bytes=(folder/name).stat().st_size, sha256=canonical.sha(folder/name))
                     for name in ('summary.json', 'provenance-private.json')}
            save(folder/'complete.json', dict(protocol=supplement.TREE_PROTOCOL, outer=step,
                **{k:identity[k] for k in ('group', 'run_id', 'config_sha256')},
                boundary_sha256=boundary_sha, files=files))
            leaf = output/f'trajectories/outer-{step:06d}-attempt-{step:06d}'
            leaf.mkdir(parents=True)
            (leaf/'trajectories.jsonl.gz').write_bytes(b'fixture gzip bytes')
            (leaf/'actor-inputs.pt').write_bytes(b'excluded tensor fixture')
            save(leaf/'manifest.json', dict(status='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED',
                identity=dict(outer_step=step, source=SOURCE, config=identity['config_sha256']),
                payloads={'trajectories.jsonl.gz':canonical.sha(leaf/'trajectories.jsonl.gz'),
                          'actor-inputs.pt':canonical.sha(leaf/'actor-inputs.pt')}))
        args = (root, destination, canonical_path, canonical.sha(canonical_path), SOURCE,
                summary_root, IMPLEMENTATION, fixture.ready)
        original = canonical.copy_member
        count = [0]
        def interrupted(row, *rest):
            value = original(row, *rest)
            count[0] += 1
            if count[0] == 2:
                raise OSError('fixture transfer interrupted after accepted member')
            return value
        with patch.object(canonical, 'copy_member', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'transfer interrupted'):
                supplement.copy(*args)
        actual = supplement.copy(*args)
        complete_path = Path(actual['complete_receipt'])
        complete = json.loads(complete_path.read_text())
        plan = json.loads(Path(complete['inventory']).read_text())
        self.assertTrue(complete_path.is_relative_to(destination/'supplements'))
        self.assertFalse((complete_path.parent/'files').exists())
        self.assertFalse(any(row['path'].endswith('.pt') for row in plan['members']))
        self.assertEqual(set(plan['tree_coverage']), {'5', '50', '100', '150'})
        self.assertEqual([value['closed_questions'] for value in plan['tree_coverage'].values()], [512]*4)
        self.assertEqual(len(plan['exclusions']), 4)
        self.assertTrue(all(Path(row['path']).is_file() for row in plan['members']))
        self.assertEqual(canonical.sha(canonical_path), args[3])
        verified = supplement.verify(root, destination, complete_path, canonical.sha(complete_path), fixture.ready)
        self.assertTrue(verified['destination_full_sha_verified_this_call'])
        reused = supplement.copy(*args)
        self.assertEqual(reused['status'], 'SUPPLEMENT_FRESH_FULL_SHA_VERIFIED')
        destination_member = Path(plan['members'][0]['archive_path'])
        destination_member.write_bytes(b'corrupt fixture')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            supplement.verify(root, destination, complete_path, canonical.sha(complete_path), fixture.ready)

    def test_escape_symlink_changed_source_and_bad_partial_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            root, target = base/'source', base/'destination'
            root.mkdir()
            target.mkdir()
            source = root/'log.jsonl'
            source.write_bytes(b'immutable original\n')
            row = dict(canonical.metadata(source, root), sha256=canonical.sha(source),
                       archive_path=str(target/'log.jsonl'))
            with self.assertRaisesRegex(ValueError, 'outside'):
                canonical.copy_member(dict(row, archive_path=str(base/'escape.log')), root, target, lambda v: None)
            (target/'link').symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                canonical.copy_member(dict(row, archive_path=str(target/'link/log.jsonl')), root, target, lambda v: None)
            (target/'log.jsonl.partial').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'partial prefix'):
                canonical.copy_member(row, root, target, lambda v: None)
            source.write_bytes(b'changed original\n')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                canonical.copy_member(row, root, target, lambda v: None)
            self.assertFalse((target/'log.jsonl').exists())


if __name__ == '__main__':
    unittest.main()

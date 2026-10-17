"""CPU tests for the real fit/adopt clock and checkpoint recovery metadata."""
import json
from pathlib import Path
import tempfile
import unittest

from . import grounding as g


class GroundingLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.source = self.root / 'old'; self.source.mkdir()
        self.base = self.root / 'new'; self.base.mkdir()
        self.config = dict(run_id='fresh', grounding_refresh_enabled=True,
            grounding_refit_rounds=[50, 100, 150], grounding_calibration_questions='/mnt/calibration.parquet',
            grounding_calibration_sha256='calibration', grounding_refit_visible_device=7,
            grounding_calibration_question_ids=['question-%d' % index for index in range(300)],
            model_files_sha256={'weights.safetensors': 'original'})
        self.zero_manifest = self.make_manifest(self.source, 0, 'old:0:real', 'original')
        self.config.update(grounding_manifest=str(self.zero_manifest),
            grounding_manifest_sha256=g.sha(self.zero_manifest), grounding_probe_version='old:0:real')
        self.zero = g.initialize(self.base, self.config)
        self.zero = self.adopt(self.zero)

    def tearDown(self): self.tmp.cleanup()

    def make_manifest(self, root, ordinal, version, encoder_hash):
        root.mkdir(parents=True, exist_ok=True)
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            (root / name).write_bytes(('fitted-' + str(ordinal)).encode())
        value = dict(run_id='old' if ordinal == 0 else 'fresh', round=ordinal, version=version,
            encoder=str(root / 'encoder'), encoder_sha256={'weights.safetensors': encoder_hash},
            head_sha256={name: g.sha(root / name) for name in ('commit_probe.pkl', 'boundary_probe.pkl')},
            commit=str(root / 'commit_probe.pkl'), boundary=str(root / 'boundary_probe.pkl'),
            label_proof=dict(question_count=300, question_ids=self.config['grounding_calibration_question_ids']),
            calibration_report=dict(unique_question_count=300))
        path = root / 'version-manifest.json'; path.write_text(json.dumps(value))
        return path

    def adopt(self, state):
        rows = [dict(rank=rank, ok=True, round=state['round'], version=state['version'], encoder=state['encoder'])
                for rank in range(8)]
        return g.adopt(self.base, self.config, state, rows)

    def next_fit(self, ordinal=50):
        path = self.make_manifest(self.base / 'grounding/rounds' / ('round-%06d' % ordinal),
                                  ordinal, 'fresh:%d:real' % ordinal, 'actor-%d' % ordinal)
        state = dict(g._manifest_state(self.config, path), source_actor_version='actor-%d' % ordinal)
        (path.parent / 'fit-complete.json').write_text(json.dumps(state))
        return state

    def joint(self, outer, state):
        point = self.base / 'outers' / ('outer-%06d' % outer); point.mkdir(parents=True)
        snapshot = point / 'grounding-state-at-completion.json'; snapshot.write_text(json.dumps(state))
        actor = dict(completed_outer=outer, actor_version='actor-%d' % outer,
                     grounding_state=state, grounding_probe_version=state['version'])
        return dict(outer=outer, actor=actor, grounding=state, grounding_snapshot=str(snapshot),
                    grounding_snapshot_sha256=g.sha(snapshot))

    def test_original_clock_before_and_after_milestones(self):
        self.assertEqual([g.expected_round(n) for n in (0, 49, 50, 99, 100, 149, 150, 200)],
                         [0, 0, 50, 50, 100, 100, 150, 150])

    def test_round0_is_a_real_import_and_fresh_actor_uses_it(self):
        self.assertTrue(self.zero['imported_actual_round0_fit'])
        self.assertFalse(self.zero['freshly_fitted_in_this_run'])
        bound = g.bind_attempt(self.config, self.zero, None, 1)
        self.assertEqual(bound['grounding_probe_version'], 'old:0:real')

    def test_one_unadopted_rank_cannot_advance(self):
        rows = [dict(rank=rank, ok=True, round=0, version=self.zero['version'], encoder=self.zero['encoder'])
                for rank in range(8)]
        rows[-1]['version'] = 'stale'
        with self.assertRaises(ValueError): g.adopt(self.base, self.config, self.zero, rows)

    def test_refit_actor50_changes_only_next_update51(self):
        actor50 = self.joint(50, self.zero)['actor']; new = self.adopt(self.next_fit())
        bound = g.bind_attempt(self.config, new, actor50, 51)
        self.assertEqual(bound['grounding_state']['round'], 50)
        with self.assertRaises(ValueError): g.bind_attempt(self.config, self.zero, actor50, 51)
        with self.assertRaises(ValueError): g.bind_attempt(self.config, new, actor50, 50)

    def test_wrong_actor_refit_does_not_restore(self):
        actor50 = self.joint(50, self.zero)['actor']; new = self.adopt(self.next_fit())
        new['source_actor_version'] = 'another-checkpoint'
        with self.assertRaises(ValueError): g.bind_attempt(self.config, new, actor50, 51)

    def test_later100_and150_transitions_preserve_the_actual_clock(self):
        state = self.zero
        for ordinal in (50, 100, 150):
            actor = self.joint(ordinal, state)['actor']
            state = self.adopt(self.next_fit(ordinal))
            bound = g.bind_attempt(self.config, state, actor, ordinal + 1)
            self.assertEqual(bound['grounding_state']['round'], ordinal)

    def test_recovery50_keeps_complete_actor_and_pending_fit(self):
        joint = self.joint(50, self.zero)
        saved = g.recovery_state(self.base, self.config, joint)
        self.assertEqual(saved['round'], 0)
        self.assertEqual(joint['actor']['completed_outer'], 50)

    def test_recovery_finished_fit_does_not_regenerate(self):
        joint = self.joint(50, self.zero); fitted = self.next_fit()
        recovered = g.recovery_state(self.base, self.config, joint)
        self.assertEqual(recovered['version'], fitted['version'])
        self.assertNotIn('adoption_receipt', recovered)

    def test_recovery_adopted50_keeps_exact_version(self):
        joint = self.joint(50, self.zero); adopted = self.adopt(self.next_fit())
        path = self.base / 'outers/outer-000050/grounding-refresh.json'
        path.write_text(json.dumps(dict(run_id='fresh', outer=50, required_before_outer=51, grounding=adopted)))
        recovered = g.recovery_state(self.base, self.config, joint)
        self.assertEqual(recovered, adopted)

    def test_dynamic_version_is_not_an_optimizer_scientific_reset(self):
        before = g.static_binding(self.config)
        after = g.static_binding(dict(self.config, grounding_probe_version='fresh:50:new'))
        self.assertEqual(before, after)

    def test_changed_immutable_manifest_is_rejected(self):
        path = Path(self.zero['manifest_path']); value = json.loads(path.read_text())
        value['encoder'] = '/mnt/unrelated'; path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): g.checked_state(self.config, self.zero)


if __name__ == '__main__': unittest.main()

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prefix_selection200 import storage


class StorageOwnership(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.parent = Path(self.temporary.name).resolve()
        self.base = self.parent/'search-g1-prefix-selection200-fixture'
        self.base.mkdir()
        self.patch = patch.object(storage, 'REMOTE_PARENT', self.parent)
        self.patch.start()
        self.run = 'owned-formal-run'

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def payload(self, path, data=b'dummy fixture, not tensor data'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return dict(path=str(path), bytes=len(data), sha256='a'*64)

    def actor(self, outer):
        attempt = self.base/'actor/outers'/('outer-%04d-attempt-abc'%outer)
        hf = attempt/'huggingface'
        weight = self.payload(hf/'model.safetensors')
        self.write(hf/'config.json', {'fixture': True})
        self.payload(attempt/'undeclared-evidence.pt')
        ranks = [self.payload(attempt/('rank-%d.pt'%r)) for r in range(8)]
        driver = self.payload(attempt/'driver.pt')
        receipt = dict(role='prefix_selection200_actor_complete', run_id=self.run, completed_outer=outer,
            source_commit='c'*40, actor_version='actor'+str(outer), state_path=str(attempt), state_dir=str(attempt),
            hf_model_dir=str(hf), model_dir=str(hf), actor_model_dir=str(hf),
            actual_adam_steps_per_rank=[20*outer]*8, scheduler_completed_outer=outer,
            milestone=outer in storage.MILESTONES, hf_files={'model.safetensors': weight},
            rank_states=ranks, driver_state=driver)
        self.write(attempt/'complete.json', receipt)
        return receipt

    def fixture(self, outer):
        config = dict(base=str(self.base), run_id=self.run, source_commit='c'*40,
                      outers=200, world_size=8, probe_updates_per_outer=10)
        self.write(self.base/'preparation/launch-config.json', config)
        self.write(self.base/'logs/process-ledger.json', dict(run_id=self.run,
            jobs={'retriever': dict(state='STARTED'), 'outer-%06d-actor'%outer: dict(state='STOPPED')}))
        actors = {n: self.actor(n) for n in range(1, outer+1)}
        for n in range(1, outer+1):
            root = self.base/'outers'/('outer-%06d'%n)
            snapshot = root/'probe-state-at-completion.pt'
            producer = self.payload(snapshot)
            probe = dict(run_id=self.run, completed_outer=n, actual_adam_steps_per_rank=[10*n]*8,
                         bytes=producer['bytes'], sha256=producer['sha256'])
            joint = dict(run_id=self.run, source_commit='c'*40, outer=n, actor=actors[n], probe=probe,
                         probe_snapshot=str(snapshot), rollout=dict(actor_model_dir=actors[max(1,n-1)]['hf_model_dir']))
            self.write(root/'complete.json', joint)
        current = self.base/'outers'/('outer-%06d'%outer)
        joint = storage.read(current/'complete.json')
        self.write(self.base/'latest-joint.json', joint)
        self.write(self.base/'actor/latest.json', joint['actor'])
        self.write(self.base/'probe/latest.json', joint['probe'])
        os.link(current/'probe-state-at-completion.pt', self.base/'probe/latest.pt')
        self.payload(self.base/'probe/head-outer-0000.pt')
        shards = []
        for rank in range(8):
            name = 'selected-features-rank-%d.pt'%rank
            item = self.payload(current/'probe-score'/name)
            shards.append(dict(rank=rank, file=name, count=320, bytes=item['bytes'], sha256=item['sha256']))
        manifest = dict(role='selected_actual_prefix_official_encoder_features', run_id=self.run,
                        outer=outer, count=2560, shards=shards, selection_sha256='b'*64)
        self.write(current/'probe-score/manifest.json', manifest)
        self.write(current/'probe-score/index.json', ['preserve identity metadata'])
        update = dict(status='SELECTED512G5_ONLINE_PROBE_10_ADAM_COMPLETE', run_id=self.run,
                      outer=outer, checkpoint=joint['probe'], new_adam_steps_per_rank=10,
                      actual_adam_steps_per_rank=[10*outer]*8, selection_sha256='b'*64,
                      scored_features_manifest_sha256=storage.small_sha(current/'probe-score/manifest.json'))
        self.write(current/'probe-update/complete.json', update)
        for name in ('prefix/states.jsonl', 'full/own-em.jsonl', 'full/trajectories.jsonl',
                     'questions.json', 'probe-update/metrics.json'):
            self.payload(current/name)
        return current, actors

    def test_prune_consumed_features_and_old_unreferenced_states_only(self):
        current, actors = self.fixture(3)
        result = storage.prune_completed(self.base, 3)
        self.assertEqual(result['status'], 'COMPLETED_RUN_OWNED_ROTATING_TENSORS_PRUNED')
        self.assertFalse((current/'probe-score/selected-features-rank-0.pt').exists())
        self.assertFalse((self.base/'outers/outer-000002/probe-state-at-completion.pt').exists())
        old = Path(actors[1]['state_dir'])
        self.assertFalse((old/'huggingface/model.safetensors').exists())
        self.assertFalse((old/'rank-0.pt').exists())
        self.assertTrue((old/'huggingface/config.json').exists())
        self.assertTrue((old/'complete.json').exists())
        self.assertTrue((old/'undeclared-evidence.pt').exists())
        # The current joint rollout references actor2; keep its complete state.
        self.assertTrue((Path(actors[2]['state_dir'])/'rank-0.pt').exists())
        self.assertTrue((Path(actors[3]['state_dir'])/'rank-0.pt').exists())
        for name in ('prefix/states.jsonl', 'full/own-em.jsonl', 'full/trajectories.jsonl',
                     'questions.json', 'probe-update/metrics.json', 'probe-score/manifest.json', 'probe-score/index.json'):
            self.assertTrue((current/name).exists())
        self.assertTrue((self.base/'probe/latest.pt').exists())
        self.assertTrue((self.base/'probe/head-outer-0000.pt').exists())
        self.assertFalse(result['tensor_contents_read_or_rehashed'])
        again = storage.prune_completed(self.base, 3)
        self.assertEqual(again['status'], 'EXISTING_STORAGE_PRUNE_RECEIPT_NO_REPLAY')

    def test_milestone_probe_and_actor_are_kept(self):
        _, actors = self.fixture(6)
        storage.prune_completed(self.base, 6)
        self.assertTrue((self.base/'outers/outer-000005/probe-state-at-completion.pt').exists())
        self.assertTrue((Path(actors[5]['state_dir'])/'rank-0.pt').exists())
        self.assertTrue((Path(actors[5]['state_dir'])/'huggingface/model.safetensors').exists())
        # Outer4 was the sampling input for milestone5; it is not itself a
        # retained checkpoint once latest-joint6 no longer references it.
        self.assertFalse((Path(actors[4]['state_dir'])/'rank-0.pt').exists())

    def test_external_symlink_is_rejected_before_any_deletion(self):
        current, actors = self.fixture(3)
        target = Path(actors[1]['state_dir'])/'rank-0.pt'
        external = self.parent/'old-run-retained.pt'
        external.write_bytes(target.read_bytes())
        target.unlink(); target.symlink_to(external)
        with self.assertRaises(ValueError):
            storage.prune_completed(self.base, 3)
        self.assertTrue(external.exists())
        self.assertTrue((current/'probe-score/selected-features-rank-0.pt').exists())
        self.assertFalse((current/'storage-prune-intent.json').exists())

    def test_wrong_actual_boundary_and_active_phase_preserve_everything(self):
        current, _ = self.fixture(3)
        probe = storage.read(self.base/'probe/latest.json')
        self.write(self.base/'probe/latest.json', dict(probe, actual_adam_steps_per_rank=[29]*8))
        with self.assertRaises(ValueError):
            storage.prune_completed(self.base, 3)
        self.write(self.base/'probe/latest.json', probe)
        self.write(self.base/'logs/process-ledger.json', dict(run_id=self.run,
                   jobs={'outer-000004-prefix': dict(state='STARTED')}))
        with self.assertRaises(RuntimeError):
            storage.prune_completed(self.base, 3)
        self.assertTrue((current/'probe-score/selected-features-rank-0.pt').exists())

    def test_pending_intent_is_not_replayed_and_other_namespace_is_rejected(self):
        current, _ = self.fixture(2)
        self.write(current/'storage-prune-intent.json', dict(run_id=self.run, outer=2))
        result = storage.prune_completed(self.base, 2)
        self.assertEqual(result['status'], 'EXISTING_STORAGE_PRUNE_INTENT_RECONCILE_REQUIRED')
        self.assertTrue((current/'probe-score/selected-features-rank-0.pt').exists())
        old = self.parent/'old-warmup-run'
        old.mkdir()
        with self.assertRaises(ValueError):
            storage.prune_completed(old, 2)


if __name__ == '__main__':
    unittest.main()

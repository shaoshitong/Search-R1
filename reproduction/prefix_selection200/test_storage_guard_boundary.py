"""Tiny CPU metadata/owned-file checks for retained partial-update pruning."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import storage
from .actor_safety import attempt_result, opportunity_result


def accounting(before, count):
    healthy = dict(allowed=True, reasons=[])
    guarded = dict(allowed=False, reasons=['negative_advantage_tail'])
    rows = []
    for step in range(20):
        cursor = before + min(step, count)
        wrote = step < count
        rows.append(opportunity_result(opportunity=step, before=cursor, after=cursor + int(wrote),
            pre=healthy if wrote else guarded, post=guarded if step >= count - 1 else healthy))
    return attempt_result(rows, writes_before=before)


def actor_receipt(outer, before, count):
    result = accounting(before, count)
    ranks = [dict(result, rank=rank, opportunities=20, actual_adam_steps=before + count, new_adam_steps=count)
             for rank in range(storage.WORLD)]
    return dict(role='prefix_selection200_actor_complete', run_id='constructed', completed_outer=outer,
        status='complete', complete_outer=True, actual_adam_steps_per_rank=[before + count]*storage.WORLD,
        new_adam_steps_per_rank=[count]*storage.WORLD, skipped_update_opportunities=[20-count]*storage.WORLD,
        opportunities_processed=20, full_budget_complete=count == 20, automatic_rollback_performed=False,
        scheduler_completed_outer=outer, rank_receipts=ranks)


class StorageGuardBoundaryTests(unittest.TestCase):
    def test_actual_counts_require20_opportunities_equal_ranks_and_exact_skips(self):
        receipt = actor_receipt(3, 4, 2)
        self.assertTrue(storage.actual_actor_boundary(receipt, 3))
        for field, value in [('actual_adam_steps_per_rank',[6,6,7,6]),
                ('new_adam_steps_per_rank',[0]*4), ('skipped_update_opportunities',[17]*4),
                ('opportunities_processed',19), ('full_budget_complete',True)]:
            with self.subTest(field=field):
                changed = dict(receipt, **{field:value})
                self.assertFalse(storage.actual_actor_boundary(changed, 3))
        changed = copy.deepcopy(receipt)
        changed['rank_receipts'][0]['opportunity_statuses'][0] = 'skipped_guard'
        self.assertFalse(storage.actual_actor_boundary(changed, 3))
        self.assertFalse(storage.actual_actor_boundary(actor_receipt(3, 1, 1), 3))

    def test_current_partial_boundary_can_prune_old_partial_actor(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve(); base = parent/'search-g1-prefix-selection200-guard-fixture'
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
            def payload(path):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'small constructed fixture')
                return dict(path=str(path), bytes=path.stat().st_size, sha256='a'*64)
            config = dict(base=str(base), run_id='constructed', source_commit='b'*40, outers=200, world_size=4,
                actor_hidden_native=True, probe_update_period=10, probe_updates_per_period=50,
                probe_updates_per_outer=0, probe_update_at_final=False)
            write(base/'preparation/launch-config.json', config)
            write(base/'logs/process-ledger.json', dict(run_id='constructed', jobs={}))
            actors = {}; before = 0
            for outer, count in enumerate((1,2,3), 1):
                attempt = base/'actor/outers'/('outer-%04d-attempt-abc'%outer); hf = attempt/'huggingface'
                receipt = actor_receipt(outer, before, count); before += count
                receipt.update(state_dir=str(attempt), state_path=str(attempt), model_dir=str(hf),
                    hf_model_dir=str(hf), actor_model_dir=str(hf), milestone=False,
                    hf_files={'model.safetensors':payload(hf/'model.safetensors')},
                    rank_states=[payload(attempt/('rank-%d.pt'%rank)) for rank in range(4)],
                    driver_state=payload(attempt/'driver.pt'))
                write(attempt/'complete.json', receipt); actors[outer] = receipt
            probe_payload = payload(base/'probe/latest.pt')
            probe = dict(run_id='constructed', completed_outer=0, actual_adam_steps_per_rank=[0]*4,
                         bytes=probe_payload['bytes'], sha256=probe_payload['sha256'])
            for outer in (1,2,3):
                root = base/'outers'/('outer-%06d'%outer); root.mkdir(parents=True)
                os.link(base/'probe/latest.pt', root/'probe-state-at-completion.pt')
                joint = dict(run_id='constructed', source_commit=config['source_commit'], outer=outer,
                    actor=actors[outer], probe=probe, probe_snapshot=str(root/'probe-state-at-completion.pt'),
                    rollout=dict(actor_model_dir=actors[max(1,outer-1)]['model_dir']))
                write(root/'complete.json', joint)
            root = base/'outers/outer-000003'
            write(base/'latest-joint.json', joint); write(base/'actor/latest.json', actors[3]); write(base/'probe/latest.json', probe)
            write(root/'probe-frozen.json', dict(status='PROBE_FROZEN_NO_OPTIMIZER_UPDATE', outer=3,
                  new_adam_steps_per_rank=0, actual_adam_steps_per_rank=[0]*4))
            write(root/'probe-score/manifest.json', dict(role='current_actor_native_hidden_features',
                  run_id='constructed', outer=3, count=12800, independent_encoder_forward=False))
            feature = payload(root/'prefix/scan/hidden-rank-0.pt')
            write(root/'prefix/scan/scan-complete.json', dict(actor_hidden_native=True,
                  run_id='constructed', actor_hidden_shards=[feature]))
            with patch.object(storage, 'REMOTE_PARENT', parent):
                self.assertEqual(storage.current_boundary(base, config, 3)[1]['actor']['actual_adam_steps_per_rank'], [6]*4)
                result = storage.prune_completed(base, 3)
            self.assertEqual(result['status'], 'COMPLETED_RUN_OWNED_ROTATING_TENSORS_PRUNED')
            self.assertFalse((Path(actors[1]['state_dir'])/'rank-0.pt').exists())
            self.assertFalse((Path(actors[1]['model_dir'])/'model.safetensors').exists())
            self.assertTrue((Path(actors[1]['state_dir'])/'complete.json').exists())
            self.assertTrue((Path(actors[2]['state_dir'])/'rank-0.pt').exists())
            self.assertTrue((Path(actors[3]['state_dir'])/'rank-0.pt').exists())
            self.assertFalse(Path(feature['path']).exists())


if __name__ == '__main__': unittest.main()

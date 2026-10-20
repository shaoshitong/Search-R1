"""CPU regressions for isolated event continuation; never real-model qualification."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from .event_candidate_loss import event_candidate_loss
from .event_finetune import (OBJECTIVE, REQUIRED_CODE, evaluation_stats,
                             expected_steps, position, retain_first_update,
                             summarized_stats, validate_config)


def groups():
    return [dict(qid='q%d' % i, labels=([1, 0, 0, 0, 0] if i < 6 else [0]*5))
            for i in range(12)]


def config(source_epoch=5, source_steps=15736):
    return dict(source_epoch=source_epoch, start_epoch=source_epoch+1,
                fit_epochs=source_epoch+2, source_steps=source_steps, seed=1729,
                fit_epoch_groups=67198, fit_world=4, prefix_tokens=20,
                question_max_tokens=273, pair_weight=1., pair_margin=1., lr=5e-6,
                weight_decay=0., ema_decay=.99, objective=OBJECTIVE,
                source_checkpoint=dict(epoch=source_epoch, steps=source_steps,
                                       completed_epoch=True, cursor=0),
                source_commit='a'*40, source_file_manifest={name:'b'*64 for name in REQUIRED_CODE})


class EventContinuationTests(unittest.TestCase):
    def test_config_accepts_any_completed_source_epoch_not_cp1(self):
        validate_config(config(5, 15736))
        validate_config(config(2, 11782))
        bad = config()
        bad['start_epoch'] = 2
        with self.assertRaises(AssertionError):
            validate_config(bad)
        bad = config()
        bad['source_checkpoint']['completed_epoch'] = False
        with self.assertRaises(AssertionError):
            validate_config(bad)
        bad = config()
        bad['prefix_tokens'] = 5
        with self.assertRaises(AssertionError):
            validate_config(bad)

    def test_step_count_and_pending_evaluation_do_not_skip_or_repeat_updates(self):
        c = config(5, 15736)
        batches = [[0, 1, 6, 7], [2, 3, 8, 9]]
        with mock.patch('reproduction.operations.qwen7b_warmup.event_finetune.plan', return_value=batches):
            self.assertEqual(expected_steps(c, groups(), 6, 1), 15737)
            pending = dict(epoch=6, cursor=2, completed_epoch=False, steps=15738,
                           stage='event_evaluation_pending', metrics=None)
            self.assertEqual(position(pending, c, groups()), (6, 2, 15738))
            sealed = dict(pending, cursor=0, completed_epoch=True,
                          stage='event_epoch_completed', metrics={'diagnostic':{}})
            self.assertEqual(position(sealed, c, groups()), (7, 0, 15738))
            self.assertEqual(expected_steps(c, groups(), 7, 1), 15739)
            with self.assertRaises(AssertionError):
                position(dict(sealed, steps=15737), c, groups())
            with self.assertRaises(AssertionError):
                position(dict(sealed, metrics=None), c, groups())

    def test_event_objective_is_not_reported_as_binary_bce(self):
        z = torch.tensor([[-1.6]*5, [.2, -.2, .4, -.4, .1]], dtype=torch.float64,
                         requires_grad=True)
        y = torch.tensor([[1]*5, [1, 0, 1, 0, 0]], dtype=torch.float64)
        mask = (y[:, :, None] == 1) & (y[:, None, :] == 0)
        value = event_candidate_loss(z, y, mask, 1., 1., .5)
        self.assertGreater(float(value['event_correction'].detach()), 0.)
        torch.testing.assert_close(value['group_event_nll'], value['group_bce']+value['event_correction'])
        torch.testing.assert_close(value['loss'], value['trajectory_bce']+value['group_event_nll']+value['pairwise_loss'])

    def test_single_pass_accumulator_matches_existing_metrics_and_chunking(self):
        from reproduction.prefix20_group_joint100.loss import joint_loss
        torch.manual_seed(7)
        z = torch.randn(6, 5, dtype=torch.float64)
        y = torch.tensor([[0]*5, [1]*5, [1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 0], [0]*5], dtype=torch.float64)
        masks = (y[:, :, None] == 1) & (y[:, None, :] == 0)
        masks[2, 0, 1] = False  # Preserve excluded indistinguishable-pair semantics.
        c = config()
        whole, _ = evaluation_stats(z, y, masks, c, .5)
        chunked = torch.zeros_like(whole)
        for start, stop in ((0, 1), (1, 4), (4, 6)):
            chunked += evaluation_stats(z[start:stop], y[start:stop], masks[start:stop], c, .5)[0]
        torch.testing.assert_close(whole, chunked, rtol=1e-12, atol=1e-12)
        summary = summarized_stats(whole)
        reference = joint_loss(z, y)
        self.assertAlmostEqual(summary['trajectory_bce'], float(reference['trajectory_bce']), places=12)
        self.assertAlmostEqual(summary['group_bce'], float(reference['group_bce']), places=12)
        positive = reference['mixed_label'].bool()
        predicted = reference['mixed_probability'] >= .5
        tp, tn = int((positive & predicted).sum()), int((~positive & ~predicted).sum())
        self.assertEqual(summary['confusion']['tp'], tp)
        self.assertEqual(summary['confusion']['tn'], tn)
        self.assertAlmostEqual(summary['mixed_balanced_accuracy'], .5*(tp/3+tn/3))
        self.assertAlmostEqual(summary['optimized_total_loss'], summary['trajectory_bce']+
                               summary['group_event_nll']+summary['pairwise_loss'])
        keep = positive
        mixed_all, _ = evaluation_stats(z[keep], y[keep], masks[keep], c, 1.)
        mixed_parts = torch.zeros_like(mixed_all)
        for i in range(len(z)):
            if bool(keep[i]):
                mixed_parts += evaluation_stats(z[i:i+1], y[i:i+1], masks[i:i+1], c, 1.)[0]
        torch.testing.assert_close(mixed_all, mixed_parts, rtol=1e-12, atol=1e-12)

    def test_first_update_metadata_recovery_does_not_repeat_training(self):
        c = config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'inflight.pt'
            checkpoint.write_bytes(b'synthetic small checkpoint')
            receipt = dict(path=str(checkpoint), epoch=6, cursor=1, steps=15737, sha256='mock-sha')
            retain_first_update(root, receipt, c)
            self.assertEqual(json.loads((root/'first-event-update.json').read_text())['steps'], 15737)
            self.assertEqual((root/'first-event-update.pt').stat().st_ino, checkpoint.stat().st_ino)
            retain_first_update(root, receipt, c)
            retain_first_update(root, dict(receipt, cursor=200, steps=15936), c)

    def test_toy_adam_resume_retains_moments_and_replays_exactly(self):
        torch.manual_seed(1729)
        model = torch.nn.Linear(3, 5).double()
        ema = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=0.)
        x = torch.tensor([[1., 2., 3.], [-2., .5, 1.]], dtype=torch.float64)
        y = torch.tensor([[1, 0, 1, 0, 0], [1]*5], dtype=torch.float64)
        mask = (y[:, :, None] == 1) & (y[:, None, :] == 0)
        def update():
            optimizer.zero_grad(set_to_none=True)
            event_candidate_loss(model(x), y, mask, 1., 1., .5)['loss'].backward()
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters()):
                    a.mul_(.99).add_(b, alpha=.01)
        update()
        checkpoint = dict(raw=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(ema.state_dict()),
                          optimizer=copy.deepcopy(optimizer.state_dict()), rng=torch.get_rng_state())
        answers = []
        for _ in range(2):
            model.load_state_dict(checkpoint['raw'])
            ema.load_state_dict(checkpoint['ema'])
            optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
            torch.set_rng_state(checkpoint['rng'])
            update()
            answers.append((copy.deepcopy(model.state_dict()), copy.deepcopy(ema.state_dict()),
                            copy.deepcopy(optimizer.state_dict())))
        for key in ('raw', 'ema'):
            for name in answers[0][0 if key == 'raw' else 1]:
                self.assertTrue(torch.equal(answers[0][0 if key == 'raw' else 1][name],
                                            answers[1][0 if key == 'raw' else 1][name]))
        for item in checkpoint['optimizer']['state'].values():
            self.assertEqual(int(item['step']), 1)
        for i in answers[0][2]['state']:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(answers[0][2]['state'][i][key], answers[1][2]['state'][i][key]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

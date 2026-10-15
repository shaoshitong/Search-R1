"""One small CPU fixture exercises actual warmup adoption and online semantics."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import torch

from search_g1.qprobe_token_model import TokenBinaryWarmup, token_binary_loss
from search_g1.qprobe_token_online import (
    OnlineTokenBinaryQProbe, TokenOnlineQProbeConfig, token_binary_group_loss,
)


class TinyTokenProvider:
    spec_sha = 'a'*64
    feature_dim = 3
    spec = dict(global_max_tokens=2)

    def __init__(self):
        self.ids = [f'q{i}' for i in range(5120)]
        self.index = {qid: dict(length=1+i%2, question_hash=hashlib.sha256(qid.encode()).hexdigest())
                      for i, qid in enumerate(self.ids)}
        self.calls, self.fail_after = 0, None

    def get_batch(self, qids, device='cpu'):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError('deliberate provider failure after actual Adam write')
        self.calls += 1
        lengths = torch.tensor([self.index[qid]['length'] for qid in qids])
        positions = torch.arange(int(lengths.max()))[None, :]
        mask = positions < lengths[:, None]
        features = torch.zeros(len(qids), int(lengths.max()), 3)
        features[:, :, 0] = torch.tensor([int(qid[1:])%6 for qid in qids])[:, None]/5
        features[:, :, 1] = positions
        features[:, :, 2] = 1.
        features.masked_fill_(~mask[:, :, None], float('nan'))
        return dict(features=features.to(device), attention_mask=mask.to(device))


def labels(provider, qids, outer, *, bootstrap=False):
    result = []
    for qid in qids:
        count = int(qid[1:])%6
        result.append(dict(qid=qid, question_hash=provider.index[qid]['question_hash'], complete=True,
            strict_em=[1]*count+[0]*(5-count), correct_count=count, source_outer=outer,
            selection_source='bootstrap_all_candidate' if bootstrap else 'predicted_mixed',
            run_id='teacher' if bootstrap else 'formal', attempt_id=f'attempt-{outer}',
            policy_snapshot_id=f'actor-before-{outer}', sampling_contract_sha='s'*64,
            strict_scoring_contract_sha='c'*64, retriever_version='fixed-retriever'))
    return result


def advance(head, provider, outer):
    qids = provider.ids[:512]
    return head.update_completed_outer(outer, labels(provider, qids, outer), qids,
        policy_snapshot_id=f'actor-before-{outer}', sampling_contract_sha='s'*64,
        strict_scoring_contract_sha='c'*64, retriever_version='fixed-retriever',
        run_id='formal', attempt_id=f'attempt-{outer}')


class TokenOnlineTests(unittest.TestCase):
    def test_actual_adoption_replay_weighting_roundtrip_and_explicit_rollback(self):
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            provider = TinyTokenProvider()
            config = TokenOnlineQProbeConfig(max_tokens=2, feature_dim=3)
            mapped = TokenOnlineQProbeConfig.from_mapping(dict(asdict(config),seed=20261013,head_seed=1729))
            self.assertEqual(mapped, config)
            global_rng = torch.get_rng_state().clone()
            warmup = TokenBinaryWarmup(config.token_config())
            qids = provider.ids
            counts = torch.tensor([int(qid[1:])%6 for qid in qids])
            sizes = [provider.index[qid]['length'] for qid in qids]
            metrics = warmup.fit(provider, sizes, counts, qids)
            self.assertEqual(metrics['actual_head_adam_updates'],200)
            self.assertEqual(len(metrics['minibatch_history']),200)
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
            with tempfile.TemporaryDirectory() as directory:
                raw = Path(directory)/'warmup.pt'
                torch.save(dict(token_feature_spec_sha256=provider.spec_sha,
                    label_producer_provenance=dict(protocol=dict(sampling_contract_sha='s'*64,
                        strict_scoring_contract_sha='c'*64,retriever_version='fixed-retriever')),
                    head=warmup.state_dict()),raw)
                digest = hashlib.sha256(raw.read_bytes()).hexdigest()
                corpus = labels(provider,qids,0,bootstrap=True)
                head = OnlineTokenBinaryQProbe.adopt_warmup(raw,corpus,qids,
                    expected_checkpoint_sha=digest,expected_feature_spec_sha=provider.spec_sha,
                    provenance=dict(actor_snapshot_id='actor:0',warmup_ready_sha256='b'*64),
                    config=config,feature_cache=provider)
                self.assertEqual((head.completed_outer,head.update_count,head._adam_steps()),(0,0,200))
                self.assertEqual(len(head.replay_labels),5120)
                self.assertTrue(head.initialization_provenance['no_refit'])
                torch.testing.assert_close(head.rng.get_state(),warmup.rng.get_state(),rtol=0,atol=0)
                for key,value in warmup.model.state_dict().items():
                    torch.testing.assert_close(head.model.state_dict()[key],value,rtol=0,atol=0)
                corpus[0]['strict_em'][0] = 1  # Caller mutation cannot change recorded labels.
                self.assertEqual(head.replay_labels[0]['strict_em'],[0]*5)
                original_provenance = head.initialization_provenance
                original_provenance['no_refit'] = False
                self.assertTrue(head.initialization_provenance['no_refit'])
                prediction = head.predict(qids[:3])
                self.assertEqual(set(prediction),{'mixed_probability','predicted_sample_std'})
                self.assertTrue(all(not value.requires_grad for value in prediction.values()))

                # Same unreduced objective as immutable warmup, including its gradients.
                batch = provider.get_batch(qids[:3]); lengths = torch.tensor(sizes[:3])
                output = head.model(batch['features'],lengths)
                per,parts = token_binary_group_loss(output,counts[:3],config)
                reference,reference_parts = token_binary_loss(output,counts[:3],config.token_config())
                torch.testing.assert_close(per.mean(),reference)
                for key in parts: torch.testing.assert_close(parts[key].mean(),reference_parts[key])
                grad, = torch.autograd.grad(per.mean(),head.model.output.weight,retain_graph=True)
                reference_grad, = torch.autograd.grad(reference,head.model.output.weight)
                torch.testing.assert_close(grad,reference_grad)

                result = advance(head,provider,1)
                self.assertEqual((result['fit_epochs'],result['head_optimizer_updates']),(2,44))
                self.assertEqual((result['head_adam_steps_before'],result['head_adam_steps_after']),(200,244))
                self.assertEqual(len(head.replay_features),5632)
                # Compare a mixed-age mini against the full-replay global denominator.
                expected_rng = torch.Generator().set_state(warmup.rng.get_state())
                first = torch.randperm(5632,generator=expected_rng)[:256]
                raw_weights = torch.tensor([2**(-1/4)]*5120+[1.]*512)
                normalized = raw_weights*(5632/raw_weights.sum())
                with torch.no_grad():
                    old_engine = OnlineTokenBinaryQProbe(3,config,provider.spec_sha,feature_cache=provider)
                    old_engine.model.load_state_dict(warmup.model.state_dict())
                    ids = qids+qids[:512]
                    _,lens = old_engine._resolve(ids)
                    tokens,token_lengths = old_engine._engine._batch(provider,first,lens,ids)
                    old_prediction = old_engine.model(tokens,token_lengths)
                    group_losses,_ = token_binary_group_loss(old_prediction,torch.cat([counts,counts[:512]])[first],config)
                    expected_loss = (group_losses*normalized[first]).mean()
                self.assertAlmostEqual(result['minibatch_history'][0]['pre_step_weighted_loss'],float(expected_loss),places=6)
                for row in result['minibatch_history']:
                    self.assertAlmostEqual(row['pre_step_weighted_loss'],row['loss_parts']['binary_bce']+
                                           .3*row['loss_parts']['regression_mse_normalized'],places=6)

                # At16 outer0 expires; at17 outer1 expires, respecting the16-outer window.
                advance(head,provider,16)
                self.assertEqual({row['source_outer'] for row in head.replay_labels},{1,16})
                advance(head,provider,17)
                self.assertEqual({row['source_outer'] for row in head.replay_labels},{16,17})
                checkpoint = Path(directory)/'formal.pt'
                receipt = head.save(checkpoint,actor_snapshot_id='actor:17')
                self.assertEqual(receipt['head_version'],head.version)
                self.assertEqual(receipt['replay_identity_sha256'],head.replay_identity_sha256)
                restored = OnlineTokenBinaryQProbe.load(checkpoint,expected_actor_snapshot_id='actor:17',
                    expected_feature_spec_sha=provider.spec_sha,expected_checkpoint_sha=receipt['sha256'])
                with self.assertRaisesRegex(ValueError,'NOT_ATTACHED'): restored.predict(qids[:2])
                restored.attach_feature_provider(provider)
                self.assertEqual(restored.version,head.version)
                self.assertEqual(restored.replay_labels,head.replay_labels)
                torch.testing.assert_close(restored.rng.get_state(),head.rng.get_state(),rtol=0,atol=0)
                for key,value in head.predict(qids[:2]).items():
                    torch.testing.assert_close(restored.predict(qids[:2])[key],value,rtol=0,atol=0)
                a,b = advance(head,provider,18),advance(restored,provider,18)
                self.assertEqual(a['minibatch_history'],b['minibatch_history'])
                self.assertEqual(head.version,restored.version)

                # Reject labels from a different immutable question or protocol before fitting.
                before = head.state_dict(); wrong = labels(provider,qids[:512],19)
                wrong[0]['question_hash'] = 'f'*64
                with self.assertRaisesRegex(ValueError,'QUESTION_HASH'):
                    head.update_completed_outer(19,wrong,qids[:512],policy_snapshot_id='actor-before-19',
                        sampling_contract_sha='s'*64,strict_scoring_contract_sha='c'*64,
                        retriever_version='fixed-retriever',run_id='formal',attempt_id='attempt-19')
                self.assertEqual(head._adam_steps(),before['head_adam_updates'])
                # Actual successful partial updates survive failure; only explicit caller rollback undoes them.
                provider.calls,provider.fail_after = 0,1
                with self.assertRaisesRegex(RuntimeError,'after actual Adam'): advance(head,provider,19)
                self.assertEqual(head._adam_steps(),before['head_adam_updates']+1)
                self.assertEqual(head._engine.update_count,head._adam_steps())
                self.assertTrue(head.state_dict()['fit_failed'])
                self.assertEqual(head.completed_outer,18)
                provider.fail_after = None
                head.load_state_dict(before)
                self.assertEqual(head._adam_steps(),before['head_adam_updates'])
                self.assertEqual(head.replay_identity_sha256,before['replay_identity_sha256'])
                self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
                self.assertEqual(advance(head,provider,19)['minibatch_history'],
                                 advance(restored,provider,19)['minibatch_history'])
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()

"""CPU tiny-Qwen equality checks; no remote access or model download.

Run in the existing training environment:
    python -m unittest search_g1.test_selected_qwen_head -v
These tests do not claim FSDP/CUDA acceptance or physical-batch capacity.
"""
import copy
import importlib.util
import unittest


DEPENDENCIES = (importlib.util.find_spec('torch') is not None
                and importlib.util.find_spec('transformers') is not None)


@unittest.skipUnless(DEPENDENCIES, 'Existing torch+transformers environment required; no downloads')
class SelectedQwenHeadTest(unittest.TestCase):
    def make_models(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from search_g1.selected_qwen_head import install_selected_qwen_head
        torch.manual_seed(918)
        config = Qwen2Config(vocab_size=61, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0.0, use_cache=False,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
        dense = Qwen2ForCausalLM(config).double()
        selected = copy.deepcopy(dense)
        install_selected_qwen_head(selected)
        return dense, selected

    def inputs(self):
        import torch
        ids = torch.tensor([[0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                            [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0]])
        mask = (ids != 0).long()
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        # Noncontinuous policy prediction positions, interleaved with tool text.
        indices = torch.tensor([2, 3, 6, 8, 9], dtype=torch.long)
        authored = torch.tensor([[1., 1., 0., 1., 1.], [1., 0., 1., 1., 0.]], dtype=torch.float64)
        return ids, mask, positions, indices, authored

    def test_selected_log_probs_weighted_loss_and_every_parameter_gradient(self):
        import torch
        dense, selected = self.make_models()
        ids, mask, positions, indices, authored = self.inputs()
        baseline = dense(input_ids=ids, attention_mask=mask, position_ids=positions,
                         use_cache=False, return_dict=True).logits.index_select(1, indices)
        narrowed = selected(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False, return_dict=True, prediction_positions=indices).logits
        self.assertEqual(narrowed.shape, baseline.shape)
        labels = ids.index_select(1, indices + 1)
        base_lp = baseline.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        fast_lp = narrowed.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(fast_lp, base_lp, rtol=1e-6, atol=1e-7)
        old = base_lp.detach() + torch.tensor([[.04, -.08, .2, -.1, .3], [-.03, .05, .1, .2, -.06]])
        advantage = torch.tensor([[1., -.3, .8, -.7, .4], [-.4, .9, -.6, .2, -.5]], dtype=torch.float64)
        weights = authored * torch.tensor([[.5, 1., 1.5, .75, .25], [1., .25, .5, .75, 1.25]], dtype=torch.float64)

        def ppo_loss(log_prob):
            ratio = (log_prob - old).exp()
            loss = torch.maximum(-advantage * ratio,
                                 -advantage * ratio.clamp(.8, 1.2))
            loss = torch.where(advantage < 0, torch.minimum(loss, -3. * advantage), loss)
            return (loss * weights).sum() / weights.sum()

        loss_dense, loss_selected = ppo_loss(base_lp), ppo_loss(fast_lp)
        torch.testing.assert_close(loss_selected, loss_dense, rtol=1e-6, atol=1e-7)
        loss_dense.backward(); loss_selected.backward()
        first, second = dict(dense.named_parameters()), dict(selected.named_parameters())
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            with self.subTest(parameter=name):
                self.assertIsNotNone(first[name].grad)
                self.assertIsNotNone(second[name].grad)
                torch.testing.assert_close(second[name].grad, first[name].grad, rtol=1e-5, atol=2e-7)
        self.assertEqual(dense.state_dict().keys(), selected.state_dict().keys())

    def test_checkpointed_decoder_unchanged_and_dense_path_restored(self):
        import torch
        dense, selected = self.make_models()
        dense.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        selected.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        dense.train(); selected.train()
        ids, mask, positions, indices, _ = self.inputs()
        kwargs = dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                      use_cache=False, return_dict=True)
        reference = dense(**kwargs).logits.index_select(1, indices)
        narrowed = selected(**kwargs, prediction_positions=indices).logits
        reference.square().mean().backward(); narrowed.square().mean().backward()
        for (name, before), (other, after) in zip(dense.named_parameters(), selected.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(after.grad, before.grad, rtol=1e-5, atol=2e-7)
        torch.testing.assert_close(selected(**kwargs).logits, dense(**kwargs).logits, rtol=1e-6, atol=1e-7)
        self.assertFalse(selected._selected_qwen_head_active)
        self.assertEqual(len(selected.lm_head._forward_pre_hooks), 0)

    def test_invalid_positions_or_cache_leave_no_hook(self):
        import torch
        _, model = self.make_models()
        ids, mask, positions, _, _ = self.inputs()
        kwargs = dict(input_ids=ids, attention_mask=mask, position_ids=positions, return_dict=True)
        for indices in (torch.tensor([], dtype=torch.long), torch.tensor([3, 2]), torch.tensor([2, 2]),
                        torch.tensor([-1]), torch.tensor([999])):
            with self.subTest(indices=indices.tolist()):
                with self.assertRaises(ValueError):
                    model(**kwargs, prediction_positions=indices)
                self.assertFalse(getattr(model, '_selected_qwen_head_active', False))
                self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)
        with self.assertRaises(ValueError):
            model(**kwargs, prediction_positions=torch.tensor([2]), use_cache=True)


if __name__ == '__main__':
    unittest.main()

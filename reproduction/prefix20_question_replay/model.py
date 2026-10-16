"""Contextual question plus twenty policy positions, one copied Qwen block, BCE logit.

The physical layout is [Qmax question slots, 20 policy slots]. Question slots
are right padded, but the policy slots stay at the tail so checkpoint migration
can preserve every learned prefix coefficient and Adam moment exactly. Actual
RoPE positions are independent of these storage slots and remain unchanged.
"""
from collections.abc import Mapping

import torch
from torch import nn

from reproduction.prefix20_online.model import PrefixCorrectnessProbe, validate_prefix_positions


POLICY_TOKENS = 20
HIDDEN_SIZE = 2048


def masked_flatten(hidden, mask):
    """Padding contributes exactly zero to the flattened linear input."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Hidden states and boolean attention mask have incompatible shapes')
    return hidden.masked_fill(~mask.unsqueeze(-1), 0).flatten(1)


def validate_question_policy_layout(batch, question_max_tokens, max_positions):
    """Accept the intentional gap between short question prompts and fixed policy slots."""
    features = batch['features']
    mask = batch['attention_mask']
    positions = batch['position_ids']
    lengths = batch['question_lengths']
    question_mask = batch['question_mask']
    policy_mask = batch['policy_mask']
    width = question_max_tokens + POLICY_TOKENS
    if (features.ndim != 3 or features.shape[1:] != (width, HIDDEN_SIZE)
            or mask.shape != features.shape[:2] or mask.dtype != torch.bool
            or positions.shape != mask.shape or positions.dtype != torch.long):
        raise ValueError('Require fixed question+twenty-policy feature, mask, and position shapes')
    if lengths.shape != (features.shape[0],) or lengths.dtype != torch.long:
        raise ValueError('One actual prompt length is required per trajectory')
    if (question_mask.shape != mask.shape or question_mask.dtype != torch.bool
            or policy_mask.shape != mask.shape or policy_mask.dtype != torch.bool):
        raise ValueError('Question and policy slot masks must span the complete feature sequence')
    if bool(((lengths < 1) | (lengths > question_max_tokens)).any()):
        raise ValueError('Actual prompt length must fit the scanned question maximum')
    question_slots = torch.arange(width, device=mask.device).unsqueeze(0)
    expected_question = question_slots < lengths.unsqueeze(1)
    if not torch.equal(question_mask, expected_question):
        raise ValueError('Question prompt must occupy the initial right-padded question slots')
    if bool(policy_mask[:, :question_max_tokens].any()):
        raise ValueError('Policy features cannot occupy question slots')
    if not torch.equal(mask, question_mask | policy_mask):
        raise ValueError('Combined attention mask differs from the actual question and policy masks')
    expected_positions = question_slots.expand(features.shape[0], -1)
    if not torch.equal(positions[question_mask], expected_positions[question_mask]):
        raise ValueError('Question RoPE positions must preserve actual prompt token positions')
    validate_prefix_positions(policy_mask[:, question_max_tokens:],
                              positions[:, question_max_tokens:],
                              batch['policy_segments'], max_positions)


class QuestionPrefixCorrectnessProbe(PrefixCorrectnessProbe):
    """The complete actual prompt and retained policy tokens enter the same head block."""

    def __init__(self, model_dir, question_max_tokens):
        # Reuse the existing exact official last-block copy and BF16 FA2 backend.
        super().__init__(model_dir, max_tokens=POLICY_TOKENS)
        if type(question_max_tokens) is not int or not 1 <= question_max_tokens < self.max_positions:
            raise ValueError('Question maximum must be an actual scanned positive token length')
        self.question_max_tokens = question_max_tokens
        self.policy_tokens = POLICY_TOKENS
        self.hidden_size = HIDDEN_SIZE
        self.max_tokens = question_max_tokens + POLICY_TOKENS
        self.output = nn.Linear(self.max_tokens * HIDDEN_SIZE, 1)
        # This experiment always restores the latest learned head. Zero is also
        # the explicit initialization for the newly introduced question columns.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.copy_metadata.update(shared_max_tokens=self.max_tokens,
                                  question_max_tokens=question_max_tokens,
                                  policy_tokens=POLICY_TOKENS,
                                  head_input='actual_question_and_retained_policy_positions',
                                  question_hidden_passed_to_head=True,
                                  feature_layout='question_right_pad_then_fixed_twenty_policy_tail')

    def forward(self, batch):
        validate_question_policy_layout(batch, self.question_max_tokens, self.max_positions)
        features = batch['features'].detach()
        mask = batch['attention_mask']
        positions = batch['position_ids']
        kwargs = dict(attention_mask=mask, position_ids=positions,
                      position_embeddings=self.rotary(features, positions),
                      use_cache=False, output_attentions=False)
        if self._cache_position:
            kwargs['cache_position'] = torch.arange(features.shape[1], device=features.device)
        hidden = self.block(features, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.output(masked_flatten(hidden, mask)).squeeze(-1).float()


def _expand_policy_tail(tensor, target_shape, policy_width):
    if tuple(tensor.shape) != (1, policy_width) or len(target_shape) != 2 or target_shape[0] != 1:
        raise ValueError('Only the old flattened policy Linear weight may be expanded')
    if target_shape[1] <= policy_width:
        raise ValueError('Question-aware Linear must add at least one question position')
    expanded = tensor.new_zeros(target_shape)
    expanded[:, -policy_width:] = tensor
    return expanded


def migrate_checkpoint(saved, target_model, *, learning_rate=3e-6):
    """Expand the saved learned Linear and its moments; preserve all other state.

    The authorized new experiment changes the source LR3e-5 to target LR3e-6
    after retaining the complete Adam state. No source object is mutated.
    Unmodified checkpoint tensors are deliberately
    shared in the returned CPU checkpoint, avoiding a second copy of the large
    official block and its Adam state. Call normal model/optimizer load methods,
    then restore the original per-rank RNG exactly as in the existing trainer.
    This preserves learned parameters and Adam history, not identical logits:
    the head block now also attends the question, as the user requested.
    """
    source_model = saved['model']
    source_optimizer = saved['optimizer']
    target_state = target_model.state_dict()
    policy_tokens = int(getattr(target_model, 'policy_tokens', POLICY_TOKENS))
    hidden_size = int(getattr(target_model, 'hidden_size', HIDDEN_SIZE))
    policy_width = policy_tokens * hidden_size
    if policy_tokens != POLICY_TOKENS or hidden_size != HIDDEN_SIZE:
        raise ValueError('Checkpoint migration requires twenty official 2048-dimensional policy positions')
    if set(source_model) != set(target_state):
        raise ValueError('Saved and target head state keys differ')
    if source_model['output.weight'].dtype != torch.float32:
        raise ValueError('Learned Linear master weights must remain FP32')
    for name, target in target_state.items():
        if name != 'output.weight' and source_model[name].shape != target.shape:
            raise ValueError('Only the flattened Linear input width may change: ' + name)
    parameter_names = [name for name, _ in target_model.named_parameters()]
    groups = source_optimizer['param_groups']
    source_ids = [identity for group in groups for identity in group['params']]
    if len(source_ids) != len(parameter_names) or len(set(source_ids)) != len(source_ids):
        raise ValueError('Saved optimizer parameter ordering is incomplete or duplicated')
    if set(source_optimizer['state']) != set(source_ids):
        raise ValueError('Every learned head parameter must retain its existing Adam state')
    source_learning_rate = 3e-5
    if learning_rate != 3e-6 or any(float(group['lr']) != source_learning_rate for group in groups):
        raise ValueError('This authorized migration restores source LR3e-5 and applies target LR3e-6')
    actual_steps = saved['actual_adam_steps_per_rank']
    if len(actual_steps) != 8 or len(set(actual_steps)) != 1 or actual_steps[0] < 1:
        raise ValueError('Source must have a complete synchronized eight-rank Adam boundary')
    name_to_id = dict(zip(parameter_names, source_ids))
    for group in groups:
        if 'param_names' in group:
            actual_names = [parameter_names[source_ids.index(identity)] for identity in group['params']]
            if group['param_names'] != actual_names:
                raise ValueError('Saved optimizer names disagree with model parameter order')
    for name, identity in name_to_id.items():
        state = source_optimizer['state'][identity]
        if not isinstance(state, Mapping) or 'exp_avg' not in state or 'exp_avg_sq' not in state or 'step' not in state:
            raise ValueError('Existing complete Adam moments and step are required: ' + name)
        step = state['step']
        value = step.item() if isinstance(step, torch.Tensor) else step
        if value != actual_steps[0]:
            raise ValueError('Source Adam parameter steps disagree with the actual rank boundary')
        for moment in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if moment in state and (state[moment].shape != source_model[name].shape
                                    or state[moment].dtype != torch.float32):
                raise ValueError('Saved FP32 Adam moment shape differs: ' + name + '/' + moment)
    migrated = dict(saved)
    migrated['model'] = dict(source_model)
    migrated['model']['output.weight'] = _expand_policy_tail(
        source_model['output.weight'], target_state['output.weight'].shape, policy_width)
    weight_identity = name_to_id['output.weight']
    migrated_optimizer = dict(source_optimizer)
    migrated_optimizer['state'] = dict(source_optimizer['state'])
    migrated_weight_state = dict(source_optimizer['state'][weight_identity])
    for moment in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if moment in migrated_weight_state:
            migrated_weight_state[moment] = _expand_policy_tail(
                migrated_weight_state[moment], target_state['output.weight'].shape, policy_width)
    migrated_optimizer['state'][weight_identity] = migrated_weight_state
    migrated_optimizer['param_groups'] = [dict(group, params=list(group['params']), lr=learning_rate)
                                          for group in groups]
    migrated['optimizer'] = migrated_optimizer
    proof = dict(schema_version=1, source_epoch=saved['epoch'],
                 source_actual_adam_steps_per_rank=list(actual_steps),
                 source_linear_shape=list(source_model['output.weight'].shape),
                 target_linear_shape=list(target_state['output.weight'].shape),
                 old_policy_weight_location='tail_twenty_positions_exact_copy',
                 new_question_weights='zero_initialized',
                 new_question_adam_moments='zero_initialized',
                 block_weights_and_adam='unchanged', output_bias_and_adam='unchanged',
                 source_learning_rate=source_learning_rate, target_learning_rate=learning_rate,
                 learning_rate=learning_rate, learning_rate_override_applied=True,
                 rank_rng='unchanged_saved_state',
                 identical_logits_claimed=False)
    migrated['question_representation_transition'] = proof
    return migrated, proof
